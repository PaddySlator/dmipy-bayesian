import numpy as np
from copy import copy, deepcopy
import scipy
import matplotlib.pyplot as plt


# NOTE: will need fix if other models have parameters with cardinality > 1 (other than orientation)
def tform_params(param_dict, parameter_names, model, direction):
    param_dict = deepcopy(param_dict)  # because dicts are mutable, and don't want to alter inside function
    if direction == 'f':
        for param in parameter_names:
            if '_mu' not in param:  # don't transform orientation parameters
                # NB. Add/subtract 1e-5 to avoid nan/inf if parameter is on upper/lower bound from LS fit. Transformed
                # parameters shouldn't ever reach bounds (i.e. from sampling in Metropolis Hastings step)
                lb = (model.parameter_ranges[param][0] - 1e-5) * model.parameter_scales[param]  # lower bound
                ub = (model.parameter_ranges[param][1] + 1e-5) * model.parameter_scales[param]  # upper bound
                param_dict[param] = np.log(param_dict[param] - lb) - np.log(ub - param_dict[param])

    elif direction == 'r':
        for param in parameter_names:
            if '_mu' not in param:  # don't transform orientation parameters
                # NB. Add/subtract 1e-5 to avoid nan/inf if parameter is on upper/lower bound from LS fit. Transformed
                # parameters shouldn't ever reach bounds (i.e. from sampling in Metropolis Hastings step)
                lb = (model.parameter_ranges[param][0] - 1e-5) * model.parameter_scales[param]  # lower bound
                ub = (model.parameter_ranges[param][1] + 1e-5) * model.parameter_scales[param]  # upper bound
                param_dict[param] = (lb + ub * np.exp(param_dict[param])) / (1 + np.exp(param_dict[param]))
                param_dict[param] = [ub if np.isnan(x) else x for x in param_dict[param]]

    else:
        print('Incorrect input! Nothing is happening...')

    return param_dict


# FIXME: create as class like modelling_framework.py?
def fit(model, acq_scheme, data, mask=None, nsteps=1000, burn_in=500):
    # FIXME: data checks?

    # set mask default
    if mask is None:
        mask = (data[:, 0] > 0).astype('uint8')
    # convert to int if input as bool
    elif mask.any():
        mask = mask.astype('uint8')

    # extract useful values
    nvox = np.prod(mask.shape)  # np.sum(mask > 0)  # number of voxels in mask
    nparams = np.sum(np.array(list(model.parameter_cardinality.values())))
    ndw = len(acq_scheme.bvalues)
    ncomp = model.partial_volume_names.__len__()

    # extract ROIs if present in mask; check enough voxels in each ROI to avoid df error in sigma calculation
    roi_vals = np.unique(mask)[np.unique(mask) > 0]  # list of unique integers that identify each ROI (ignore 0's)
    roi_nvox = [[xx for xx, x in enumerate(mask == roi_vals[roi]) if x].__len__() for roi in range(roi_vals.__len__())] # number of voxels in each ROI
    to_remove = [roi for roi in range(roi_vals.__len__()) if roi_nvox[roi] < 2 * nparams] # indices of ROIs with too few voxels
    roi_vals = np.delete(roi_vals, to_remove)
    nroi = roi_vals.__len__()  # no. ROIs

    # do initial LQS fit for parameter initialisation
    lsq_fit = model.fit(acq_scheme, data, mask=mask)

    # get number of compartments; only fit no. compartments - 1
    parameters_to_fit = [name for name in model.parameter_names if name != model.partial_volume_names[-1]]
    dependent_fraction = model.partial_volume_names[-1]

    # remove dependent volume fraction from model
    model_reduced = deepcopy(model)
    del model_reduced.parameter_ranges[dependent_fraction]
    del model_reduced.parameter_cardinality[dependent_fraction]
    del model_reduced.parameter_scales[dependent_fraction]
    del model_reduced.parameter_types[dependent_fraction]
    del model_reduced.parameter_optimization_flags[dependent_fraction]

    # create dictionary of model parameter names and LSQ fit values
    init_param = dict.fromkeys(model.parameter_names)
    for param in model.parameter_names:
        init_param[param] = np.squeeze(np.nan * np.ones([nvox, model.parameter_cardinality[param]]))
        init_param[param][mask > 0, ] = lsq_fit.fitted_parameters[param][mask > 0]

    # TODO: remove perturbations from final version
    # perturb params for testing
    idx_roi = [xx for xx, x in enumerate(mask == roi_vals[0]) if x]
    vox = idx_roi[0]
    init_param['C1Stick_1_lambda_par'][vox] = (model.parameter_ranges['C1Stick_1_lambda_par'][1]
                                               - model.parameter_ranges['C1Stick_1_lambda_par'][1] / 50)\
                                              * model.parameter_scales['C1Stick_1_lambda_par']             # Dpar
    init_param['G1Ball_1_lambda_iso'][vox] = (model.parameter_ranges['G1Ball_1_lambda_iso'][1]
                                              - model.parameter_ranges['G1Ball_1_lambda_iso'][1] / 50)\
                                             * model.parameter_scales['G1Ball_1_lambda_iso']               # Diso
    init_param['partial_volume_0'][vox] = .5                  # fpar
    fig, ax = plt.subplots()

    # create dict of LSQ fit values, original values
    params_all_orig = deepcopy(init_param)
    # create dict of LSQ fit values, log transform variables (non-orientation only) (original -> log)
    params_all_tform = tform_params(init_param, model.parameter_names, model, 'f')

    # TODO: play with weights created from ranges - affects convergence
    # initial weights for Metropolis-Hastings parameter sampling
    w = dict.fromkeys(parameters_to_fit)
    for param in parameters_to_fit:  # get scaled parameter ranges
        w[param] = (np.array(model.parameter_ranges[param])) * model.parameter_scales[param]
    w = tform_params(w, parameters_to_fit, model, 'f')  # transform parameter ranges
    for param in parameters_to_fit:  # set weight as x * range
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                w[param][card] = 0.005 * np.abs(np.subtract(w[param][card, 1], w[param][card, 0]))
            w[param] = w[param][range(model.parameter_cardinality[param]), 0]
            w[param] = np.tile(w[param], (nvox, 1))  # tile to create weight for each voxel
        elif model.parameter_cardinality[param] == 1:
            w[param] = 0.01 * np.abs(np.subtract(w[param][1], w[param][0]))
            w[param] = np.tile(w[param], nvox)  # tile to create weight for each voxel


    # w = deepcopy(init_param)
    # for param in parameters_to_fit:
    #     if 'C1Stick_1_mu' in param:
    #         w[param] = w[param] / 100
    #     else:
    #         w[param] = w[param] / 10                    # initial w = starting value / 10 (Gustafsson et al., 2018)
    # w = tform_params(w, model.parameter_names, model, 'f')  # transform
    # for param in parameters_to_fit:
    #     w[param] = np.abs(w[param])


    # measured signals
    E_dat = data  # [mask > 0]

    # initial LSQ-estimated signals
    E_fit = np.array([model.simulate_signal(acq_scheme, model.parameters_to_parameter_vector(**params_all_orig)[i, :])
                      for i in range(nvox)])

    # initialise variables to track state of optimisation at each step (each parameter tracked independently)
    accepted = dict.fromkeys(parameters_to_fit)                         # total accepted moves over all steps
    accepted_per_100 = dict.fromkeys(parameters_to_fit)                 # accepted moves per 100 steps (updates weights)
    acceptance_rate = dict.fromkeys(parameters_to_fit)                  # accepted moves at each step
    param_conv = dict.fromkeys(parameters_to_fit)                       # parameter convergence
    gibbs_mu = np.zeros((nroi, nparams - 1, nsteps))                    # gibbs mu at each step
    gibbs_sigma = np.zeros((nroi, nparams - 1, nparams - 1, nsteps))    # gibbs sigma at each step
    likelihood_stored = dict.fromkeys(parameters_to_fit)                # likelihood at each step
    w_stored = dict.fromkeys(parameters_to_fit)                         # weights at each weight update

    # initialise dictionaries (param_conv, accepted, accepted_per_100, acceptance_rate)
    for param in parameters_to_fit:
        accepted[param] = np.squeeze(np.zeros((nvox, model.parameter_cardinality[param])))
        accepted_per_100[param] = np.squeeze(np.zeros((nvox, model.parameter_cardinality[param])))
        acceptance_rate[param] = np.squeeze(np.zeros((nvox, model.parameter_cardinality[param], nsteps)))
        param_conv[param] = np.squeeze(np.zeros((nvox, model.parameter_cardinality[param], nsteps)))
        likelihood_stored[param] = np.squeeze(np.zeros((nvox, model.parameter_cardinality[param], nsteps)))
        w_stored[param] = np.squeeze(np.zeros((nvox, model.parameter_cardinality[param], np.int(np.floor(burn_in/200)+1))))
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                w_stored[param][:, card, 0] = w[param][:, card]
        else:
            w_stored[param][:, 0] = w[param]

    #------------------------------------------- MCMC ------------------------------------------------------------------
    # loop over ROIs
    for roi in range(nroi):
        idx_roi = [xx for xx, x in enumerate(mask == roi_vals[roi]) if x]  # indices into mask of voxels in ROI
        nvox_roi = idx_roi.__len__()  # no. voxels in ROI
        # initialise sigma for this ROI
        sigma = np.cov(np.transpose(model_reduced.parameters_to_parameter_vector(**params_all_tform)[idx_roi]))
        print("ROI " + str(roi+1) + "/" + str(nroi) + "; " + str(nvox_roi) + " voxels")

        # NB i (voxel loop) and j (MC loop) in keeping with Orton paper
        for j in range(0, nsteps):
            print(j)
            # Gibbs moves to update priors.
            # Gibbs 1. sample mu from multivariate normal dist defined by current param estimates.
            parameter_vector = model_reduced.parameters_to_parameter_vector(**params_all_tform)[idx_roi, :]
            m = np.mean(parameter_vector, axis=0)
            V = sigma / nvox_roi
            mu = np.random.multivariate_normal(m, V)
            # Gibbs 2. sample sigma from inverse Wishart distribution (using newly updated mu)
            phi = np.sum([np.outer(parameter_vector[i, :] - mu, parameter_vector[i, :] - mu)
                          for i in range(0, nvox_roi)], axis=0)
            sigma = scipy.stats.invwishart(scale=phi, df=nvox_roi - nparams - 1).rvs()

            # save Gibbs parameters for this step (careful of parameter ordering)
            gibbs_mu[roi, :, j] = copy(mu)
            gibbs_sigma[roi, :, :, j] = copy(sigma)

            # Metropolis-Hastings parameter updates
            params_all_new = deepcopy(params_all_tform)
            # NOTE: both direction parameters are updated at the same time
            for p in parameters_to_fit:
                # sample parameter
                if model.parameter_cardinality[p] > 1:
                    for card in range(model.parameter_cardinality[p]):
                        params_all_new[p][idx_roi, card] = np.random.normal(params_all_tform[p][idx_roi, card],
                                                                            w[p][idx_roi, card])
                elif model.parameter_cardinality[p] == 1:
                    params_all_new[p][idx_roi] = np.random.normal(params_all_tform[p][idx_roi],
                                                                  np.matrix.flatten(w[p][idx_roi]))

                # if a volume fraction was sampled, re-compute dependent fraction
                # create boolean prior to avoid sum(volume fractions) > 1
                if 'partial_volume_' in p:
                    f_indep = [params_all_new[name][idx_roi] for name in model.partial_volume_names
                               if name != dependent_fraction]
                    f_indep = np.exp(f_indep) / (1 + np.exp(f_indep))  # tform indept fractions (log -> orig)
                    for c in range(ncomp - 1):  # check transform
                        f_indep[c, :] = [1 if np.isnan(x) else x for x in f_indep[c, :]]
                    prior_new = 1 * (np.sum(f_indep, axis=0) < 1)  # boolean prior to control total vol frac
                    f_dept = np.array([np.max([0, 1 - np.sum(f_indep[:, f], axis=0)])
                                       for f in range(nvox_roi)])  # compute dept fraction
                    f_dept = np.log(f_dept) - np.log(1 - f_dept)  # tform dept fraction (orig -> log)
                    params_all_new[dependent_fraction][idx_roi] = f_dept
                else:
                    prior_new = np.log(np.ones(nvox_roi))  # dummy prior otherwise

                # compute acceptance
                y = copy(E_dat[idx_roi, :])  # actual measured signal
                g = copy(E_fit[idx_roi, :])  # model-predicted signal (old params)

                parameter_vector = tform_params(params_all_new, model.parameter_names, model, 'r')
                parameter_vector = model.parameters_to_parameter_vector(**parameter_vector)  # [i, :]
                g_new = model.simulate_signal(acq_scheme, parameter_vector[idx_roi, :])  # model-predicted signal (new params)

                # calculate posteriors and PDFs (log scale)
                inner_y_y = np.sum(np.multiply(np.squeeze(y), np.squeeze(y)), 1)
                inner_y_g = np.sum(np.multiply(np.squeeze(y), np.squeeze(g)), 1)
                inner_g_g = np.sum(np.multiply(np.squeeze(g), np.squeeze(g)), 1)
                inner_y_gnew = np.sum(np.multiply(np.squeeze(y), np.squeeze(g_new)), 1)
                inner_gnew_gnew = np.sum(np.multiply(np.squeeze(g_new), np.squeeze(g_new)), 1)
                likelihood = (-ndw / 2) * np.log(inner_y_y - (inner_y_g ** 2 / inner_g_g))
                likelihood_new = (-ndw / 2) * np.log(inner_y_y - (inner_y_gnew ** 2 / inner_gnew_gnew))

                parameter_vector = model_reduced.parameters_to_parameter_vector(**params_all_tform)[idx_roi, :]
                prior = np.log(scipy.stats.multivariate_normal.pdf(parameter_vector, mu, sigma, allow_singular=1))
                parameter_vector = model_reduced.parameters_to_parameter_vector(**params_all_new)[idx_roi, :]  # [i, :]
                prior_new = prior_new + np.log(scipy.stats.multivariate_normal.pdf(parameter_vector, mu, sigma, allow_singular=1))

                # TODO: investigate big discrepancy between r and alpha
                alpha = [np.min([0, (likelihood_new[i] + prior_new[i]) - (likelihood[i] + prior[i])]) for i in
                         range(nvox_roi)]
                r = np.log(np.random.uniform(0, 1, nvox_roi))

                # accept new parameter value if criteria met (col 1 -> roi voxel indices, col 2 -> fov voxel indices)
                to_accept = np.array([[i, idx_roi[i]] for i in range(nvox_roi) if r[i] < alpha[i]])
                to_reject = np.array([[i, idx_roi[i]] for i in range(nvox_roi) if r[i] > alpha[i]])
                if model.parameter_cardinality[p] > 1:
                    for card in range(model.parameter_cardinality[p]):
                        if to_accept.shape != (0,):  # account for error thrown by no accepted moves
                            accepted[p][to_accept[:, 1], card] += 1
                            accepted_per_100[p][to_accept[:, 1], card] += 1
                            params_all_tform[p][to_accept[:, 1], card] = copy(params_all_new[p][to_accept[:, 1], card])
                            likelihood_stored[p][to_accept[:, 1], card, j] = likelihood_new[to_accept[:, 0]] + prior_new[to_accept[:, 0]]
                        likelihood_stored[p][to_reject[:, 1], card, j] = likelihood[to_reject[:, 0]] + prior[to_reject[:, 0]]
                elif model.parameter_cardinality[p] == 1:
                    if to_accept.shape != (0,):  # account for error thrown by no accepted moves
                        accepted[p][to_accept[:, 1]] += 1
                        accepted_per_100[p][to_accept[:, 1]] += 1
                        params_all_tform[p][to_accept[:, 1]] = copy(params_all_new[p][to_accept[:, 1]])
                        likelihood_stored[p][to_accept[:, 1], j] = likelihood_new[to_accept[:, 0]] + prior_new[to_accept[:, 0]]
                    likelihood_stored[p][to_reject[:, 1], j] = likelihood[to_reject[:, 0]] + prior[to_reject[:, 0]]
                if to_accept.shape != (0,):  # account for error thrown by no accepted moves
                    E_fit[to_accept[:, 1], :] = g_new[to_accept[:, 0], :]

                # FIXME: add this back in during testing
                # else:
                #     if Accepted / (it * nvox_roi) < 0.23:
                #         continue
                #         print("Stopping criterion met {}".format(Accepted/(it*nvox_roi)))
                #         return Acceptance_rate

                # for posterior distribution and plotting
                if model.parameter_cardinality[p] > 1:
                    for card in range(model.parameter_cardinality[p]):
                        param_conv[p][idx_roi, card, j] = tform_params(params_all_tform, model.parameter_names, model, 'r')[p][idx_roi, card]
                        acceptance_rate[p][idx_roi, card, j] = accepted[p][idx_roi, card] / (j + 1)  # * nvox_roi)
                elif model.parameter_cardinality[p] == 1:
                    param_conv[p][idx_roi, j] = np.array(tform_params(params_all_tform, model.parameter_names, model, 'r')[p])[idx_roi]
                    acceptance_rate[p][idx_roi, j] = accepted[p][idx_roi] / (j + 1)  # * nvox_roi)

            if np.remainder(j, 100) == 0 and 0 < j <= burn_in / 2:
                for param in parameters_to_fit:
                    if model.parameter_cardinality[param] > 1:
                        for card in range(model.parameter_cardinality[p]):
                            # w[param][idx_roi, card] = w[param][idx_roi, card] * (101 / (2 * (101 - (accepted_per_100[param][idx_roi, card] / nvox_roi))))
                            # w[param][idx_roi, card] = w[param][idx_roi, card] * (101 / (2 * (101 - (accepted_per_100[param][idx_roi, card]))))
                            w[param][idx_roi, card] = w[param][idx_roi, card] * (101 / (2 * (101 - (accepted_per_100[param][idx_roi, card])/model.parameter_cardinality[param])))
                            w_stored[param][idx_roi, card, np.int((j+1)/100)] = w[param][idx_roi, card]
                            accepted_per_100[param][idx_roi, card] = np.zeros(nvox_roi)
                    elif model.parameter_cardinality[param] == 1:
                        # w[param][idx_roi] = w[param][idx_roi] * 101 / (2 * (101 - (accepted_per_100[param][idx_roi] / nvox_roi)))
                        w[param][idx_roi] = w[param][idx_roi] * 101 / (2 * (101 - (accepted_per_100[param][idx_roi])))
                        w_stored[param][idx_roi, np.int((j+1)/100)] = w[param][idx_roi]
                        accepted_per_100[param] = np.zeros(nvox)
                    # plot weights
                    # if (param == 'partial_volume_0') & (roi == 0):
                    if roi == 0:
                        if model.parameter_cardinality[param] > 1:
                            col = ['tab:purple', 'tab:pink']
                            mkr = ['v', '^']
                            for card in range(model.parameter_cardinality[p]):
                                lb = (model.parameter_ranges[param][card][0] - 1e-5) * model.parameter_scales[param][card]  # lower bound
                                ub = (model.parameter_ranges[param][card][1] + 1e-5) * model.parameter_scales[param][card]  # upper bound
                                # tmp = (lb + ub * np.exp(w[param][card][vox])) / (1 + np.exp(w[param][card][vox]))
                                tmp = (lb + ub * np.exp(w[param][vox, card])) / (1 + np.exp(w[param][vox, card]))
                                ax.plot(j, tmp, color=col[card], marker=mkr[card], markersize=15)
                        else:
                            lb = (model.parameter_ranges[param][0] - 1e-5) * model.parameter_scales[param]  # lower bound
                            ub = (model.parameter_ranges[param][1] + 1e-5) * model.parameter_scales[param]  # upper bound
                            tmp = (lb + ub * np.exp(w[param][vox])) / (1 + np.exp(w[param][vox]))
                            if param == 'C1Stick_1_lambda_par':
                                ax.plot(j, tmp*1e9, color='tab:red', marker='v', markersize=15)
                            elif param == 'G1Ball_1_lambda_iso':
                                ax.plot(j, tmp*1e9, color='tab:green', marker='^', markersize=15)
                            elif param == 'partial_volume_0':
                                ax.plot(j, tmp, color='tab:blue', marker='o', markersize=15)

    # params_all = tform_params(params_all_new, model.parameter_names, model, 'r')
    params_all = tform_params(params_all_new, model.parameter_names, model, 'r')
    for param in parameters_to_fit:
        params_all[param] = np.mean(param_conv[param][:, burn_in:-1], axis=1)

    return acceptance_rate, param_conv, params_all, params_all_orig, likelihood_stored, w_stored
