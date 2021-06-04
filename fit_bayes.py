import numpy as np
from copy import copy, deepcopy
import scipy
#from dmipy.core import modeling_framework
#from dmipy.core import fitted_modeling_framework


# FIXME: volume fraction transform for all parameters
def tform_params(param_dict, parameter_names, direction):
    param_dict = deepcopy(param_dict)  # because dicts are mutable, and don't want to alter inside function
    if direction == 'f':
        for param in parameter_names:
            if 'partial_volume_' in param:
                param_dict[param] = np.log(param_dict[param]) - np.log(1 - param_dict[param])
            elif '_mu' in param:
                # do nothing
                param_dict[param] = param_dict[param]
            else:
                param_dict[param] = np.log(param_dict[param])

    elif direction == 'r':
        for param in parameter_names:
            if 'partial_volume_' in param:
                param_dict[param] = np.exp(param_dict[param]) / (1 + np.exp(param_dict[param]))
                param_dict[param] = [1 if np.isnan(x) else x for x in param_dict[param]]
            elif '_mu' in param:
                # do nothing
                param_dict[param] = param_dict[param]
            else:
                param_dict[param] = np.exp(param_dict[param])

    else:
        print('Incorrect input! Nothing is happening...')

    return param_dict


# FIXME: create as class like modelling_framework.py?
def fit_bayes(model, acq_scheme, data, mask=None, nsteps=1000, burn_in=500):
    # FIXME: data checks?

    # set mask default
    if mask is None:
        mask = (data[:, 0] > 0).astype('uint8')
    # convert to int if input as bool
    elif mask.any():
        mask = mask.astype('uint8')

    # extract ROIs if present in mask
    roi_vals = np.unique(mask)  # list of unique integers that identify each ROI
    nroi = roi_vals.shape[0]  # no. ROIs

    # extract other useful values
    nvox = np.sum(mask > 0)  # number of voxels in mask
    nparams = np.sum(np.array(list(model.parameter_cardinality.values())))
    ndw = len(acq_scheme.bvalues)
    ncomp = model.partial_volume_names.__len__()


    # do initial LQS fit for parameter initialisation
    lsq_fit = model.fit(acq_scheme, data, mask=mask>0)

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
        init_param[param] = lsq_fit.fitted_parameters[param][mask>0]

    # perturb params for testing
    init_param[model.parameter_names[1]][0] = 10e-9  # Dpar
    init_param[model.parameter_names[2]][0] = 10e-9  # Diso
    #init_param[model.parameter_names[3]][0] = .5     # fpar

    # create dict of LSQ fit values, original values
    params_all_orig = copy(init_param)
    # create dict of LSQ fit values, log transform variables (non-orientation only) (original -> log)
    params_all_tform = tform_params(init_param, model.parameter_names, 'f')

    # initialise sigma
    sigma = np.cov(np.transpose(model_reduced.parameters_to_parameter_vector(**params_all_tform)))

    # TODO: play with weights created from ranges
    # initial weights for Metropolis-Hastings parameter sampling
    w = dict.fromkeys(parameters_to_fit)
    for param in parameters_to_fit:                                     # get scaled parameter ranges
        w[param] = np.array(model.parameter_ranges[param]) * model.parameter_scales[param]
    w = tform_params(w, parameters_to_fit, 'f')                         # transform parameter ranges
    for param in parameters_to_fit:                                     # set weight as x * range
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                w[param][card] = 0.1 * np.abs(np.subtract(w[param][card, 1], w[param][card, 0]))
            w[param] = w[param][range(model.parameter_cardinality[param]), 0]
            w[param] = np.tile(w[param], (nvox, 1))  # tile to create weight for each voxel
        elif model.parameter_cardinality[param] == 1:
            w[param] = 0.1 * np.abs(np.subtract(w[param][1], w[param][0]))
            w[param] = np.tile(w[param], nvox)                         # tile to create weight for each voxel

    # values for f, D, D* from Orton, orientations guessed
#    w['C1Stick_1_mu'] = np.array((.1, .1))
#    w['C1Stick_1_lambda_par'] = .5
#    w['G1Ball_1_lambda_iso'] = .5
#    w['partial_volume_0'] = .5

    # measured signals
    E_dat = data[mask>0]

    # initial LSQ-estimated signals
    E_fit = np.array([model.simulate_signal(acq_scheme, model.parameters_to_parameter_vector(**params_all_orig)[i, :])
                      for i in range(nvox)])

    # initialise variables to track state of optimisation at each step
    accepted = dict.fromkeys(parameters_to_fit)             # track total accepted moves for each param
    accepted_per_100 = dict.fromkeys(parameters_to_fit)     # track accepted moves for each param per 100 steps (to update weights)
    acceptance_rate = dict.fromkeys(parameters_to_fit)      # track accepted moves for each param at each step
    param_conv = dict.fromkeys(parameters_to_fit)           # track parameter convergence
    gibbs_mu = np.zeros((nroi, nparams-1, nsteps))                # track mu at each step
    gibbs_sigma = np.zeros((nroi, nparams-1, nparams-1, nsteps))  # track sigma at each step

    # initialise dictionaries (param_conv, accepted, accepted_per_100, acceptance_rate)
    for param in parameters_to_fit:
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                accepted[param] = np.zeros((nvox, model.parameter_cardinality[param]))
                accepted_per_100[param] = np.zeros((nvox, model.parameter_cardinality[param]))
                acceptance_rate[param] = np.zeros((nvox, model.parameter_cardinality[param], nsteps))
                param_conv[param] = np.zeros((nvox, model.parameter_cardinality[param], nsteps))
        elif model.parameter_cardinality[param] == 1:
            accepted[param] = np.zeros(nvox)
            accepted_per_100[param] = np.zeros(nvox)
            acceptance_rate[param] = np.zeros((nvox, nsteps))
            param_conv[param] = np.zeros((nvox, nsteps))

    # loop over ROIs
    for roi in range(nroi):
        idx_roi = [xx for xx, x in enumerate(mask == roi_vals[roi]) if x]  # indices into mask of voxels in ROI
        nvox_roi = idx_roi.__len__()  # no. voxels in ROI

        # NB i (voxel loop) and j (MC loop) in keeping with Orton paper
        for j in range(0, nsteps):
            print(j)
            # Gibbs moves to update priors.
            # Gibbs 1. sample mu from multivariate normal dist defined by current param estimates.
            m = np.mean(model_reduced.parameters_to_parameter_vector(**params_all_tform), axis=0)
            V = sigma / nvox_roi
            mu = np.random.multivariate_normal(m, V)
            # Gibbs 2. sample sigma from inverse Wishart distribution (using newly updated mu)
            phi = np.sum([np.outer(model_reduced.parameters_to_parameter_vector(**params_all_tform)[i, :] - mu,
                                   model_reduced.parameters_to_parameter_vector(**params_all_tform)[i, :] - mu)
                          for i in range(0, nvox_roi)], axis=0)
            sigma = scipy.stats.invwishart(scale=phi, df=nvox_roi - nparams-1).rvs()

            # save Gibbs parameters for this step (careful of parameter ordering)
            gibbs_mu[roi, :, j] = copy(mu)
            gibbs_sigma[roi, :, :, j] = copy(sigma)

            # Metropolis-Hastings parameter updates
            params_all_new = deepcopy(params_all_tform)
            # for i in idx_roi:  # loop over voxels in ROI
            # TODO: investigate order of parameter loop and convergence
            # FIXME: loop over directions in separate loops
            for p in parameters_to_fit:
                # sample parameter
                if model.parameter_cardinality[p] > 1:
                    for card in range(model.parameter_cardinality[p]):
#                            params_all_new[p][i][card] = np.random.normal(params_all_tform[p][i][card], w[p][card])
                        params_all_new[p][idx_roi, card] = np.random.normal(params_all_tform[p][idx_roi, card], w[p][idx_roi, card])
                elif model.parameter_cardinality[p] == 1:
#                       params_all_new[p][i] = np.random.normal(params_all_tform[p][i], w[p])
                    params_all_new[p][idx_roi] = np.random.normal(params_all_tform[p][idx_roi], np.matrix.flatten(w[p][idx_roi]))

                # if a volume fraction was sampled, re-compute dependent fraction
                if 'partial_volume_' in p:
                    f_indep = [params_all_new[name][idx_roi] for name in model.partial_volume_names if name != dependent_fraction]
                    f_indep = np.exp(f_indep) / (1 + np.exp(f_indep))       # tform indept fractions (log -> orig)
                    for c in range(ncomp-1):                                # check transform
                        f_indep[c, :] = [1 if np.isnan(x) else x for x in f_indep[c, :]]
                    f_dept = 1 - np.sum(f_indep, axis=0)                    # compute dept fraction
                    f_dept = np.log(f_dept) - np.log(1 - f_dept)            # tform dept fraction (orig -> log)
                    params_all_new[dependent_fraction][idx_roi] = f_dept
                    # FIXME: check that dependent vol fraction is not < 1 (could happen for n_compartments > 2)

                # compute acceptance
#                    y_i = copy(E_dat[i, ])  # actual measured signal
#                    g_i = copy(E_fit[i, ])  # model-predicted signal (old params)
                y = copy(E_dat[idx_roi, :])  # actual measured signal
                g = copy(E_fit[idx_roi, :])  # model-predicted signal (old params)

                parameter_vector = tform_params(params_all_new, model.parameter_names, 'r')
                parameter_vector = model.parameters_to_parameter_vector(**parameter_vector) #[i, :]
                g_new = model.simulate_signal(acq_scheme, parameter_vector[idx_roi, :])  # model-predicted signal (new params)
#                    g_i_new = model.simulate_signal(acq_scheme, parameter_vector[i, :])  # model-predicted signal (new params)

                # calculate posteriors and PDFs (log scale)
                inner_y_y = np.sum(np.multiply(np.squeeze(y), np.squeeze(y)), 1)
                inner_y_g = np.sum(np.multiply(np.squeeze(y), np.squeeze(g)), 1)
                inner_g_g = np.sum(np.multiply(np.squeeze(g), np.squeeze(g)), 1)
                inner_y_gnew = np.sum(np.multiply(np.squeeze(y), np.squeeze(g_new)), 1)
                inner_gnew_gnew = np.sum(np.multiply(np.squeeze(g), np.squeeze(g_new)), 1)
                likelihood = (-ndw / 2) * np.log(inner_y_y - (inner_y_g ** 2 / inner_g_g))
                likelihood_new = (-ndw / 2) * np.log(inner_y_y - (inner_y_gnew ** 2 / inner_gnew_gnew))

#               likelihood = (-ndw / 2) * np.log(np.inner(y_i, y_i) - ((np.inner(y_i, g_i)) ** 2 / np.inner(g_i, g_i)))
#               likelihood_new = (-ndw / 2) * np.log(np.inner(y_i, y_i) - ((np.inner(y_i, g_i_new)) ** 2 / np.inner(g_i_new, g_i_new)))

                parameter_vector = model_reduced.parameters_to_parameter_vector(**params_all_tform)[idx_roi, :] # [i, :]
                prior = np.log(scipy.stats.multivariate_normal.pdf(parameter_vector, mu, sigma, allow_singular=1))
                parameter_vector = model_reduced.parameters_to_parameter_vector(**params_all_new)[idx_roi, :]  # [i, :]
                prior_new = np.log(scipy.stats.multivariate_normal.pdf(parameter_vector, mu, sigma, allow_singular=1))

                # FIXME: investigate big discrepancy between r and alpha
#                    alpha = np.min([0, (likelihood_new + prior_new) - (likelihood + prior)])
                alpha = [np.min([0, (likelihood_new[i] + prior_new[i]) - (likelihood[i] + prior[i])]) for i in range(nvox_roi)]
                r = np.log(np.random.uniform(0, 1, nvox_roi))

                # reject new parameter value if criteria not met
#                    if r < alpha:
#                        accepted[p] += 1
#                        accepted_per_100[p] += 1
#                        params_all_tform[p][i] = copy(params_all_new[p][i])
#                        E_fit[i, ] = copy(g_i_new)

                to_accept = [idx_roi[i] for i in range(nvox_roi) if r[i] < alpha[i]]
                if model.parameter_cardinality[p] > 1:
                    for card in range(model.parameter_cardinality[p]):
                        accepted[p][to_accept, card] += 1
                        accepted_per_100[p][to_accept, card] += 1
                        params_all_tform[p][to_accept, card] = copy(params_all_new[p][to_accept, card])
                elif model.parameter_cardinality[p] == 1:
                    accepted[p][to_accept] += 1
                    accepted_per_100[p][to_accept] += 1
                    params_all_tform[p][to_accept] = copy(params_all_new[p][to_accept])
                E_fit[to_accept, :] = g_new[r<alpha, :]

                # FIXME: add this back in during testing
                # else:
                #     if Accepted / (it * nvox_roi) < 0.23:
                #         continue
                #         print("Stopping criterion met {}".format(Accepted/(it*nvox_roi)))
                #         return Acceptance_rate

                # for posterior distribution and plotting
                if model.parameter_cardinality[p] > 1:
                    for card in range(model.parameter_cardinality[p]):
#                       param_conv[p][card][i, j] = tform_params(params_all_tform, model.parameter_names, 'r')[p][i][card]
                        param_conv[p][idx_roi, card, j] = [tform_params(params_all_tform, model.parameter_names, 'r')[p][i, card] for i in idx_roi]
                        acceptance_rate[p][idx_roi, card, j] = accepted[p][idx_roi, card] / ((j+1) * nvox_roi)
                elif model.parameter_cardinality[p] == 1:
#                   param_conv[p][i, j] = tform_params(params_all_tform, model.parameter_names, 'r')[p][i]
                    param_conv[p][idx_roi, j] = [tform_params(params_all_tform, model.parameter_names, 'r')[p][i] for i in idx_roi]
                    acceptance_rate[p][idx_roi, j] = accepted[p][idx_roi] / ((j+1) * nvox_roi)

            if np.remainder(j, 100) == 0 and 0 < j <= burn_in / 2:
                for param in parameters_to_fit:
#                    w[param] = w[param] * (101 / (2 * (101 - (accepted_per_100[param] / nvox_roi))))
#                    accepted_per_100[param] = 0
                    if model.parameter_cardinality[param] > 1:
                        for card in range(model.parameter_cardinality[p]):
                            w[param][idx_roi, card] = w[param][idx_roi, card] * (101 / (2 * (101 - (accepted_per_100[param][idx_roi, card] / nvox_roi))))
                            accepted_per_100[param][idx_roi, card] = np.zeros(nvox_roi)
                    elif model.parameter_cardinality[param] == 1:
                        w[param][idx_roi] = w[param][idx_roi] * 101 / (2 * (101 - (accepted_per_100[param][idx_roi] / nvox_roi)))
                        accepted_per_100[param] = np.zeros(nvox)

                # print(w)

    params_all = tform_params(params_all_new, model.parameter_names, 'r')

    return acceptance_rate, param_conv, params_all, params_all_orig
