import numpy as np
from copy import copy, deepcopy
import scipy
from dmipy.core import modeling_framework
from dmipy.core import fitted_modeling_framework


# FIXME: sigmoid transform for all parameters?
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


# FIXEDME: existing dmipy fnc that does this? (model.parameters_to_parameter_vector)
'''
def dict_to_array(model, param_dict, param_names):
    # get array of parameter values
    param_array = model.parameters_to_parameter_vector(**param_dict)

    # remove
    for param in param_dict.keys():
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                param_array.append(param_dict[param][:, card])
        elif model.parameter_cardinality[param] == 1:
            param_array.append(param_dict[param])

    return np.asarray(param_array)
'''
# FIXEDME: existing dmipy fnc that does this? (model.parameter_vector_to_parameters)
'''
def array_to_dict(model, param_array):
    l = 0
    param_dict = dict.fromkeys(model.parameter_names)
    # FIXME: remove this harcoding once no. compartments fix sorted (line ~223)
    for param in model.parameter_names[0:param_array.shape[0]-1]:
        if '_mu' in param:  # if an orientation parameter
            # two maps in the final dimension
            this_param_map = np.zeros((np.shape(param_array)[1], 2))
            this_param_map[:, 0] = param_array[l, :]
            this_param_map[:, 1] = param_array[l + 1, :]
            param_dict[param] = this_param_map
            l = l + 2
        else:
            this_param_map = param_array[l, :]
            param_dict[param] = this_param_map
            l = l + 1

    return param_dict
'''


# FIXME: set defaults for nsteps, burn_in and mask?
# FIXEDME: normalisation of data c.f. line 1110 model_framework.fit - don't need to bother, S0 is normalised away in Bayes
def fit_bayes(model, acq_scheme, data, mask, nsteps=1000, burn_in=500):
    '''
    self._check_tissue_model_acquisition_scheme(acq_scheme)
    self._check_model_params_with_acquisition_params(acq_scheme)
    self._check_acquisition_scheme_has_b0s(acq_scheme)
    self._check_if_volume_fractions_are_fixed()
    self._check_if_sh_coeff_fixed_if_present()
    '''
    # extract some useful values
    nvox = np.sum(mask)  # number of voxels in mask
    all_model_param_names = '\t'.join(model.parameter_names)
    n_orient_params = all_model_param_names.count('_mu')
    nparams = len(model.parameter_names) + n_orient_params  # no. parameters, accounting for both orientations
    ndw = len(acq_scheme.bvalues)

    # do initial LQS fit for parameter initialisation
    lsq_fit = model.fit(acq_scheme, data, mask=mask)

    # get number of compartments; only fit no. compartments - 1
    n_compartments = model.partial_volume_names.__len__()
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
    # FIXEDME: only need to fit no. compartments - 1 'partial_volume_' parameters
    init_param = dict.fromkeys(model.parameter_names)
    for param in model.parameter_names:
        init_param[param] = lsq_fit.fitted_parameters[param][mask]

    # perturb params for testing
    init_param[model.parameter_names[1]][0] = 10e-9  # Dpar
    init_param[model.parameter_names[2]][0] = 10e-9  # Diso
    #init_param[model.parameter_names[3]][0] = .5     # fpar

    # FIXEDME: prob want to keep as dictionary - would help with identifying orientation params in MCMC loop
    # create dict of LSQ fit values, original values
    # params_all_orig = dict_to_array(model, init_param)
    params_all_orig = copy(init_param)
    # create dict of LSQ fit values, log transform variables (non-orientation only) (original -> log)
    # params_all_tform = dict_to_array(model, tform_params(init_param, model, 'f'))
    params_all_tform = tform_params(init_param, model.parameter_names, 'f')

    # initialise sigma
    # FIXEDME: remove hardcoding of [0:nparams-1, :] once no. compartments fix sorted (line ~223)
    # sigma = np.cov(params_all_tform[0:nparams-1, :])
    sigma = np.cov(np.transpose(model_reduced.parameters_to_parameter_vector(**params_all_tform)))

    # initial weights for Metropolis-Hastings parameter sampling (f, D, D* from Orton, orientations guessed)
    # FIXEDME: calculate using parameter ranges
    # w = [.5, .5, .5, .1, .1]
    # w = [.1, .1, .5, .5, .5]  # ori, ori, Dpar, Diso, fpar
    w = dict.fromkeys(parameters_to_fit)
    for param in parameters_to_fit:                                     # get scaled parameter ranges
        w[param] = np.array(model.parameter_ranges[param]) * model.parameter_scales[param]
    w = tform_params(w, parameters_to_fit, 'f')                         # transform parameter ranges
    for param in parameters_to_fit:                                     # set weight
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                w[param][card] = 0.1 * np.abs(np.subtract(w[param][card][1], w[param][card][0]))
            w[param] = w[param][range(model.parameter_cardinality[param]), 0]
        elif model.parameter_cardinality[param] == 1:
            w[param] = 0.1 * np.abs(np.subtract(w[param][1], w[param][0]))

    w['C1Stick_1_mu'] = np.array((.1, .1))
    w['C1Stick_1_lambda_par'] = .5
    w['G1Ball_1_lambda_iso'] = .5
    w['partial_volume_0'] = .5

    # measured signals
    E_dat = data[mask]

    # initial LSQ-estimated signals
    # FIXEDME: need to mask E_fit - nvox is the number of voxels in the mask, so automatically masked
    #E_fit = np.zeros((nvox, ndw))
    #for i in range(nvox):
    #    parameter_vector = params_all_orig[:, i]
    #    E_fit[i, :] = model.simulate_signal(acq_scheme, parameter_vector)
    E_fit = np.array([model.simulate_signal(acq_scheme, model.parameters_to_parameter_vector(**params_all_orig)[i, :])
                      for i in range(nvox)])

    # FIXEDME: maybe need to deal with nparams-1 once generalised for >2 compartments - this is fine
    # accepted = np.zeros(nparams-1)          # count total accepted moves for each param
    # accepted_per_100 = np.zeros(nparams-1)  # track accepted moves for each param per 100 steps (to update weights)
    # acceptance_rate = np.zeros([nparams-1, nsteps])   # track accepted moves for each param at each step
    accepted = dict.fromkeys(parameters_to_fit)          # count total accepted moves for each param
    accepted_per_100 = dict.fromkeys(parameters_to_fit)  # track accepted moves for each param per 100 steps (to update weights)
    acceptance_rate = dict.fromkeys(parameters_to_fit)   # track accepted moves for each param at each step

    # FIXEDME: need to track all parameter values at each step for a posterior distribution (currently tracking 1 voxel)
    # FIXME: also need to store final mu and sigma (?)
    param_conv = dict.fromkeys(parameters_to_fit)       # track parameter convergence
    gibbs_mu = np.zeros((nparams-1, nvox))              # save final mu
    # gibbs_sigma = np.zeros((nparams-1, nvox))         # save final sigma

    for param in parameters_to_fit:
        accepted[param] = 0
        accepted_per_100[param] = 0
        acceptance_rate[param] = np.zeros(nsteps)
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                param_conv[param] = np.zeros((model.parameter_cardinality[param], nvox, nsteps))
        elif model.parameter_cardinality[param] == 1:
            param_conv[param] = np.zeros((nvox, nsteps))

    # FIXME: add in loop for different ROIs (version in bayesian-fitting-toy-example.py)

    # NB i (voxel loop) and j (MC loop) in keeping with Orton paper
    for j in range(0, nsteps):
        print(j)
        # it = j + 1
        # Gibbs moves to update priors
        # sample mu from multivariate normal distribution defined by current parameter estimates
        # FIXEDME: remove hardcoding of [0:nparams-1, :] once generalised for >2 compartments
        #m = np.mean(params_all_tform[0:nparams-1, :], axis=1)
        m = np.mean(model_reduced.parameters_to_parameter_vector(**params_all_tform), axis=0)
        V = sigma / nvox
        mu = np.random.multivariate_normal(m, V)

        # sample sigma from inverse Wishart distribution (using newly updated mu)
        # NB scaled parameters used in calculation of priors in Metropolis-Hastings updates
        # FIXEDME: remove hardcoding of [0:nparams-1, :] once generalised for >2 compartments
        #phi = np.sum([np.outer(params_all_tform[0:nparams-1, i] - mu,
        #                       params_all_tform[0:nparams-1, i] - mu)
        #              for i in range(0, nvox)], axis=0)
        phi = np.sum([np.outer(model_reduced.parameters_to_parameter_vector(**params_all_tform)[i, :] - mu,
                               model_reduced.parameters_to_parameter_vector(**params_all_tform)[i, :] - mu)
                      for i in range(0, nvox)], axis=0)
        sigma = scipy.stats.invwishart(scale=phi, df=nvox - nparams).rvs()

        # Metropolis-Hastings parameter updates
        params_all_new = deepcopy(params_all_tform)
        for i in range(0, nvox):
            # FIXEDME: use parameter names
            # FIXME: order of parameter loop affects convergence (?!)
            for p in ['C1Stick_1_lambda_par', 'G1Ball_1_lambda_iso', 'partial_volume_0', 'C1Stick_1_mu']:  # parameters_to_fit:  # [2, 3, 4, 0, 1]:  # range(nparams-1):
                # sample parameter
                '''
                if p == 0:  # stick orientation (theta)
                    u = np.random.uniform(0, 1, 1)
                    params_all_new[p, i] = np.arccos(1 - 2*u)
                elif p == 1:  # stick orientation (phi)
                    u = np.random.uniform(0, 1, 1)
                    params_all_new[p, i] = 2 * np.pi * u
                elif p == 2 or p == 3:    # Dpar, Diso
                    params_all_new[p, i] = np.random.normal(params_all_tform[p, i], w[p])
                elif p == 4:
                    # FIXME: generalise for >2 compartments
                    params_all_new[p, i] = np.random.normal(params_all_tform[p, i], w[p])
                    params_all_new[p+1, i] = 1 - params_all_new[p, i]
                '''
                if model.parameter_cardinality[p] > 1:
                    for card in range(model.parameter_cardinality[p]):
                        params_all_new[p][i][card] = np.random.normal(params_all_tform[p][i][card], w[p][card])
                elif model.parameter_cardinality[p] == 1:
                    params_all_new[p][i] = np.random.normal(params_all_tform[p][i], w[p])

                if 'partial_volume_' in p:
                    f = [params_all_new[name][i] for name in model.partial_volume_names if name != dependent_fraction]
                    f = np.exp(f) / (1 + np.exp(f))             # transform log -> original
                    f = [1 if np.isnan(x) else x for x in f]    # check
                    f = 1 - np.sum(f)                           # compute dependent fraction
                    f = np.log(f) - np.log(1 - f)               # transform original -> log
                    params_all_new[dependent_fraction][i] = f
                    # FIXME: check that dependent vol fraction is not < 1 (could happen for n_compartments > 2)

                # compute acceptance
                y_i = copy(E_dat[i, ])  # actual measured signal
                g_i = copy(E_fit[i, ])  # model-predicted signal (old params)

                # params_all_new = dict_to_array(model, tform_params(array_to_dict(model, params_all_new), model, 'r'))  # revert transform for signal calc (log -> original)
                # parameter_vector = params_all_new[:, i]
                # g_i_new = model.simulate_signal(acq_scheme, parameter_vector)  # model-predicted signal (new params)
                # params_all_new = dict_to_array(model, tform_params(array_to_dict(model, params_all_new), model, 'f'))  # redo transform (original -> log)
                parameter_vector = tform_params(params_all_new, model.parameter_names, 'r')
                np.add(parameter_vector['partial_volume_0'], parameter_vector['partial_volume_1'])
                parameter_vector = model.parameters_to_parameter_vector(**parameter_vector)[i, :]
                g_i_new = model.simulate_signal(acq_scheme, parameter_vector)  # model-predicted signal (new params)

                # calculate posteriors and PDFs (log scale)
                likelihood = (-ndw / 2) * np.log(np.inner(y_i, y_i) -
                                                 ((np.inner(y_i, g_i)) ** 2 / np.inner(g_i, g_i)))
                likelihood_new = (-ndw / 2) * np.log(np.inner(y_i, y_i) -
                                                     ((np.inner(y_i, g_i_new)) ** 2 / np.inner(g_i_new, g_i_new)))

                # FIXEDME: remove hardcoding of [0:nparams-1, :] once no. compartments fix sorted (line ~223)
                # prior = np.log(scipy.stats.multivariate_normal.pdf(params_all_tform[0:nparams-1, i], mu, sigma, allow_singular=1))
                # prior_new = np.log(scipy.stats.multivariate_normal.pdf(params_all_new[0:nparams-1, i], mu, sigma, allow_singular=1))
                parameter_vector = model_reduced.parameters_to_parameter_vector(**params_all_tform)[i, :]
                prior = np.log(scipy.stats.multivariate_normal.pdf(parameter_vector, mu, sigma, allow_singular=1))
                parameter_vector = model_reduced.parameters_to_parameter_vector(**params_all_new)[i, :]
                prior_new = np.log(scipy.stats.multivariate_normal.pdf(parameter_vector, mu, sigma, allow_singular=1))

                alpha = np.min([0, (likelihood_new + prior_new) - (likelihood + prior)])
                r = np.log(np.random.uniform(0, 1))

                # reject new parameter value if criteria not met
                if r < alpha:
                    accepted[p] += 1
                    accepted_per_100[p] += 1
                    params_all_tform[p][i] = copy(params_all_new[p][i])
                    E_fit[i, ] = copy(g_i_new)
                    # FIXME: add this back in
                #            else:
                #                if Accepted / (it * nvox) < 0.23:
                #                    continue
                #                    # print("Stopping criterion met {}".format(Accepted/(it*nvox)))
                #                    # return Acceptance_rate

                # for posterior distribution and plotting
                # param_conv[p][i, j] = copy(params_all_tform[p][i])
                if model.parameter_cardinality[p] > 1:
                    for card in range(model.parameter_cardinality[p]):
                        param_conv[p][card][i, j] = tform_params(params_all_tform, model.parameter_names, 'r')[p][i][card]
                elif model.parameter_cardinality[p] == 1:
                    param_conv[p][i, j] = tform_params(params_all_tform, model.parameter_names, 'r')[p][i]
                gibbs_mu[:, i] = copy(mu)
                # gibbs_sigma[:, i] = copy(sigma)

                # acceptance_rate.append(accepted / (it * nvox))
                acceptance_rate[p][j] = accepted[p] / ((j+1) * nvox)

        if np.remainder(j, 100) == 0 and 0 < j <= burn_in / 2:
            # w = w * (101 / (2 * (101-(accepted_per_100/nvox))))
            # accepted_per_100 = np.zeros((w.__len__()))
            for param in parameters_to_fit:
                w[param] = w[param] * (101 / (2 * (101 - (accepted_per_100[param] / nvox))))
                accepted_per_100[param] = 0
            print(w)

    # params_all = dict_to_array(model, tform_params(array_to_dict(model, params_all_new), model, 'r'))
    params_all = tform_params(params_all_new, model.parameter_names, 'r')
    # FIXEDME: don't hardcode calculation of last compartment fraction
    # param_conv = dict_to_array(model, tform_params(array_to_dict(model, param_conv), model, 'r'))

    return acceptance_rate, param_conv, params_all, params_all_orig
