import numpy as np
import copy
import scipy
from dmipy.core import modeling_framework


# FIXME: sigmoid transform for all parameters?
def tform_params(param_dict, model, direction):
    if direction == 'f':
        for param in model.parameter_names:
            if 'partial_volume_' in param:
                param_dict[param] = np.log(param_dict[param]) - np.log(1 - param_dict[param])
            elif '_mu' in param:
                # do nothing
                param_dict[param] = param_dict[param]
            else:
                param_dict[param] = np.log(param_dict[param])

    elif direction == 'r':
        for param in model.parameter_names:
            if 'partial_volume_' in param:
                param_dict[param] = np.exp(param_dict[param]) / (1 + np.exp(param_dict[param]))
                param_dict[param] = [1 if np.isnan(x) else x for x in param_dict[param]]
            elif '_mu' in param:
                # do nothing
                param_dict[param] = param_dict[param]
            else:
                param_dict[param] = np.exp(param_dict[param])

    return param_dict


# FIXME: existing dmipy fnc that does this? (model.parameters_to_parameter_vector)
def dict_to_array(model, param_dict):
    # create array of LSQ fit values
    all_model_param_names = '\t'.join(model.parameter_names)
    n_orient_params = all_model_param_names.count('_mu')
    nparams = len(model.parameter_names) + n_orient_params  # no. parameters, accounting for both orientations
    nvox = param_dict[model.parameter_names[0]].shape[0]

    param_array = np.empty([nparams, nvox])
    partial_volume_index = []  # find the positions that are volume fractions
    l = 0
    for param in model.parameter_names:
        if '_mu' in param:  # if an orientation parameter
            param_array[l:l + 2, :] = np.transpose(param_dict[param])
            l = l + 2
        elif 'partial_volume_' in param:
            partial_volume_index.append(l)  # need to know the index of volume fractions for the MH proposal step
            param_array[l, :] = np.transpose(param_dict[param])
            l = l + 1
        else:
            param_array[l, :] = np.transpose(param_dict[param])
            l = l + 1

    return param_array


# FIXME: existing dmipy fnc that does this? (model.parameter_vector_to_parameters)
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


# FIXME: set defaults for nsteps, burn_in and mask?
def fit_bayes(model, acq_scheme, data, nsteps, burn_in, mask):
    # extract some useful values
    nvox = np.sum(mask)  # number of voxels in mask
    all_model_param_names = '\t'.join(model.parameter_names)
    n_orient_params = all_model_param_names.count('_mu')
    nparams = len(model.parameter_names) + n_orient_params  # no. parameters, accounting for both orientations
    ndw = len(acq_scheme.bvalues)

    # do initial LQS fit for parameter initialisation
    lsq_fit = model.fit(acq_scheme, data, mask=mask)

    # create dictionary of model parameter names and LSQ fit values
    # FIXME: only need to fit no. compartments - 1 'partial_volume_' parameters
    init_param = dict.fromkeys(model.parameter_names)
    for param in model.parameter_names:
        init_param[param] = lsq_fit.fitted_parameters[param][mask]

    # perturb params for testing
    init_param[model.parameter_names[1]][0] = 10e-9  # Dpar
    init_param[model.parameter_names[2]][0] = 10e-9  # Diso
    #init_param[model.parameter_names[3]][0] = .5     # fpar

    # FIXME: prob want to keep as dictionary - would help with identifying orientation params in MCMC loop
    # create array of LSQ fit values, original values
    params_all_orig = dict_to_array(model, init_param)
    # create array of LSQ fit values, log transform variables (non-orientation only) (original -> log)
    params_all_tform = dict_to_array(model, tform_params(init_param, model, 'f'))

    # initialise sigma
    # FIXME: remove hardcoding of [0:nparams-1, :] once no. compartments fix sorted (line ~223)
    sigma = np.cov(params_all_tform[0:nparams-1, :])

    # initial weights for Metropolis-Hastings parameter sampling (f, D, D* from Orton, orientations guessed)
    # FIXME: calculate using parameter ranges
    # w = [.5, .5, .5, .1, .1]
    w = [.1, .1, .5, .5, .5]  # ori, ori, Dpar, Diso, fpar

    # measured signals
    E_dat = data[mask]

    # initial LSQ-estimated signals
    # FIXME: need to mask E_fit
    E_fit = np.zeros((nvox, ndw))
    for i in range(nvox):
        parameter_vector = params_all_orig[:, i]
        E_fit[i, :] = model.simulate_signal(acq_scheme, parameter_vector)

    # FIXME: maybe need to deal with nparams-1 once generalised for >2 compartments
    accepted = np.zeros(nparams-1)          # count total accepted moves for each param
    accepted_per_100 = np.zeros(nparams-1)  # track accepted moves for each param per 100 steps (to update weights)
    acceptance_rate = np.zeros([nparams-1, nsteps])   # track accepted moves for each param at each step

    # FIXME: need to track all parameter values at each step for a posterior distribution (currently tracking 1 voxel)
    # FIXME: also need to store final mu and sigma
    param_conv = np.zeros((nparams-1, nsteps))  # track parameter convergence
    tmpgibbs = np.zeros((nparams-1, nsteps))

    # FIXME: add in loop for different ROIs (version in bayesian-fitting-toy-example.py)

    # NB i (voxel loop) and j (MC loop) in keeping with Orton paper
    for j in range(0, nsteps):
        print(j)
        it = j + 1
        # Gibbs moves to update priors
        # sample mu from multivariate normal distribution defined by current parameter estimates
        # FIXME: remove hardcoding of [0:nparams-1, :] once generalised for >2 compartments
        m = np.mean(params_all_tform[0:nparams-1, :], axis=1)
        V = sigma / nvox
        mu = np.random.multivariate_normal(m, V)

        # sample sigma from inverse Wishart distribution (using newly updated mu)
        # NB scaled parameters used in calculation of priors in Metropolis-Hastings updates
        # FIXME: remove hardcoding of [0:nparams-1, :] once generalised for >2 compartments
        phi = np.sum([np.outer(params_all_tform[0:nparams-1, i] - mu,
                               params_all_tform[0:nparams-1, i] - mu)
                      for i in range(0, nvox)], axis=0)
        sigma = scipy.stats.invwishart(scale=phi, df=nvox - nparams).rvs()

        # Metropolis-Hastings parameter updates
        params_all_new = copy.copy(params_all_tform)
        for i in range(0, nvox):
            # FIXME: order of parameter loop affects convergence (?!)
            for p in [2, 3, 4, 0, 1]:  # range(nparams-1):
                # FIXME: use parameter names
                # sample parameter
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

                #if i == 0 and p == 2:
                #    tmp1 = copy.copy(params_all_tform[p, i])
                #    tmp2 = copy.copy(params_all_new[p, i])

                # compute acceptance
                y_i = copy.copy(E_dat[i, ])  # actual measured signal
                g_i = copy.copy(E_fit[i, ])  # model-predicted signal (old params)

                params_all_new = dict_to_array(model, tform_params(array_to_dict(model, params_all_new), model, 'r'))  # revert transform for signal calc (log -> original)
                parameter_vector = params_all_new[:, i]
                g_i_new = model.simulate_signal(acq_scheme, parameter_vector)  # model-predicted signal (new params)
                params_all_new = dict_to_array(model, tform_params(array_to_dict(model, params_all_new), model, 'f'))  # redo transform (original -> log)

                # calculate posteriors and PDFs (log scale)
                likelihood = (-ndw / 2) * np.log(np.inner(y_i, y_i) -
                                                 ((np.inner(y_i, g_i)) ** 2 / np.inner(g_i, g_i)))
                likelihood_new = (-ndw / 2) * np.log(np.inner(y_i, y_i) -
                                                     ((np.inner(y_i, g_i_new)) ** 2 / np.inner(g_i_new, g_i_new)))

                # FIXME: remove hardcoding of [0:nparams-1, :] once no. compartments fix sorted (line ~223)
                prior = np.log(scipy.stats.multivariate_normal.pdf(params_all_tform[0:nparams-1, i], mu, sigma, allow_singular=1))
                prior_new = np.log(scipy.stats.multivariate_normal.pdf(params_all_new[0:nparams-1, i], mu, sigma, allow_singular=1))

                alpha = np.min([0, (likelihood_new + prior_new) - (likelihood + prior)])
                r = np.log(np.random.uniform(0, 1))

                # reject new parameter value if criteria not met
                if r < alpha:
                    accepted[p] += 1
                    accepted_per_100[p] += 1
                    params_all_tform[p, i] = copy.copy(params_all_new[p, i])
                    E_fit[i, ] = copy.copy(g_i_new)
                    # FIXME: add this back in
                #            else:
                #                if Accepted / (it * nvox) < 0.23:
                #                    continue
                #                    # print("Stopping criterion met {}".format(Accepted/(it*nvox)))
                #                    # return Acceptance_rate

                # for plotting
                if i == 0:
                    param_conv[p, j] = copy.copy(params_all_tform[p, i])
                    #if p == 2:
                    #    print((np.exp(tmp1), np.exp(tmp2), np.exp(param_conv[p, j])))
                    # FIXME: remove hardcoding of [0:nparams-1, :] once no. compartments fix sorted (line ~223)
                    tmpgibbs[0:nparams-1, j] = copy.copy(mu)

                # acceptance_rate.append(accepted / (it * nvox))
                acceptance_rate[p, j] = accepted[p] / (it * nvox)

                #if i == 0 and p == 2:
                #    print([w[p], accepted_per_100[p]])

        if np.remainder(j, 100) == 0 and 0 < j <= burn_in / 2:
            w = w * (101 / (2 * (101-(accepted_per_100/nvox))))
            accepted_per_100 = np.zeros((w.__len__()))

    params_all = dict_to_array(model, tform_params(array_to_dict(model, params_all_new), model, 'r'))
    # FIXME: don't hardcode calculation of last compartment fraction
    # param_conv = dict_to_array(model, tform_params(array_to_dict(model, param_conv), model, 'r'))

    return acceptance_rate, param_conv, params_all
