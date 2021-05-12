# load some necessary modules
from dmipy.core import modeling_framework
from os.path import join
import numpy as np
import scipy.stats
import copy
import matplotlib.pyplot as plt
import time
import math

# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes

# ball stick and spherical mean ball-stick model
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel, MultiCompartmentModel


# Make axes square
def make_square_axes(ax):
    ax.set_aspect(1 / ax.get_data_ratio())


# update figure font size
def update_font_size_all(plt, ax, fontsz, legend=1, cbar=1):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsz)
    if legend:
        ax.legend(fontsize=fontsz)
    if cbar:
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=fontsz)


def Compute_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("/nTOTAL OPTIMIZATION TIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    time_string = ("TOTAL TIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return time_string


def compute_temp_schedule(startTemp, endTemp, MAX_ITER):
    SA_schedule = []
    it = np.arange(MAX_ITER + 1)
    SA_schedule.append([-math.log(endTemp / startTemp) / i for i in it])
    return SA_schedule[0]


def sign_par():
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    nmeas = len(acq_scheme.bvalues)

    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    ballstick = MultiCompartmentModel(models=[stick, ball])
    return acq_scheme, nmeas, stick, ball, ballstick


def load_data():
    acq_scheme, nmeas, stick, ball, ballstick = sign_par()
    # baseline parameters for the simulated ball-stick voxel
    Dpar = 1.7e-9  # in m^2/s
    Diso = 2.5e-9  # in m^2/s
    fpar = 0.3
    # fiso = 1-fpar
    stick_ori = (np.pi, 0)

    # simulate a simple 10x10 image
    dimx = 10
    dimy = 10
    nvox = dimx * dimy

    Dpar_sim = np.zeros((dimx, dimy))
    Diso_sim = np.zeros((dimx, dimy))
    fpar_sim = np.zeros((dimx, dimy))
    stick_ori_sim = np.zeros((dimx, dimy, 2))
    E_sim = np.zeros((dimx, dimy, nmeas))
    E_fit = np.zeros((dimx, dimy, nmeas))
    for x in range(0, dimx):
        for y in range(0, dimy):
            # vary parameters
            Dpar_sim[x, y] = Dpar + np.random.normal(0, 0.1e-9)
            Diso_sim[x, y] = Diso + np.random.normal(0, 0.1e-9)
            fpar_sim[x, y] = fpar + np.random.normal(0, 0.1)
            stick_ori_sim[x, y, :] = stick_ori + np.random.normal(0, 0.1, (1, 2))
            # generate signal
            parameter_vector = ballstick.parameters_to_parameter_vector(
                C1Stick_1_mu=stick_ori_sim[x, y,],
                C1Stick_1_lambda_par=Dpar_sim[x, y],
                G1Ball_1_lambda_iso=Diso_sim[x, y],
                partial_volume_0=fpar_sim[x, y],
                partial_volume_1=1 - fpar_sim[x, y])
            E_sim[x, y, :] = ballstick.simulate_signal(acq_scheme, parameter_vector) + np.random.normal(0, .01)
            E_fit[x, y, :] = ballstick.fit(acq_scheme, E_sim[x, y,]).predict()

    E_sim = np.reshape(E_sim, (nvox, nmeas))  # flatten signal to get [nvox x nmeas] array
    E_fit = np.reshape(E_fit, (nvox, nmeas))  # flatten signal to get [nvox x nmeas] array

    # TO DO: should be from LSQs fit - cheat and use GT for now
    # initial voxel-wise estimates
    Dpar_init = Dpar_sim
    Diso_init = Diso_sim
    fpar_init = fpar_sim
    fiso_init = 1 - fpar_sim
    stick_ori_init = stick_ori_sim

    # flatten params to get [5 x nvox] array
    params_all_correct = np.array([np.ndarray.flatten(Dpar_init),
                                   np.ndarray.flatten(Diso_init),
                                   np.ndarray.flatten(fpar_init),
                                   np.ndarray.flatten(stick_ori_init[:, :, 0]),
                                   np.ndarray.flatten(stick_ori_init[:, :, 1])])
    params_all = copy.copy(params_all_correct)

    # TEST: perturb non-orientation variables
    x = 0
    y = 0
    parameter_vector = ballstick.parameters_to_parameter_vector(
        C1Stick_1_mu=stick_ori_sim[x, y],
        C1Stick_1_lambda_par=10e-9, # Dpar_sim[x, y],
        G1Ball_1_lambda_iso=10e-9,  # Diso_sim[x, y],
        partial_volume_0=.5,        # fpar_sim[x, y],
        partial_volume_1=1 - fpar_sim[x, y])
    E_fit[0, :] = ballstick.simulate_signal(acq_scheme, parameter_vector)
    params_all[0, 0] = 10e-9  # Dpar
    params_all[1, 0] = 10e-9  # Diso_sim[x, y]
    params_all[2, 0] = .5     # fpar_sim[x, y]
    return params_all, E_sim, E_fit, nvox, params_all_correct


def tform_params(params_all, direction):
    if direction == 'f':
        params_all[0, :] = np.log(params_all[0, :])
        params_all[1, :] = np.log(params_all[1, :])
        params_all[2, :] = np.log(params_all[2, :]) - np.log(1 - params_all[2, :])

    elif direction == 'r':
        params_all[0, :] = np.exp(params_all[0, :])
        params_all[1, :] = np.exp(params_all[1, :])
        params_all[2, :] = np.exp(params_all[2, :]) / (1 + np.exp(params_all[2, :]))
        params_all[2, :] = [1 if np.isnan(x) else x for x in params_all[2, :]]

    return params_all


def scale_params(params_all, direction):
    if direction == 'f':
        params_all[0, :] = params_all[0, :] * 1e9
        params_all[1, :] = params_all[1, :] * 1e9

    elif direction == 'r':
        params_all[0, :] = params_all[0, :] * 1e-9
        params_all[1, :] = params_all[1, :] * 1e-9

    return params_all


def run_optimization(params_all, E_sim, E_fit, nvox, nMCMCsteps, burn_in):
    acq_scheme, nmeas, stick, ball, ballstick = sign_par()

    # log transform variables (non-orientation only) (original -> log)
    params_all = tform_params(params_all, 'f')

    # initialise sigma
    sigma = np.cov(params_all)

    # initial weights for Metropolis-Hastings parameter sampling (f, D, D* from Orton, orientations guessed)
    w = [.2, .5, .5, .1, .1]

    T = compute_temp_schedule(2000, 10 ** -3, nMCMCsteps)

    accepted = np.zeros((w.__len__()))          # count total accepted moves for each param
    accepted_per_100 = np.zeros((w.__len__()))  # track accepted moves for each param per 100 steps (to update weights)
    acceptance_rate = np.zeros([(w.__len__()), nMCMCsteps])   # track accepted moves for each param at each step

    param_conv = np.zeros((5, nMCMCsteps))  # track parameter convergence
    tmpgibbs = np.zeros((5, nMCMCsteps))

    # NB i (voxel loop) and j (MC loop) in keeping with Orton paper
    for j in range(0, nMCMCsteps):
        print(j)
        it = j + 1
        # Gibbs moves to update priors
        # sample mu from multivariate normal distribution defined by current parameter estimates
        m = np.mean(params_all, axis=1)
        V = sigma / nvox
        mu = np.random.multivariate_normal(m, V)

        # sample sigma from inverse Wishart distribution (using newly updated mu)
        # NB scaled parameters used in calculation of priors in Metropolis-Hastings updates
        phi = np.sum([np.outer(params_all[:, i] - mu,
                               params_all[:, i] - mu)
                      for i in range(0, nvox)], axis=0)
        sigma = scipy.stats.invwishart(scale=phi, df=nvox - 5).rvs()

        # Metropolis-Hastings parameter updates
        params_all_new = copy.copy(params_all)
        for i in range(0, nvox):
            for p in range(5):
                # sample parameter
                if p <= 2:    # Dpar, Diso, fpar
                    params_all_new[p, i] = np.random.normal(params_all[p, i], w[p])
                elif p == 3:  # stick orientation (theta)
                    u = np.random.uniform(0, 1, 1)
                    params_all_new[p, i] = np.arccos(1 - 2*u)
                elif p == 4:  # stick orientation (phi)
                    u = np.random.uniform(0, 1, 1)
                    params_all_new[p, i] = 2 * np.pi * u

                # compute acceptance
                y_i = copy.copy(E_sim[i, ])  # actual measured signal
                g_i = copy.copy(E_fit[i, ])  # model-predicted signal (old params)

                params_all_new = tform_params(params_all_new, 'r')  # revert transform for signal calc (log -> original)
                if i == 0:
                    print([j, i, p, params_all_new[[0, 1, 2], i]])
                parameter_vector = ballstick.parameters_to_parameter_vector(
                                    C1Stick_1_mu=[params_all_new[3, i], params_all_new[4, i]],
                                    C1Stick_1_lambda_par=params_all_new[0, i],
                                    G1Ball_1_lambda_iso=params_all_new[1, i],
                                    partial_volume_0=params_all_new[2, i],
                                    partial_volume_1=1 - params_all_new[2, i])
                g_i_new = ballstick.simulate_signal(acq_scheme, parameter_vector)  # model-predicted signal (new params)
                params_all_new = tform_params(params_all_new, 'f')  # redo transform (original -> log)

                # calculate posteriors and PDFs (log scale)
                likelihood = (-nmeas / 2) * np.log(np.inner(y_i, y_i) -
                                                   ((np.inner(y_i, g_i)) ** 2 / np.inner(g_i, g_i)))
                likelihood_new = (-nmeas / 2) * np.log(np.inner(y_i, y_i) -
                                                       (np.inner(y_i, g_i_new) ** 2 / np.inner(g_i_new, g_i_new)))

                if p <= 2:  # non-orientation parameters
                    prior = np.log(scipy.stats.multivariate_normal.pdf(params_all[:, i], mu, sigma, allow_singular=1))
                    prior_new = np.log(scipy.stats.multivariate_normal.pdf(params_all_new[:, i], mu, sigma, allow_singular=1))
                else:  # orientation parameters (the prior probability of orientation parameter samples are calculated based on gaussian ROI prior pdf, as above)
                    prior = log(scipy.stats.multivariate_normal.pdf(params_all[:, i], mu, sigma, allow_singular=1))
                    prior_new = np.log(scipy.stats.multivariate_normal.pdf(params_all_new[:, i], mu, sigma, allow_singular=1))

                alpha = np.min([0, (likelihood_new + prior_new) - (likelihood + prior)])
                r = np.log(np.random.uniform(0, 1))

                # reject new parameter value if criteria not met
                if r < alpha:
                    accepted[p] += 1
                    accepted_per_100[p] += 1
                    params_all[p, i] = copy.copy(params_all_new[p, i])
                    E_fit[i, ] = copy.copy(g_i_new)
                #            else:
                #                if Accepted / (it * nvox) < 0.23:
                #                    continue
                #                    # print("Stopping criterion met {}".format(Accepted/(it*nvox)))
                #                    # return Acceptance_rate

                # for plotting
                if i == 0:
                    param_conv[p, j] = copy.copy(params_all[p, i])
                    tmpgibbs[:, j] = copy.copy(mu)

                # acceptance_rate.append(accepted / (it * nvox))
                acceptance_rate[p, j] = accepted[p] / (it * nvox)

                # TO DO: update weights every 100 steps
        if np.remainder(j, 100) == 0 and j < burn_in/2:
            # w = [v * ( 51/(2*(51 - w_acc/(it*nvox*5))) ) for v in w]
            w = w * (101 / (2 * (101-(accepted_per_100/nvox))))
            accepted_per_100 = np.zeros((w.__len__()))
            #print(w)

    params_all = tform_params(params_all, 'r')
    param_conv = tform_params(param_conv, 'r')

    return acceptance_rate, param_conv, params_all


def main():
    params_all, E_sim, E_fit, nvox, params_all_correct = load_data()

    params_all_init = copy.copy(params_all)
    E_sim_init = copy.copy(E_sim)
    E_fit_init = copy.copy(E_fit)

    params_all = copy.copy(params_all_init)
    E_sim = copy.copy(E_sim_init)
    E_fit = copy.copy(E_fit_init)

    nMCMCsteps = 1000
    burn_in = 500
    Proc_start = time.time()
    Acceptance_rate, param_conv, params_all_new = run_optimization(params_all, E_sim, E_fit, nvox, nMCMCsteps, burn_in)
    Compute_time(Proc_start, time.time())

    # print: initalisation, correct value, Bayes-fitted value
    print((params_all_init[0, 0], params_all_correct[0, 0], params_all_new[0, 0]))
    print((params_all_init[1, 0], params_all_correct[1, 0], params_all_new[1, 0]))
    print((params_all_init[2, 0], params_all_correct[2, 0], params_all_new[2, 0]))

    # plot parameter convergence
    color = 'tab:blue'
    fig, ax = plt.subplots()
    ax.set_ylabel("Dpar", color=color)
    ax.set_xlabel("MCMC iteration", color=color)
    ax.scatter(range(nMCMCsteps), param_conv[0, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("Diso", color=color)
    ax.set_xlabel("MCMC iteration", color=color)
    ax.scatter(range(nMCMCsteps), param_conv[1, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("fpar", color=color)
    ax.set_xlabel("MCMC iteration", color=color)
    ax.scatter(range(nMCMCsteps), param_conv[2, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    # plot parameter distributions after burn-in period
    fig, ax = plt.subplots()
    ax.set_ylabel("freq", color=color)
    ax.set_xlabel("Dpar", color=color)
    ax.hist(param_conv[0, burn_in:-1], bins=50)
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("freq", color=color)
    ax.set_xlabel("Diso", color=color)
    ax.hist(param_conv[1, burn_in:-1], bins=50)
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("freq", color=color)
    ax.set_xlabel("fpar", color=color)
    ax.hist(param_conv[2, burn_in:-1], bins=50)
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    tmpim = np.reshape(params_all_correct, (5, 10, 10))
    fig, ax = plt.subplots()
    plt.title('D stick')
    plt.imshow(tmpim[0, :, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # plot acceptance rate
    fig, ax = plt.subplots()
    color = 'tab:blue'
    ax.set_ylabel("Acceptance Rate", color=color)
    ax.plot(np.arange(len(Acceptance_rate)), Acceptance_rate, marker=",", color=color)
    ax.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
