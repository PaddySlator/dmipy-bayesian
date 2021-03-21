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
    params_all[1, 0] = 10e-9  # Diso
    params_all[2, 0] = .5    # fpar
    return params_all, E_sim, E_fit, nvox, params_all_correct


def run_optimization(params_all, E_sim, E_fit, nvox, iter):
    acq_scheme, nmeas, stick, ball, ballstick = sign_par()

    # TO DO: log transform variables (non-orientation only)
    #params_all[0, :] = np.log(params_all[0, :])
    #params_all[1, :] = np.log(params_all[1, :])
    #params_all[2, :] = np.log(params_all[2, :]) - np.log(1 - params_all[2, :])

    # initialise sigma
    sigma = np.cov(params_all)

    # TO DO: tune weights for Metropolis-Hastings parameter sampling (f, D, D* from Orton, orientations guessed)
    w = [.5e-9, .5e-9, .1, .02, .02]
    #w = [1, 1, .1, .1, .1]
    w_acc = 0

    nMCMCsteps = iter
    T = compute_temp_schedule(2000, 10 ** -3, nMCMCsteps)

    Accepted = 1
    Acceptance_rate = []
    Acceptance_rate.append(Accepted)

    tmppar = np.zeros((3, nMCMCsteps))
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
        mu_scaled = mu  # * [1e9, 1e9, 1, 1, 1]

        # sample sigma from inverse Wishart distribution (using newly updated mu)
        # NB scaled parameters used in calculation of priors in Metropolis-Hastings updates
        phi = np.sum([np.outer(params_all[:, i] - mu,
                               params_all[:, i] - mu)
                      for i in range(0, nvox)], axis=0)
        phi_scaled = np.sum([np.outer((params_all[:, i] - mu) * [1e9, 1e9, 1, 1, 1],
                                      (params_all[:, i] - mu) * [1e9, 1e9, 1, 1, 1])
                             for i in range(0, nvox)], axis=0)
        sigma = scipy.stats.invwishart(scale=phi, df=nvox - 5).rvs()
        sigma_scaled = scipy.stats.invwishart(scale=phi_scaled, df=nvox - 5).rvs()

        # Metropolis-Hastings parameter updates
        params_all_new = copy.copy(params_all)
        for i in range(0, nvox):
            # TO DO: loop over parameters (Lizzie)
            for p in range(3):
            # p = 0  # for now just look at
                # sample parameter
                params_all_new[p, i] = np.random.normal(params_all[p, i], w[p])
                if i == 0 and p == 0:
                    print([params_all_new[p, i], params_all[p, i]])
                # compute acceptance
                y_i = copy.copy(E_sim[i, ])  # actual measured signal
                g_i = copy.copy(E_fit[i, ])  # model-predicted signal (old params)
                parameter_vector = ballstick.parameters_to_parameter_vector(
                    C1Stick_1_mu=[params_all_new[3, i], params_all_new[4, i]],
                    C1Stick_1_lambda_par=params_all_new[0, i],
                    G1Ball_1_lambda_iso=params_all_new[1, i],
                    partial_volume_0=params_all_new[2, i],
                    partial_volume_1=1 - params_all_new[2, i])
                g_i_new = ballstick.simulate_signal(acq_scheme, parameter_vector)  # model-predicted signal (new params)
                # calculate posteriors and PDFs (log scale)
                likelihood = (-nmeas / 2) * np.log(np.inner(y_i, y_i) -
                                                   ((np.inner(y_i, g_i)) ** 2 / np.inner(g_i, g_i)))
                likelihood_new = (-nmeas / 2) * np.log(np.inner(y_i, y_i) -
                                                       (np.inner(y_i, g_i_new) ** 2 / np.inner(g_i_new, g_i_new)))

                if p <= 2:  # non-orientation parameters
                    # both priors are the same without scaling
                    prior = np.log(scipy.stats.multivariate_normal.pdf(params_all[:, i], mu, sigma, allow_singular=1))
                    prior_new = np.log(scipy.stats.multivariate_normal.pdf(params_all_new[:, i], mu, sigma, allow_singular=1))
                    # scaling parameters helps
                    prior_scaled = np.log(scipy.stats.multivariate_normal.pdf(params_all[:, i] * [1e9, 1e9, 1, 1, 1],
                                                                              mu_scaled, sigma_scaled,
                                                                              allow_singular=1))
                    prior_new_scaled = np.log(scipy.stats.multivariate_normal.pdf(params_all_new[:, i] * [1e9, 1e9, 1, 1, 1],
                                                            mu_scaled, sigma_scaled, allow_singular=1))
                else:  # orientation parameters (Chris to modify) (scaling here?)
                    prior = 1
                    prior_new = 1
                    prior_scaled = 1
                    prior_new_scaled = 1

                alpha = np.min([0, (likelihood_new + prior_new) - (likelihood + prior)])
                alpha_scaled = np.min([0, (likelihood_new + prior_new_scaled) - (likelihood + prior_scaled)])
                r = np.log(np.random.uniform(0, 1))

                # reject new parameter value if criteria not met
                if r < alpha_scaled:
                    Accepted += 1
                    params_all[p, i] = copy.copy(params_all_new[p, i])
                    E_fit[i, ] = copy.copy(g_i_new)
                #            else:
                #                if Accepted / (it * nvox) < 0.23:
                #                    continue
                #                    # print("Stopping criterion met {}".format(Accepted/(it*nvox)))
                #                    # return Acceptance_rate

                # for plotting
                if i == 0:
                    tmppar[p, j] = copy.copy(params_all[p, i])
                    tmpgibbs[:, j] = copy.copy(mu)

                Acceptance_rate.append(Accepted / (it * nvox))

                # TO DO: update weights every 100 steps
        if j%50 == 0:
            w = [v * ( 51/(2*(51 - w_acc/(it*nvox*5))) ) for v in w]
            print(w)
            w_acc = 0


    return Acceptance_rate, tmppar


def main():
    params_all, E_sim, E_fit, nvox, params_all_correct = load_data()

    params_all_init = copy.copy(params_all)
    E_sim_init = copy.copy(E_sim)
    E_fit_init = copy.copy(E_fit)

    params_all = copy.copy(params_all_init)
    E_sim = copy.copy(E_sim_init)
    E_fit = copy.copy(E_fit_init)

    nMCMCsteps = 1000
    Proc_start = time.time()
    Acceptance_rate, tmppar = run_optimization(params_all, E_sim, E_fit, nvox, nMCMCsteps)
    Compute_time(Proc_start, time.time())

    color = 'tab:blue'
    fig, ax = plt.subplots()
    ax.set_ylabel("Dpar", color=color)
    ax.set_xlabel("MCMC iteration", color=color)
    ax.scatter(range(nMCMCsteps), tmppar[0, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("Diso", color=color)
    ax.set_xlabel("MCMC iteration", color=color)
    ax.scatter(range(nMCMCsteps), tmppar[1, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("fpar", color=color)
    ax.set_xlabel("MCMC iteration", color=color)
    ax.scatter(range(nMCMCsteps), tmppar[2, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)




    fig, ax = plt.subplots()
    ax.set_ylabel("freq", color=color)
    ax.set_xlabel("Dpar", color=color)
    ax.hist(tmppar[0, 500:-1], bins=50)
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("freq", color=color)
    ax.set_xlabel("Diso", color=color)
    ax.hist(tmppar[1, 500:-1], bins=50)
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("freq", color=color)
    ax.set_xlabel("fpar", color=color)
    ax.hist(tmppar[2, 500:-1], bins=50)
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    print((params_all_init[0, 0], params_all_correct[0, 0], tmppar[0, -1]))
    print((params_all_init[1, 0], params_all_correct[1, 0], tmppar[1, -1]))
    print((params_all_init[2, 0], params_all_correct[2, 0], tmppar[2, -1]))

    tmpim = np.reshape(params_all_correct, (5, 10, 10))
    fig, ax = plt.subplots()
    plt.title('D stick')
    plt.imshow(tmpim[0, :, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig, ax = plt.subplots()
    color = 'tab:blue'
    ax.set_ylabel("Acceptance Rate", color=color)
    ax.plot(np.arange(len(Acceptance_rate)), Acceptance_rate, marker=",", color=color)
    ax.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
