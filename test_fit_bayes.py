# load some necessary modules
from dmipy.core import modeling_framework
from os.path import join
import numpy as np
import scipy.stats
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import time
import math
from fit_bayes import fit_bayes, tform_params  # , dict_to_array, array_to_dict

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
    dimx = 9
    dimy = 9
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
            if fpar_sim[x, y] < 0:  # make sure 0 < f < 1
                fpar_sim[x, y] = 0.01
            elif fpar_sim[x, y] > 0:  # make sure 0 < f < 1
                fpar_sim[x, y] = 0.99
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
    params_all = copy(params_all_correct)

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

    # create mask with regional ROIs
    mask = np.ones([dimx, dimy])
    nx = int(np.round(dimx/4))
    ny = int(np.round(dimy/4))
    mask[nx:dimx-nx, ny:dimy-ny] = 2
    mask = np.ndarray.flatten(mask)  # flatten signal to get [nvox x nmeas] array

    return params_all, E_sim, E_fit, nvox, params_all_correct, mask


def main():
    # FIXME: E_sim and mask need to have dim = [x, y, z, ndw]
    params_all, E_sim, E_fit, nvox, params_all_correct, mask = load_data()
    # mask = np.ones(E_sim.shape[0])
    acq_scheme, nmeas, stick, ball, ballstick = sign_par()

    params_all_init = copy(params_all)
    E_sim_init = copy(E_sim)
    E_fit_init = copy(E_fit)

    # params_all = copy(params_all_init)
    E_sim = copy(E_sim_init)
    E_fit = copy(E_fit_init)

    # generalise
    model = deepcopy(ballstick)
    acq_scheme = acq_scheme
    data = E_sim
    #mask = data[..., 0] > 0

    nsteps = 1000
    burn_in = 500

    Proc_start = time.time()
    Acceptance_rate, param_conv, params_all_new, params_all_orig = fit_bayes(model, acq_scheme, data, mask, nsteps, burn_in)
    Compute_time(Proc_start, time.time())

    # remove dependent volume fraction from model
    dependent_fraction = model.partial_volume_names[-1]
    model_reduced = deepcopy(model)
    del model_reduced.parameter_ranges[dependent_fraction]
    del model_reduced.parameter_cardinality[dependent_fraction]
    del model_reduced.parameter_scales[dependent_fraction]
    del model_reduced.parameter_types[dependent_fraction]
    del model_reduced.parameter_optimization_flags[dependent_fraction]

    params_all_orig_vec = np.transpose(model.parameters_to_parameter_vector(**params_all_orig))
    params_all_new_vec = np.transpose(model.parameters_to_parameter_vector(**params_all_new))
    param_conv_vec = np.vstack([param_conv['C1Stick_1_lambda_par'][0, :],
                                param_conv['G1Ball_1_lambda_iso'][0, :],
                                param_conv['partial_volume_0'][0, :],
                                param_conv['C1Stick_1_mu'][0, 0, :],
                                param_conv['C1Stick_1_mu'][1, 0, :]])

    # print: initialisation, correct value, mean (after burn-in) Bayes-fitted value
    print((params_all_init[0, 0], params_all_correct[0, 0], np.mean(param_conv_vec[0, burn_in:-1])))  # , params_all_new[2, 0]))
    print((params_all_init[1, 0], params_all_correct[1, 0], np.mean(param_conv_vec[1, burn_in:-1])))  # , params_all_new[3, 0]))
    print((params_all_init[2, 0], params_all_correct[2, 0], np.mean(param_conv_vec[2, burn_in:-1])))  # , params_all_new[4, 0]))
    print((params_all_init[3, 0], params_all_correct[3, 0], np.mean(param_conv_vec[3, burn_in:-1])))  # , params_all_new[0, 0]))
    print((params_all_init[4, 0], params_all_correct[4, 0], np.mean(param_conv_vec[4, burn_in:-1])))  # , params_all_new[1, 0]))

    # plot parameter convergence
    color = 'tab:blue'
    fig, ax = plt.subplots()
    ax.set_ylabel("Dpar", color=color)
    ax.set_xlabel("MCMC iteration", color=color)
    ax.scatter(range(nsteps), param_conv_vec[0, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("Diso", color=color)
    ax.set_xlabel("MCMC iteration", color=color)
    ax.scatter(range(nsteps), param_conv_vec[1, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("fpar", color=color)
    ax.set_xlabel("MCMC iteration", color=color)
    ax.scatter(range(nsteps), param_conv_vec[2, :])
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    # plot parameter distributions after burn-in period
    fig, ax = plt.subplots()
    ax.set_ylabel("freq", color=color)
    ax.set_xlabel("Dpar", color=color)
    ax.hist(param_conv_vec[0, burn_in:-1], bins=50)
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("freq", color=color)
    ax.set_xlabel("Diso", color=color)
    ax.hist(param_conv_vec[1, burn_in:-1], bins=50)
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("freq", color=color)
    ax.set_xlabel("fpar", color=color)
    ax.hist(param_conv_vec[2, burn_in:-1], bins=50)
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    tmpim = np.reshape(params_all_correct, (5, int(np.sqrt(nvox)), int(np.sqrt(nvox))))
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
