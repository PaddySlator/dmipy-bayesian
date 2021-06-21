# load some necessary modules
from dmipy.core import modeling_framework
from os.path import join
import os
from os import listdir
import numpy as np
import nibabel as nib
from os.path import join as pjoin
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
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues


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

def load_real_data(path_ROIs, path_diff):
    acq_scheme, nmeas, stick, ball, ballstick = sign_par()
    bvalues = np.loadtxt(join(path_diff, 'test/bvals.txt'))  # given in s/mm^2
    bvalues_SI = bvalues * 1e6  # now given in SI units as s/m^2
    gradient_directions = np.loadtxt(join(path_diff, 'test/bvecs.txt'))  # on the unit sphere

    # The delta and Delta times we know from the HCP documentation in seconds
    delta = 0.0106  
    Delta = 0.0431 

    # The acquisition scheme used in the toolbox is then created as follows:
    acq_scheme = acquisition_scheme_from_bvalues(bvalues_SI, gradient_directions, delta, Delta)


    ##-- Reading the DWI nifti image
    from dipy.io.image import load_nifti
    image_path= pjoin(path_diff, "data.nii.gz")
    data, affine, img = load_nifti(image_path, return_img=True)

    ROIs = np.zeros_like(data)

    for idx, roi_im in enumerate(listdir(path_ROIs)):
        roi_img = nib.load((pjoin(path_ROIs, roi_im))).get_fdata()
        ROIs[roi_img>0] = idx



    # plotting an axial slice
    import matplotlib.pyplot as plt
    axial_middle = data.shape[2] // 2
    plt.figure('Axial slice')
    plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
    plt.show()

    return acq_scheme, data, ballstick, ROIs


def load_data():
    acq_scheme, nmeas, stick, ball, ballstick = sign_par()
    # baseline parameters for the simulated ball-stick voxel
    Dpar = 1.7e-9  # in m^2/s
    Diso = 2.5e-9  # in m^2/s
    fpar = 0.3
    # fiso = 1-fpar
    stick_ori = (np.pi, 0)

    # simulate a simple 10x10 image
    dimx = 5
    dimy = 5
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
            if fpar_sim[x, y] < .01:  # make sure 0 < f < 1
                fpar_sim[x, y] = 0.01
            elif fpar_sim[x, y] > .99:  # make sure 0 < f < 1
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
    Dpar_init = Dpar_si
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
    '''
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
    '''

    # create mask with regional ROIs
    mask = np.ones([dimx, dimy])
    #nx = int(np.round(dimx/4))
    #ny = int(np.round(dimy/4))
    #mask[nx:dimx-nx, ny:dimy-ny] = 2
    mask = np.ndarray.flatten(mask)  # flatten signal to get [nvox x nmeas] array

    return params_all, E_sim, E_fit, nvox, params_all_correct, mask


def main():
    # FIXME: E_sim and mask need to have dim = [x, y, z, ndw]
    # params_all, E_sim, E_fit, nvox, params_all_correct, mask = load_data()
    path_ROIs = "/media/full/DATA/Software/Bayes_compartment_inference/Real_data/seg/103818_1"
    path_diff = "/media/full/DATA/Software/Bayes_compartment_inference/Real_data/TestRetestData/103818_1"
    acq_scheme, data, ballstick, ROIs = load_real_data(path_ROIs, path_diff)
    # mask = np.ones(E_sim.shape[0])
    # acq_scheme, nmeas, stick, ball, ballstick = sign_par()

    # params_all_init = copy(params_all)
    # E_sim_init = copy(E_sim)
    # E_fit_init = copy(E_fit)

    # # params_all = copy(params_all_init)
    # E_sim = copy(E_sim_init)
    # E_fit = copy(E_fit_init)

    # # generalise
    model = deepcopy(ballstick)
    # acq_scheme = acq_scheme
    # data = E_sim
    #mask = data[..., 0] > 0

    nsteps = 5000
    burn_in = 1200

    Proc_start = time.time()
    acceptance_rate, param_conv, params_all_new, params_all_orig, likelihood_stored = fit_bayes(model, acq_scheme, data, ROIs, nsteps, burn_in)
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
    print((params_all_orig['C1Stick_1_lambda_par'][0], params_all_correct[0, 0], np.mean(param_conv_vec[0, burn_in:-1])))  # , params_all_new[2, 0]))
    print((params_all_orig['G1Ball_1_lambda_iso'][0], params_all_correct[1, 0], np.mean(param_conv_vec[1, burn_in:-1])))  # , params_all_new[3, 0]))
    print((params_all_orig['partial_volume_0'][0], params_all_correct[2, 0], np.mean(param_conv_vec[2, burn_in:-1])))  # , params_all_new[4, 0]))
    print((params_all_orig['C1Stick_1_mu'][0, 0], params_all_correct[3, 0], np.mean(param_conv_vec[3, burn_in:-1])))  # , params_all_new[0, 0]))
    print((params_all_orig['C1Stick_1_mu'][0, 1], params_all_correct[4, 0], np.mean(param_conv_vec[4, burn_in:-1])))  # , params_all_new[1, 0]))

    # plot parameter convergence
    fig, ax = plt.subplots()
    ax.set_ylabel("Dpar")
    ax.set_xlabel("MCMC iteration")
    ax.scatter(range(nsteps), param_conv_vec[0, :], color='tab:red')
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("Diso")
    ax.set_xlabel("MCMC iteration")
    ax.scatter(range(nsteps), param_conv_vec[1, :], color='tab:green')
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("fpar")
    ax.set_xlabel("MCMC iteration")
    ax.scatter(range(nsteps), param_conv_vec[2, :], color='tab:blue')
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    # plot parameter distributions after burn-in period
    fig, ax = plt.subplots()
    ax.set_ylabel("freq")
    ax.set_xlabel("Dpar")
    ax.hist(param_conv_vec[0, burn_in:-1], color='tab:red')
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("freq")
    ax.set_xlabel("Diso")
    ax.hist(param_conv_vec[1, burn_in:-1], color='tab:green')
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    fig, ax = plt.subplots()
    ax.set_ylabel("freq")
    ax.set_xlabel("fpar")
    ax.hist(param_conv_vec[2, burn_in:-1], color='tab:blue')
    update_font_size_all(plt, ax, 20, legend=0, cbar=0)
    make_square_axes(ax)

    # plot acceptance rate
    fig, ax = plt.subplots()
    ax.set_ylabel("Acceptance Rate")
    ax.scatter(range(nsteps), acceptance_rate['C1Stick_1_lambda_par'][0], color='tab:red')
    ax.scatter(range(nsteps), acceptance_rate['G1Ball_1_lambda_iso'][0], color='tab:green')
    ax.scatter(range(nsteps), acceptance_rate['partial_volume_0'][0], color='tab:blue')
    ax.legend(['Dpar', 'Diso', 'fpar'])

    # plot likelihood
    fig, ax = plt.subplots()
    ax.set_ylabel("Likelihood")
    ax.scatter(range(nsteps), likelihood_stored['C1Stick_1_lambda_par'][0], color='tab:red')
    ax.scatter(range(nsteps), likelihood_stored['G1Ball_1_lambda_iso'][0], color='tab:green')
    ax.scatter(range(nsteps), likelihood_stored['partial_volume_0'][0], color='tab:blue')
    ax.legend(['Dpar', 'Diso', 'fpar'])

if __name__ == '__main__':
    main()
