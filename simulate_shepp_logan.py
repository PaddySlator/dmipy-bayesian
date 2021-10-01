#Extension of the dmipy software package for Bayesian hierarchical model fitting

#Copyright (C) 2021 Elizabeth Powell, Matteo Battocchio, Chris Parker, Paddy Slator 

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program. If not, see <https://www.gnu.org/licenses/>.


# load some necessary modules
# from dmipy.core import modeling_framework
# from os.path import join
import numpy as np
import scipy.stats
from copy import copy, deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import time
import math
from phantominator import shepp_logan, dynamic
from importlib import reload
import itertools
import shelve

import fit_bayes
from fit_bayes import fit, tform_params
fit_bayes = reload(fit_bayes)

# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes

# ball stick and spherical mean ball-stick model
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel, MultiCompartmentModel


# save workspace variables; vars = dir()
def save_workspace(filename):
    print(filename)
    shelf = shelve.open(filename, "n")
    for key in globals():
        try:
            # print(key)
            shelf[key] = globals()[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()


# load workspace variables
def load_workspace(filename):
    shelf = shelve.open(filename)
    print(shelf)
    for key in shelf:
        try:
            print(key)
            globals()[key] = shelf[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()


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


def compute_time(start, end):
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


# dims = [nx, ny, nz]
def load_shepp_phantom_old(dims, model, acq_scheme):
    # load Shepp-Logan phantom
    m0, t1, t2 = shepp_logan(dims, MR=True, zlims=(-.25, .25))
    for i, j in itertools.product(range(dims[0]), range(dims[1])):
        if m0[i, j] < .5:
            m0[i, j] = 0
        elif (m0[i, j] >= .5) & (m0[i, j] < .9):
            m0[i, j] = .7
        elif m0[i, j] >= .9:
            m0[i, j] = 1

    # create mask with integer values
    roi_vals = np.unique(m0)[np.unique(m0) > 0]
    mask = np.zeros(dims).flatten()
    for roi in range(roi_vals.__len__()):
        idx_roi = [xx for xx, x in enumerate(m0.flatten() == roi_vals[roi]) if x]
        if idx_roi.__len__() > 8:  # avoid df error in sigma calculation, i.e. must have df = (nvox - nparams - 1) >= 1
            mask[idx_roi] = roi + 1
    idx_all = [xx for xx, x in enumerate(mask > 0) if x]
    idx_all = np.linspace(0, np.prod(dims)-1, np.prod(dims))

    # use the m0 image to create ballstick model params; add small variations to avoid non-semi definite errors in sigma
    Dpar = np.zeros(np.prod(dims))
    Diso = np.zeros(np.prod(dims))
    fpar = np.zeros(np.prod(dims))
    mu = np.zeros([np.prod(dims), 2])

    # Dpar[idx_all] = [(1.7e-9 + np.random.normal(0, 0.1e-9)) * m0.flatten()[i] for i in idx_all]
    # Diso[idx_all] = [(2.5e-9 + np.random.normal(0, 0.1e-9)) * m0.flatten()[i] for i in idx_all]
    # fpar[idx_all] = [(.3 + np.random.normal(0, 0.025)) * m0.flatten()[i] for i in idx_all]
    # mu[idx_all] = [([np.pi / 2, 0] + np.random.normal(0, 0.1, 2)) * m0.flatten()[i] for i in idx_all]
    Dpar[idx_all] = 1.7e-9 + np.random.normal(0, 0.1e-9, idx_all.__len__())
    Diso[idx_all] = 2.5e-9 + np.random.normal(0, 0.1e-9, idx_all.__len__())
    fpar[idx_all] = .3 + np.random.normal(0, 0.025, idx_all.__len__())
    mu[idx_all] = np.tile([np.pi / 2, 0], (idx_all.__len__(), 1)) + np.random.normal(0, 0.1, (idx_all.__len__(), 2))

    # ensure values within ranges
    set_lb = [xx for xx, x in enumerate(Dpar[idx_all] < model.parameter_ranges['C1Stick_1_lambda_par'][0] * 1e-9) if x]
    set_ub = [xx for xx, x in enumerate(Dpar[idx_all] > model.parameter_ranges['C1Stick_1_lambda_par'][1] * 1e-9) if x]
    Dpar[set_lb] = model.parameter_ranges['C1Stick_1_lambda_par'][0] * 1e-9
    Dpar[set_ub] = model.parameter_ranges['C1Stick_1_lambda_par'][1] * 1e-9

    set_lb = [xx for xx, x in enumerate(Diso[idx_all] < model.parameter_ranges['G1Ball_1_lambda_iso'][0] * 1e-9) if x]
    set_ub = [xx for xx, x in enumerate(Diso[idx_all] > model.parameter_ranges['G1Ball_1_lambda_iso'][1] * 1e-9) if x]
    Diso[set_lb] = model.parameter_ranges['G1Ball_1_lambda_iso'][0] * 1e-9
    Diso[set_ub] = model.parameter_ranges['G1Ball_1_lambda_iso'][1] * 1e-9

    set_lb = [xx for xx, x in enumerate(fpar[idx_all] < model.parameter_ranges['partial_volume_0'][0]) if x]
    set_ub = [xx for xx, x in enumerate(fpar[idx_all] > model.parameter_ranges['partial_volume_0'][1]) if x]
    fpar[set_lb] = model.parameter_ranges['partial_volume_0'][0]
    fpar[set_ub] = model.parameter_ranges['partial_volume_0'][1]

    # simulate signal with SNR = 100
    parameter_vector = model.parameters_to_parameter_vector(
        C1Stick_1_mu=mu,
        C1Stick_1_lambda_par=Dpar,
        G1Ball_1_lambda_iso=Diso,
        partial_volume_0=fpar,
        partial_volume_1=1 - fpar)
    data = model.simulate_signal(acq_scheme, parameter_vector) \
           + np.random.normal(0, .01, (np.prod(dims), acq_scheme.number_of_measurements))

    # convert parameter vector to dictionary
    parameter_vector = model.parameter_vector_to_parameters(parameter_vector)

    return parameter_vector, mask, data, m0


def load_shepp_phantom(dims, model, acq_scheme, snr):
    # load Shepp-Logan phantom
    m0, t1, t2 = shepp_logan(dims, MR=True, zlims=(-.25, .25))
    for i, j in itertools.product(range(dims[0]), range(dims[1])):
        if m0[i, j] < .5:
            m0[i, j] = 0
        # elif (m0[i, j] >= .5) & (m0[i, j] < .8):
            # m0[i, j] = .7
        elif (m0[i, j] >= .8) & (m0[i, j] < .9):
            m0[i, j] = .8
        elif m0[i, j] >= .9:
            m0[i, j] = 1
    m0 = m0.flatten()

    # create mask with integer values
    nvox = np.prod(dims)
    roi_vals = np.unique(m0)[np.unique(m0) > 0]
    mask = np.zeros(nvox)
    for roi in range(roi_vals.__len__()):
        idx_roi = [xx for xx, x in enumerate(m0 == roi_vals[roi]) if x]
        if idx_roi.__len__() > 8:  # avoid df error in sigma calculation, i.e. must have df = (nvox - nparams - 1) >= 1
            mask[idx_roi] = roi + 1
    idx_all = [xx for xx, x in enumerate(mask > 0) if x]
    # idx_all = np.linspace(0, nvox-1, nvox)

    # Dpar = 1.7e-9 + np.random.normal(0, 0.1e-9, nvox)
    # Diso = 2.5e-9 + np.random.normal(0, 0.1e-9, nvox)
    # fpar = 0.3 + np.random.normal(0, 0.025, nvox)
    # mu = (np.pi/2, 0) + np.random.normal(0, 0.11, (nvox, 2))
    Dpar = np.zeros(np.prod(dims))
    Diso = np.zeros(np.prod(dims))
    fpar = np.zeros(np.prod(dims))
    mu = np.zeros([np.prod(dims), 2])
    Dpar[idx_all] = [(1.7e-9 + np.random.normal(0, 0.1e-9)) * m0.flatten()[i] for i in idx_all]
    Diso[idx_all] = [(2.5e-9 + np.random.normal(0, 0.1e-9)) * m0.flatten()[i] for i in idx_all]
    fpar[idx_all] = [(.3 + np.random.normal(0, 0.025)) * m0.flatten()[i] for i in idx_all]
    mu[idx_all] = [([np.pi / 2, 0] + np.random.normal(0, 0.1, 2)) * m0.flatten()[i] for i in idx_all]

    # ensure values within ranges
    set_lb = [xx for xx, x in enumerate(Dpar[idx_all] < model.parameter_ranges['C1Stick_1_lambda_par'][0] * 1e-9) if x]
    set_ub = [xx for xx, x in enumerate(Dpar[idx_all] > model.parameter_ranges['C1Stick_1_lambda_par'][1] * 1e-9) if x]
    Dpar[set_lb] = model.parameter_ranges['C1Stick_1_lambda_par'][0] * 1e-9
    Dpar[set_ub] = model.parameter_ranges['C1Stick_1_lambda_par'][1] * 1e-9

    set_lb = [xx for xx, x in enumerate(Diso[idx_all] < model.parameter_ranges['G1Ball_1_lambda_iso'][0] * 1e-9) if x]
    set_ub = [xx for xx, x in enumerate(Diso[idx_all] > model.parameter_ranges['G1Ball_1_lambda_iso'][1] * 1e-9) if x]
    Diso[set_lb] = model.parameter_ranges['G1Ball_1_lambda_iso'][0] * 1e-9
    Diso[set_ub] = model.parameter_ranges['G1Ball_1_lambda_iso'][1] * 1e-9

    set_lb = [xx for xx, x in enumerate(fpar[idx_all] < model.parameter_ranges['partial_volume_0'][0]) if x]
    set_ub = [xx for xx, x in enumerate(fpar[idx_all] > model.parameter_ranges['partial_volume_0'][1]) if x]
    fpar[set_lb] = model.parameter_ranges['partial_volume_0'][0]
    fpar[set_ub] = model.parameter_ranges['partial_volume_0'][1]

    set_lb = [xx for xx, x in enumerate(mu[idx_all, 0] < model.parameter_ranges['C1Stick_1_mu'][0][0]) if x]
    set_ub = [xx for xx, x in enumerate(mu[idx_all, 0] > model.parameter_ranges['C1Stick_1_mu'][0][1]) if x]
    mu[set_lb, 0] = model.parameter_ranges['C1Stick_1_mu'][0][0]
    mu[set_ub, 0] = model.parameter_ranges['C1Stick_1_mu'][0][1]
    set_lb = [xx for xx, x in enumerate(mu[idx_all, 1] < model.parameter_ranges['C1Stick_1_mu'][1][0]) if x]
    set_ub = [xx for xx, x in enumerate(mu[idx_all, 1] > model.parameter_ranges['C1Stick_1_mu'][1][1]) if x]
    mu[set_lb, 0] = model.parameter_ranges['C1Stick_1_mu'][1][0]
    mu[set_ub, 0] = model.parameter_ranges['C1Stick_1_mu'][1][1]

    parameter_vector = model.parameters_to_parameter_vector(
        C1Stick_1_mu=mu,
        C1Stick_1_lambda_par=Dpar,
        G1Ball_1_lambda_iso=Diso,
        partial_volume_0=fpar,
        partial_volume_1=1 - fpar)
    data = model.simulate_signal(acq_scheme, parameter_vector) + np.random.normal(0, 1/snr, (nvox, acq_scheme.number_of_measurements))

    # convert parameter vector to dictionary
    parameter_vector = model.parameter_vector_to_parameters(parameter_vector)

    return parameter_vector, mask, data, m0


def load_dynamic_phantom(dim, model, acq_scheme):
    # load concentric circle phantom
    m0 = dynamic(dim, 1)

    # use the m0 image to create ballstick model params
    Dpar = np.zeros(dim * dim)
    Diso = np.zeros(dim * dim)
    fpar = np.zeros(dim * dim)
    mu = np.zeros([dim * dim, 2])
    for i in range(np.unique(m0.flatten()).__len__()):
        idx = m0.flatten() == np.unique(m0.flatten())[i]
        Dpar[idx] = 1.7e-9 * np.unique(m0.flatten())[i]  # in m^2/s
        Diso[idx] = 2.5e-9 * np.unique(m0.flatten())[i]  # in m^2/s
        fpar[idx] = 0.3 * np.unique(m0.flatten())[i]
        mu[idx] = [np.pi * np.unique(m0.flatten())[i], 0]

    # create mask
    mask = m0 > 0
    mask = mask.flatten()

    # simulate signal with SNR = 100
    parameter_vector = model.parameters_to_parameter_vector(
        C1Stick_1_mu=mu,
        C1Stick_1_lambda_par=Dpar,
        G1Ball_1_lambda_iso=Diso,
        partial_volume_0=fpar,
        partial_volume_1=1 - fpar)
    data = model.simulate_signal(acq_scheme, parameter_vector) + np.random.normal(0, .01)

    # convert parameter vector to dictionary
    parameter_vector = model.parameter_vector_to_parameters(parameter_vector)

    return parameter_vector, mask, data, m0


def load_toy_phantom(dimx, dimy, model, acq_scheme):
    # simulate a simple 10x10 image
    nvox = dimx * dimy

    Dpar_sim = 1.7e-9 + np.random.normal(0, 0.1e-9, nvox)
    Diso_sim = 2.5e-9 + np.random.normal(0, 0.1e-9, nvox)
    fpar_sim = 0.3 + np.random.normal(0, 0.025, nvox)
    stick_ori_sim = (np.pi, 0) + np.random.normal(0, 0.11, (nvox, 2))
    parameter_vector = model.parameters_to_parameter_vector(
        C1Stick_1_mu=stick_ori_sim,
        C1Stick_1_lambda_par=Dpar_sim,
        G1Ball_1_lambda_iso=Diso_sim,
        partial_volume_0=fpar_sim,
        partial_volume_1=1 - fpar_sim)
    data = model.simulate_signal(acq_scheme, parameter_vector) + np.random.normal(0, .01, (nvox, acq_scheme.number_of_measurements))

    parameter_vector = model.parameter_vector_to_parameters(parameter_vector)

    # create mask with regional ROIs
    mask = np.ones(nvox)

    return parameter_vector, mask, data


def main():
    # set up acquisition parameters and ballstick model
    acq_scheme, nmeas, stick, ball, ballstick = sign_par()
    model = deepcopy(ballstick)

    # simulate Shepp-Logan data
    snr = 25
    dims = [32, 32, 1]
    # dims = [64, 64, 1]
    # dims = [10, 10, 1]
    parameter_vector_correct, mask, data, m0 = load_shepp_phantom(dims, model, acq_scheme, snr)
    # parameter_vector_correct, mask, data = load_toy_phantom(dims[0], dims[1], model, acq_scheme)
    # parameter_vector_correct, mask, data, m0 = load_dynamic_phantom(dims[0], model, acq_scheme)
    mask[mask > 0] = 1
    # mask = np.ones(np.prod(dims))

    nsteps = 6000  # 5000
    burn_in = 3000  # 1500

    # LSQ fitting
    lsq_fit = model.fit(acq_scheme, data, mask=mask)
    parameter_vector_lsq = lsq_fit.fitted_parameters

    # hierarchical Bayesian fitting
    proc_start = time.time()
    acceptance_rate, param_conv, parameter_vector_bayes, parameter_vector_init, likelihood_stored, w_stored = fit_bayes.fit(model, acq_scheme, data, mask, nsteps, burn_in)
    compute_time(proc_start, time.time())

    # print: initialisation, correct value, mean (after burn-in) Bayes-fitted value
    nparams = np.sum(np.array(list(model.parameter_cardinality.values())))
    roi_vals = np.unique(mask)[np.unique(mask) > 0]  # list of unique integers that identify each ROI (ignore 0's)
    roi_nvox = [[xx for xx, x in enumerate(mask == roi_vals[roi]) if x].__len__() for roi in
                range(roi_vals.__len__())]  # number of voxels in each ROI
    to_remove = [roi for roi in range(roi_vals.__len__()) if
                 roi_nvox[roi] < 2 * nparams]  # indices of ROIs with too few voxels
    roi_vals = np.delete(roi_vals, to_remove)
    idx_roi = [xx for xx, x in enumerate(mask == roi_vals[0]) if x]
    vox = idx_roi[0]

    # ------------------------------------------------------------------------------------------------------------------
    # filename = '/home/epowell/code/python/dmipy-bayesian/data/shepp_logan_' + str(dims[0]) + 'x' + str(dims[1]) \
    #            + '_snr25_nsteps' + str(nsteps) + '_burn' + str(burn_in) + '.db'
    # save_workspace(filename)

    # ------------------------------------------------------------------------------------------------------------------
    print((parameter_vector_init['C1Stick_1_lambda_par'][vox],
           parameter_vector_correct['C1Stick_1_lambda_par'][vox],
           parameter_vector_lsq['C1Stick_1_lambda_par'][vox],
           np.mean(param_conv['C1Stick_1_lambda_par'][vox, burn_in:-1])))
    print((parameter_vector_init['G1Ball_1_lambda_iso'][vox],
           parameter_vector_correct['G1Ball_1_lambda_iso'][vox],
           parameter_vector_lsq['G1Ball_1_lambda_iso'][vox],
           np.mean(param_conv['G1Ball_1_lambda_iso'][vox, burn_in:-1])))
    print((parameter_vector_init['partial_volume_0'][vox],
           parameter_vector_correct['partial_volume_0'][vox],
           parameter_vector_lsq['partial_volume_0'][vox],
           np.mean(param_conv['partial_volume_0'][vox, burn_in:-1])))
    print((parameter_vector_init['C1Stick_1_mu'][vox, 0],
           parameter_vector_correct['C1Stick_1_mu'][vox, 0],
           parameter_vector_lsq['C1Stick_1_mu'][vox, 0],
           np.mean(param_conv['C1Stick_1_mu'][vox, 0, burn_in:-1])))
    print((parameter_vector_init['C1Stick_1_mu'][vox, 1],
           parameter_vector_correct['C1Stick_1_mu'][vox, 1],
           parameter_vector_lsq['C1Stick_1_mu'][vox, 1],
           np.mean(param_conv['C1Stick_1_mu'][vox, 1, burn_in:-1])))

    # ------------------------------------------------------------------------------------------------------------------
    # plot parameter convergence
    plt.rcParams.update({'font.size': 22})
    lw = 5
    fig, axs = plt.subplots(2, 4)

    axs[0, 0].plot(range(nsteps), param_conv['C1Stick_1_lambda_par'][vox, :], color='tab:red', linewidth=lw)
    axs[0, 0].set_ylabel("Dpar")
    axs[0, 0].set_xlabel("MCMC iteration")
    make_square_axes(axs[0, 0])

    axs[0, 1].plot(range(nsteps), param_conv['G1Ball_1_lambda_iso'][vox, :], color='tab:green', linewidth=lw)
    axs[0, 1].set_ylabel("Diso")
    axs[0, 1].set_xlabel("MCMC iteration")
    make_square_axes(axs[0, 1])

    axs[0, 2].plot(range(nsteps), param_conv['partial_volume_0'][vox, :], color='tab:blue', linewidth=lw)
    axs[0, 2].set_ylabel("fpar")
    axs[0, 2].set_xlabel("MCMC iteration")
    make_square_axes(axs[0, 2])

    axs[0, 3].plot(range(nsteps), param_conv['C1Stick_1_mu'][vox, 0, :], color='tab:purple', linewidth=lw)
    axs[0, 3].set_ylabel("stick orientation")
    axs[0, 3].set_xlabel("MCMC iteration")
    make_square_axes(axs[0, 3])

    # plot parameter distributions after burn-in period
    nbins = 10
    vals = param_conv['C1Stick_1_lambda_par'][vox, burn_in:-1]*1e9  # multiply by 1e9 so gaussian has same scaling
    axs[1, 0].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=False)
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 0].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='tab:red', linewidth=lw)
    axs[1, 0].set_ylabel("frequency")
    axs[1, 0].set_xlabel("Dpar")
    make_square_axes(axs[1, 0])

    vals = param_conv['G1Ball_1_lambda_iso'][vox, burn_in:-1]*1e9   # multiply by 1e9 so gaussian has same scaling
    axs[1, 1].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=False)
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 1].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='tab:green', linewidth=lw)
    axs[1, 1].set_ylabel("frequency")
    axs[1, 1].set_xlabel("Diso")
    make_square_axes(axs[1, 1])

    vals = param_conv['partial_volume_0'][vox, burn_in:-1]
    axs[1, 2].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=False)
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 2].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='tab:blue', linewidth=lw)
    axs[1, 2].set_ylabel("frequency")
    axs[1, 2].set_xlabel("fpar")
    make_square_axes(axs[1, 2])

    vals = param_conv['C1Stick_1_mu'][vox, 0, burn_in:-1]
    axs[1, 3].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=False)
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 3].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='tab:purple', linewidth=lw)
    axs[1, 3].set_ylabel("frequency")
    axs[1, 3].set_xlabel("stick orientation")
    make_square_axes(axs[1, 3])

    # ------------------------------------------------------------------------------------------------------------------
    # plot acceptance rate
    fig, axs = plt.subplots(1, 2)
    axs[0].set_ylabel("Acceptance Rate")
    axs[0].plot(range(nsteps), acceptance_rate['C1Stick_1_lambda_par'][vox], color='tab:red', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['G1Ball_1_lambda_iso'][vox, :], color='tab:green', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['partial_volume_0'][vox, :], color='tab:blue', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['C1Stick_1_mu'][vox, 0, :], color='tab:olive', linewidth=lw)
    axs[0].legend(['Dpar', 'Diso', 'fpar', 'mu'])

    # plot likelihood
    axs[1].set_ylabel("Likelihood")
    axs[1].plot(range(nsteps), likelihood_stored['C1Stick_1_lambda_par'][vox, :], color='tab:red', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['G1Ball_1_lambda_iso'][vox, :], color='tab:green', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['partial_volume_0'][vox, :], color='tab:blue', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['C1Stick_1_mu'][vox, 0, :], color='tab:olive', linewidth=lw)
    axs[1].legend(['Dpar', 'Diso', 'fpar', 'mu'])

    # ------------------------------------------------------------------------------------------------------------------
    # plot maps: GT, LSQ, Bayes
    fig = plt.figure(figsize=(3, 3))
    grid = AxesGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.7, cbar_mode='each')
    cmap_D = mpl.cm.inferno
    # viridis = mpl.cm.get_cmap('viridis')
    viridis = mpl.cm.viridis
    viridis.colors[0] = [0, 0, 0]
    viridis.colors[1] = [0, 0, 0]
    cmap_f = viridis
    clims_D = [.1e-9, 3e-9]
    clims_f = [0, 1]

    # GT
    grid[0].imshow(np.reshape(parameter_vector_correct['C1Stick_1_lambda_par'], dims), vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)
    grid[0].set_title('GT, Dpar')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D), cax=grid[0].cax, orientation='vertical', label='Some Units')

    grid[1].imshow(np.reshape(parameter_vector_correct['G1Ball_1_lambda_iso'], dims), vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)
    grid[1].set_title('GT, Diso')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D), cax=grid[1].cax, orientation='vertical', label='Some Units')

    grid[2].imshow(np.reshape(parameter_vector_correct['partial_volume_0'], dims), vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)
    grid[2].set_title('GT, fpar')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_f[0], vmax=clims_f[1]), cmap=cmap_f), cax=grid[2].cax, orientation='vertical', label='Some Units')

    # LSQ
    grid[3].imshow(np.reshape(parameter_vector_lsq['C1Stick_1_lambda_par'], dims), vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)
    grid[3].set_title('GT, Dpar')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D), cax=grid[3].cax, orientation='vertical', label='Some Units')

    grid[4].imshow(np.reshape(parameter_vector_lsq['G1Ball_1_lambda_iso'], dims), vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)
    grid[4].set_title('GT, Diso')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D), cax=grid[4].cax, orientation='vertical', label='Some Units')

    grid[5].imshow(np.reshape(parameter_vector_lsq['partial_volume_0'], dims), vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)
    grid[5].set_title('GT, fpar')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_f[0], vmax=clims_f[1]), cmap=cmap_f), cax=grid[5].cax, orientation='vertical', label='Some Units')

    # Bayes
    grid[6].imshow(np.reshape(parameter_vector_bayes['C1Stick_1_lambda_par'], dims), vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)
    grid[6].set_title('GT, Dpar')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D), cax=grid[6].cax, orientation='vertical', label='Some Units')

    grid[7].imshow(np.reshape(parameter_vector_bayes['G1Ball_1_lambda_iso'], dims), vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)
    grid[7].set_title('GT, Diso')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D), cax=grid[7].cax, orientation='vertical', label='Some Units')

    grid[8].imshow(np.reshape(parameter_vector_bayes['partial_volume_0'], dims), vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)
    grid[8].set_title('GT, fpar')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_f[0], vmax=clims_f[1]), cmap=cmap_f), cax=grid[8].cax, orientation='vertical', label='Some Units')

    # ------------------------------------------------------------------------------------------------------------------
    # plot error maps: (GT-fit)/GT; fit = LSQ or Bayes
    fig = plt.figure(figsize=(2, 3))
    grid = AxesGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0.5, cbar_mode='each')
    cmap_err = mpl.cm.seismic
    clims_err = [-5, 5]

    # LSQ
    err = 100 * (np.reshape(parameter_vector_lsq['C1Stick_1_lambda_par'], dims)
                 - np.reshape(parameter_vector_correct['C1Stick_1_lambda_par'], dims)) \
          / np.reshape(parameter_vector_correct['C1Stick_1_lambda_par'], dims)
    grid[0].imshow(err, vmin=clims_err[0], vmax=clims_err[1], cmap=cmap_err)
    grid[0].set_title('LSQ, Dpar, '
                      + 'err = ' + str(round(np.mean(err[np.reshape(mask, dims) > 0]), 2))
                      + ', abs err = ' + str(round(np.mean(np.abs(err[np.reshape(mask, dims) > 0])), 2)))
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_err[0], vmax=clims_err[1]), cmap=cmap_err), cax=grid[0].cax, orientation='vertical', label='Some Units')

    err = 100 * (np.reshape(parameter_vector_lsq['G1Ball_1_lambda_iso'], dims)
                 - np.reshape(parameter_vector_correct['G1Ball_1_lambda_iso'], dims)) \
          / np.reshape(parameter_vector_correct['G1Ball_1_lambda_iso'], dims)
    grid[1].imshow(err, vmin=clims_err[0], vmax=clims_err[1], cmap=cmap_err)
    grid[1].set_title('LSQ, Diso, '
                      + 'err = ' + str(round(np.mean(err[np.reshape(mask, dims) > 0]), 2))
                      + ', err = ' + str(round(np.mean(np.abs(err[np.reshape(mask, dims) > 0])), 2)))
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_err[0], vmax=clims_err[1]), cmap=cmap_err), cax=grid[1].cax, orientation='vertical', label='Some Units')

    err = 100 * (np.reshape(parameter_vector_lsq['partial_volume_0'], dims)
                 - np.reshape(parameter_vector_correct['partial_volume_0'], dims)) \
          / np.reshape(parameter_vector_correct['partial_volume_0'], dims)
    grid[2].imshow(err, vmin=clims_err[0], vmax=clims_err[1], cmap=cmap_err)
    grid[2].set_title('LSQ, fpar, '
                      + 'err = ' + str(round(np.mean(err[np.reshape(mask, dims) > 0]), 2))
                      + ', abs err = ' + str(round(np.mean(np.abs(err[np.reshape(mask, dims) > 0])), 2)))
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_err[0], vmax=clims_err[1]), cmap=cmap_err), cax=grid[2].cax, orientation='vertical', label='Some Units')

    # Bayes
    err = 100 * (np.reshape(parameter_vector_bayes['C1Stick_1_lambda_par'], dims)
                 - np.reshape(parameter_vector_correct['C1Stick_1_lambda_par'], dims)) \
          / np.reshape(parameter_vector_correct['C1Stick_1_lambda_par'], dims)
    grid[3].imshow(err, vmin=clims_err[0], vmax=clims_err[1], cmap=cmap_err)
    grid[3].set_title('Bayes, Dpar, '
                      + 'err = ' + str(round(np.mean(err[np.reshape(mask, dims) > 0]), 2))
                      + ', abs err = ' + str(round(np.mean(np.abs(err[np.reshape(mask, dims) > 0])), 2)))
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_err[0], vmax=clims_err[1]), cmap=cmap_err), cax=grid[3].cax, orientation='vertical', label='Some Units')

    err = 100 * (np.reshape(parameter_vector_bayes['G1Ball_1_lambda_iso'], dims)
                 - np.reshape(parameter_vector_correct['G1Ball_1_lambda_iso'], dims)) \
          / np.reshape(parameter_vector_correct['G1Ball_1_lambda_iso'], dims)
    grid[4].imshow(err, vmin=clims_err[0], vmax=clims_err[1], cmap=cmap_err)
    grid[4].set_title('Bayes, Diso, '
                      + 'err = ' + str(round(np.mean(err[np.reshape(mask, dims) > 0]), 2))
                      + ', abs err = ' + str(round(np.mean(np.abs(err[np.reshape(mask, dims) > 0])), 2)))
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_err[0], vmax=clims_err[1]), cmap=cmap_err), cax=grid[4].cax, orientation='vertical', label='Some Units')

    err = 100 * (np.reshape(parameter_vector_bayes['partial_volume_0'], dims)
                 - np.reshape(parameter_vector_correct['partial_volume_0'], dims)) \
          / np.reshape(parameter_vector_correct['partial_volume_0'], dims)
    grid[5].imshow(err, vmin=clims_err[0], vmax=clims_err[1], cmap=cmap_err)
    grid[5].set_title('Bayes, fpar, '
                      + 'err = ' + str(round(np.mean(err[np.reshape(mask, dims) > 0]), 2))
                      + ', abs err = ' + str(round(np.mean(np.abs(err[np.reshape(mask, dims) > 0])), 2)))
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_err[0], vmax=clims_err[1]), cmap=cmap_err), cax=grid[5].cax, orientation='vertical', label='Some Units')

    '''
    plt.rcParams.update({'font.size': 24})
    fig = plt.figure(figsize=(2, 3))
    grid = AxesGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=[1.5, 1.5], cbar_mode='none', label_mode='all', share_all=False, aspect=True)

    # plot parameter convergence
    grid[0].plot(range(nsteps), param_conv['C1Stick_1_lambda_par'][vox, :], color='tab:red')
    grid[0].set_ylabel("Dpar")
    grid[0].set_xlabel("MCMC iteration")
    grid[0].axes.set_yticks(np.linspace(1e-9, 3e-9, 5))
    grid[0].axes.set_ylim([1e-9, 3e-9])

    grid[1].plot(range(nsteps), param_conv['G1Ball_1_lambda_iso'][vox, :], color='tab:green')
    grid[1].set_ylabel("Diso")
    grid[1].set_xlabel("MCMC iteration")
    grid[1].axes.set_yticks(np.linspace(1e-9, 3e-9, 5))
    grid[1].axes.set_ylim([1e-9, 3e-9])

    grid[2].plot(range(nsteps), param_conv['partial_volume_0'][vox, :], color='tab:blue')
    grid[2].set_ylabel("fpar")
    grid[2].set_xlabel("MCMC iteration")
    grid[2].axes.set_yticks(np.linspace(0, 1, 5))
    grid[2].axes.set_ylim([0, 1])

    for i in range(3):
        grid[i].set_xticks(np.linspace(0, nsteps, 6))
        make_square_axes(grid[i])

    # plot parameter distributions after burn-in period
    nbins = 50
    grid[3].hist(param_conv['C1Stick_1_lambda_par'][vox, burn_in:-1], bins=nbins, color='tab:red')
    grid[3].set_ylabel("freq")
    grid[3].set_xlabel("Dpar")
    grid[3].set_xticks(np.linspace(np.min(param_conv['C1Stick_1_lambda_par'][vox, burn_in:-1]),
                                   np.max(param_conv['C1Stick_1_lambda_par'][vox, burn_in:-1]), 6))

    grid[4].hist(param_conv['G1Ball_1_lambda_iso'][vox, burn_in:-1], bins=nbins, color='tab:green')
    grid[4].set_ylabel("freq")
    grid[4].set_xlabel("Diso")
    grid[4].set_xticks(np.linspace(np.min(param_conv['G1Ball_1_lambda_iso'][vox, burn_in:-1]),
                                   np.max(param_conv['G1Ball_1_lambda_iso'][vox, burn_in:-1]), 6))

    grid[5].hist(param_conv['partial_volume_0'][vox, burn_in:-1], bins=nbins, color='tab:blue')
    grid[5].set_ylabel("freq")
    grid[5].set_xlabel("fpar")
    grid[5].set_xticks(np.linspace(np.min(param_conv['partial_volume_0'][vox, burn_in:-1]),
                                   np.max(param_conv['partial_volume_0'][vox, burn_in:-1]), 6))

    for i in range(3, 6):
        make_square_axes(grid[i])
        
    
    plt.rcParams.update({'font.size': 24})
    fig = plt.figure(figsize=(1, 2))
    grid = AxesGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=[1.5, 1.5], cbar_mode='none', label_mode='all')

    # plot acceptance rate
    grid[0].scatter(range(nsteps), acceptance_rate['C1Stick_1_lambda_par'][vox], color='tab:red')
    grid[0].scatter(range(nsteps), acceptance_rate['G1Ball_1_lambda_iso'][vox], color='tab:green')
    grid[0].scatter(range(nsteps), acceptance_rate['partial_volume_0'][vox], color='tab:blue')
    grid[0].set_ylabel("Acceptance Rate")
    grid[0].legend(['Dpar', 'Diso', 'fpar'])

    # plot likelihood
    grid[1].scatter(range(nsteps), likelihood_stored['C1Stick_1_lambda_par'][vox], color='tab:red')
    grid[1].scatter(range(nsteps), likelihood_stored['G1Ball_1_lambda_iso'][vox], color='tab:green')
    grid[1].scatter(range(nsteps), likelihood_stored['partial_volume_0'][vox], color='tab:blue')
    grid[1].set_ylabel("Likelihood")
    grid[1].legend(['Dpar', 'Diso', 'fpar'])

    for i in range(2):
        make_square_axes(grid[i])
    
    '''

    '''
    fig, axs = plt.subplots(3, 3)
    # GT
    axs[0, 0].imshow(np.reshape(parameter_vector_correct['C1Stick_1_lambda_par'], dims), vmin=.1e-9, vmax=3e-9)
    axs[0, 0].title.set_text('GT, Dpar')
    axs[0, 1].imshow(np.reshape(parameter_vector_correct['G1Ball_1_lambda_iso'], dims), vmin=.1e-9, vmax=3e-9)
    axs[0, 1].title.set_text('GT, Diso')
    axs[0, 2].imshow(np.reshape(parameter_vector_correct['partial_volume_0'], dims), vmin=0, vmax=1)
    axs[0, 2].title.set_text('GT, fpar')
    # LSQ
    axs[1, 0].imshow(np.reshape(parameter_vector_lsq['C1Stick_1_lambda_par'], dims), vmin=.1e-9, vmax=3e-9)
    axs[1, 0].title.set_text('GT, Dpar')
    axs[1, 1].imshow(np.reshape(parameter_vector_lsq['G1Ball_1_lambda_iso'], dims), vmin=.1e-9, vmax=3e-9)
    axs[1, 1].title.set_text('GT, Diso')
    axs[1, 2].imshow(np.reshape(parameter_vector_lsq['partial_volume_0'], dims), vmin=0, vmax=1)
    axs[1, 2].title.set_text('GT, fpar')
    # Bayes
    axs[2, 0].imshow(np.reshape(parameter_vector_bayes['C1Stick_1_lambda_par'], dims), vmin=.1e-9, vmax=3e-9)
    axs[2, 0].title.set_text('GT, Dpar')
    axs[2, 1].imshow(np.reshape(parameter_vector_bayes['G1Ball_1_lambda_iso'], dims), vmin=.1e-9, vmax=3e-9)
    axs[2, 1].title.set_text('GT, Diso')
    axs[2, 2].imshow(np.reshape(parameter_vector_bayes['partial_volume_0'], dims), vmin=0, vmax=1)
    axs[2, 2].title.set_text('GT, fpar')
    '''

    '''
    theta, phi = parameter_vector_correct['C1Stick_1_mu'].T
    mu_cart = np.zeros((theta.__len__(), 3))
    sintheta = np.sin(theta)
    mu_cart[:, 0] = sintheta * np.cos(phi)
    mu_cart[:, 1] = sintheta * np.sin(phi)
    mu_cart[:, 2] = np.cos(theta)
    fig, ax = plt.quiver(np.reshape(mu_cart[:,0], dims[0:2]), np.reshape(mu_cart[:,1], dims[0:2]))
    '''


if __name__ == '__main__':
    main()
