# load some necessary modules
# from dmipy.core import modeling_framework
# from os.path import join
# import scipy.stats
# from os.path import join as pjoin
import numpy as np
import pandas as pd
# from os import listdir
# from os.path import join
import nibabel as nib
from copy import deepcopy
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import transforms
# from mpl_toolkits.axes_grid1 import AxesGrid
import time
import math
import shelve
import argparse
from contextlib import contextmanager
import sys
import os

# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes

# ball stick and spherical mean ball-stick model
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core import modeling_framework
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel, MultiCompartmentModel
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.distributions.distribute_models import SD1WatsonDistributed, BundleModel
from dmipy.utils import spherical_mean

from useful_functions import suppress_stdout, compute_time
from setup_models import _setup_model as setup_model
import fit_bayes
# from importlib import reload
# fit_bayes = reload(fit_bayes)
# from fit_bayes import fit, tform_params  # , dict_to_array, array_to_dict


def load_real_data(path_dmri, path_bval, path_bvec, path_mask, zslice=np.nan):
    bvalues = np.loadtxt(path_bval)  # given in s/mm^2
    bvalues_SI = bvalues * 1e6  # now given in SI units as s/m^2
    gradient_directions = np.loadtxt(path_bvec).transpose()  # on the unit sphere

    # The delta and Delta times we know from the HCP documentation in seconds
    delta = 0.0106
    Delta = 0.0431

    # The acquisition scheme used in the toolbox is then created as follows:
    acq_scheme = acquisition_scheme_from_bvalues(bvalues_SI, gradient_directions, delta, Delta)

    # -- Reading the DWI nifti image
    data = nib.load(path_dmri).get_fdata(dtype=np.float16)
    mask = nib.load(path_mask).get_fdata(dtype=np.float16)

    # plotting an axial slice
    # import matplotlib.pyplot as plt
    # axial_middle = data.shape[2] // 2
    # plt.figure('Axial slice')
    # plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
    # plt.show()

    # print('df zslice = ',)
    if not pd.isna(zslice):
        data = data[:, :, zslice, :]
        mask = mask[:, :, zslice]

    return acq_scheme, data, mask


def load_amico_fits(path_amico, zslice=np.nan):
    # load model fits
    hdr = {}
    fibdir = nib.load(path_amico + '/FIT_dir.nii.gz')
    hdr['SD1WatsonDistributed_1_SD1Watson_1_mu'] = fibdir.header
    fibdir = fibdir.get_data()
    od = nib.load(path_amico + '/FIT_OD.nii.gz')
    hdr['SD1WatsonDistributed_1_SD1Watson_1_odi'] = od.header
    od = od.get_data()
    isovf = nib.load(path_amico + '/FIT_ISOVF.nii.gz')
    hdr['partial_volume_0'] = isovf.header
    hdr['partial_volume_1'] = isovf.header
    isovf = isovf.get_data()
    icvf = nib.load(path_amico + '/FIT_ICVF.nii.gz')
    hdr['SD1WatsonDistributed_1_partial_volume_0'] = icvf.header
    icvf = icvf.get_data()
    fibdirpolar = cart2mu(fibdir)

    # select slice if desired
    if not pd.isna(zslice):
        od = od[:, :, zslice]
        isovf = isovf[:, :, zslice]
        icvf = icvf[:, :, zslice]
        fibdir = fibdir[:, :, zslice, :]
        fibdirpolar = fibdirpolar[:, :, zslice, :]

    # put the data in voxel form
    nvox = np.prod(np.shape(od)[0:3])
    mu_vox = np.reshape(fibdirpolar, (nvox, 2))
    od_vox = np.reshape(od, nvox)
    isovf_vox = np.reshape(isovf, nvox)  # ball
    icvf_vox = np.reshape(icvf, nvox)  # stick
    # put in the format that bayes_fit needs
    parameter_vector_init = {}
    parameter_vector_init['SD1WatsonDistributed_1_SD1Watson_1_mu'] = mu_vox
    parameter_vector_init['SD1WatsonDistributed_1_SD1Watson_1_odi'] = od_vox
    parameter_vector_init['SD1WatsonDistributed_1_partial_volume_0'] = icvf_vox
    parameter_vector_init['partial_volume_0'] = isovf_vox
    parameter_vector_init['partial_volume_1'] = 1 - isovf_vox

    return parameter_vector_init, hdr


def main():

    print('We are in!', flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsteps', type=int, nargs='+', help='Number of MCMC steps')
    parser.add_argument('--burn_in', type=int, nargs='+', help='Number of burn in steps for MCMC')
    parser.add_argument('--path_dmri', type=str, nargs='+', help='Full path to dMRI data')
    parser.add_argument('--path_mask', type=str, nargs='+', help='Full path to mask (either global or tissue-specific)')
    parser.add_argument('--path_bval', type=str, nargs='+', help='Full path to b-values')
    parser.add_argument('--path_bvec', type=str, nargs='+', help='Full path to b-vectors')
    parser.add_argument('--path_amico', type=str, nargs='+', help='Full path to AMICO fits')
    parser.add_argument('--zslice', type=int, nargs='+', help='Which slice(s) to fit data to (no argument required for full FoV)')
    args = parser.parse_args()
    print(args, flush=True)
    nsteps = args.nsteps[0]
    burn_in = args.burn_in[0]
    path_dmri = args.path_dmri[0]
    path_mask = args.path_mask[0]
    path_bval = args.path_bval[0]
    path_bvec = args.path_bvec[0]
    path_amico = args.path_amico[0]
    hdr = nib.load(path_mask)
    hdr = hdr.header
    if args.zslice is not None:
        zslice = args.zslice[0]
    else:
        zslice = np.nan

    print(str(zslice))
    '''
    path_dmri = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/data.nii.gz'
    path_mask = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/nodif_brain_mask.nii.gz'
    path_mask = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/tensor-masks/roi-mask.nii.gz'
    path_bval = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/bvals'
    path_bvec = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/bvecs'
    path_amico = '/home/epowell/data/hcp/diffusion_retest/103818_2/NODDI/'
    nsteps = 2000 ## 1000
    burn_in = 1000 ## 200
    zslice=72
    zslice=np.nan
    '''
    model_name = 'smt_noddi'  # ballstick, noddi, smt, smt_noddi
    sph_mn = False

    # set up data paths and model
    print('Loading HCP data and model (' + model_name + ')...', flush=True)
    acq_scheme_orig, data_orig, mask_orig = load_real_data(path_dmri, path_bval, path_bvec, path_mask, zslice)
    acq_scheme = acq_scheme_orig
    model = setup_model(model_name)
    data_orig = np.float32(data_orig)
    print('Done.\n', flush=True)

    # reshape data and mask
    if pd.isna(zslice):
        nx = data_orig.shape[0]
        ny = data_orig.shape[1]
        nz = data_orig.shape[2]
        ndw = data_orig.shape[3]
    else:
        nx = data_orig.shape[0]
        ny = data_orig.shape[1]
        nz = 1
        ndw = data_orig.shape[2]

    print('Reshaping data and mask...', flush=True)
    data_orig = np.reshape(data_orig, (nx*ny*nz, ndw))
    data_orig = data_orig + np.finfo(float).eps  # hack to deal with any 0's in data (e.g. if mask voxel is outside brain)
    mask_orig = np.reshape(mask_orig, (nx*ny*nz))
    print('Done.\n', flush=True)

    print('Reduce data size by only keeping voxels in mask...', flush=True)
    nonzerovox = np.nonzero(mask_orig)[0]
    data = data_orig[nonzerovox, :]
    mask = mask_orig[nonzerovox]

    # load AMICO fits for initialisation (NODDI only)
    if model_name == 'noddi':
        print('Loading AMICO fits...', flush=True)
        parameter_vector_init_orig, hdr = load_amico_fits(path_amico, zslice)
        print('Reshaping...\n', flush=True)
        parameter_vector_init = {}
        for param in parameter_vector_init_orig.keys():  # param=model.parameter_names[0]
            # reshape to vector if necessaryprint('Simulating signals...', flush=True)  # ecap
    with suppress_stdout():  # suppress annoying output in console
        E_fit = model.simulate_signal(acq_scheme, model.parameters_to_parameter_vector(**parameter_vector_init))
        E_fit = np.squeeze(E_fit)
    print('Done.\n', flush=True)
            if parameter_vector_init_orig[param].shape != (nx*ny*nz, 1):
                tmp = np.reshape(parameter_vector_init_orig[param], (nx*ny*nz, model.parameter_cardinality[param]))
            else:
                tmp = parameter_vector_init_orig[param]
            # keep only voxels in mask
            if model.parameter_cardinality[param] > 1:
                parameter_vector_init[param] = tmp[np.nonzero(mask_orig)[0], :]
            elif model.parameter_cardinality[param] == 1:
                parameter_vector_init[param] = tmp[np.nonzero(mask_orig)[0]]
        print('Done.\n', flush=True)
    else:
        parameter_vector_init_orig = {}

    # LSQ fitting (anything but NODDI)
    if model_name != 'noddi':
        print('Running initial LSQ fit...', flush=True)
        proc_start = time.time()
        lsq_fit = model.fit(acq_scheme, data, mask=mask > 0)
        compute_time(proc_start, time.time())
        parameter_vector_init = lsq_fit.fitted_parameters
        # reshape and create parameter_vector_init_orig
        for param in parameter_vector_init.keys():
            print(param)
            if model.parameter_cardinality[param] > 1:
                parameter_vector_init_orig[param] = np.zeros([nx * ny * nz * 2])
                parameter_vector_init_orig[param][nonzerovox] = parameter_vector_init[param]
                parameter_vector_init_orig[param] = np.reshape(parameter_vector_init_orig[param], [nx, ny, nz, 2])
            elif model.parameter_cardinality[param] == 1:
                parameter_vector_init_orig[param] = np.zeros([nx * ny * nz])
                parameter_vector_init_orig[param][nonzerovox] = parameter_vector_init[param]
                parameter_vector_init_orig[param] = np.reshape(parameter_vector_init_orig[param], [nx, ny, nz])
        print('Done.\n', flush=True)
        '''
        fig, axs = plt.subplots(1, len(parameter_vector_init))
        for p in range(len(parameter_vector_init)):
            param = model.parameter_names[p]
            tmp = np.zeros((nx * ny))
            tmp[nonzerovox] = parameter_vector_init[param]
            tmp = np.reshape(tmp, (nx, ny))
            axs[p].imshow(tmp)
        '''

    # simulate signals using initial parameter estimates
    print('Simulating signals...', flush=True)  # ecap
    with suppress_stdout():  # suppress annoying output in console
        E_fit = model.simulate_signal(acq_scheme, model.parameters_to_parameter_vector(**parameter_vector_init))
        E_fit = np.squeeze(E_fit)
    print('Done.\n', flush=True)

    # compute spherical mean of hcp data if model allows
    if E_fit.shape[1] == acq_scheme_orig.spherical_mean_scheme.bvalues.__len__():
        proc_start = time.time()
        print("Computing spherical mean of HCP data...")
        data = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(data[i, :], acq_scheme, sh_order=6)
                           for i in range(nonzerovox.__len__())])
        print("Normalising to b=0 shell...")
        data = np.asarray([data[i, :] / np.max(data[i, :]) for i in range(nonzerovox.__len__())])
        acq_scheme = acq_scheme_orig.spherical_mean_scheme
        print('Done.\n', flush=True)
        compute_time(proc_start, time.time())

    print('data shape = ' + str(data.shape) + '; mask shape = ' + str(mask.shape), flush=True)
    print('parameter initial vector shape = ' + str(parameter_vector_init[model.parameter_names[0]].shape), flush=True)
    print('simulated signal shape = ' + str(E_fit.shape), flush=True)
    print('no. b-values in acq_scheme = ' + str(acq_scheme.bvalues.__len__()))

    # check fits don't hit the bounds; add/subtract eps to any that do
    print('Check initial model estimates do not hit fitting bounds and that signal estimate is not NaN...', flush=True)
    for param in parameter_vector_init_orig.keys():  # param=model.parameter_names[0]
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                idx = parameter_vector_init[param][:, card] <= model.parameter_ranges[param][card][0] * model.parameter_scales[param]
                parameter_vector_init[param][idx, card] = (model.parameter_ranges[param][card][0] + np.finfo(float).eps) * model.parameter_scales[param][card]
                idx = parameter_vector_init[param][:, card] >= model.parameter_ranges[param][card][1] * model.parameter_scales[param]
                parameter_vector_init[param][idx, card] = (model.parameter_ranges[param][card][1] - np.finfo(float).eps) * model.parameter_scales[param][card]
        elif model.parameter_cardinality[param] == 1:
            idx = parameter_vector_init[param] <= model.parameter_ranges[param][0] * model.parameter_scales[param]
            parameter_vector_init[param][idx] = (model.parameter_ranges[param][0] + np.finfo(float).eps) * model.parameter_scales[param]
            idx = parameter_vector_init[param] >= model.parameter_ranges[param][1] * model.parameter_scales[param]
            parameter_vector_init[param][idx] = (model.parameter_ranges[param][1] - np.finfo(float).eps) * model.parameter_scales[param]

    # remove any voxels where estimated signal is nan. FIXME: why are they nan?!
    rmvox1 = np.where(np.asarray([np.isnan(E_fit[i, :]).sum() for i in range(mask.__len__())]) > 0)[0]
    rmvox2 = np.where(np.asarray([np.isinf(E_fit[i, :]).sum() for i in range(mask.__len__())]) > 0)[0]
    rmvox3 = np.where(np.asarray([(data[i, :] == 0).sum() for i in range(mask.__len__())]) > 0)[0]
    rmvox4 = np.asarray([i for i in range(mask.__len__()) if np.any(np.isnan(data[i, :]))], dtype=int)
    rmvox5 = np.asarray([i for i in range(mask.__len__()) if np.any(np.isinf(data[i, :]))], dtype=int)
    kpvox = [i for i in range(mask.__len__())]
    rmvox = np.concatenate((rmvox1, rmvox2, rmvox3, rmvox4, rmvox5)).tolist()
    kpvox = list(set(kpvox) - set(rmvox))
    tmp = deepcopy(parameter_vector_init)
    parameter_vector_init = dict.fromkeys(parameter_vector_init.keys(), [])
    for param in tmp.keys():
        if model.parameter_cardinality[param] > 1:
            parameter_vector_init[param] = tmp[param][kpvox, :]
        elif model.parameter_cardinality[param] == 1:
            parameter_vector_init[param] = tmp[param][kpvox]
    # update index list of non-zero / non-nan voxels in whole data set
    nonzerovox = nonzerovox[kpvox]
    print('Done.\n', flush=True)

    # stick = cylinder_models.C1Stick()
    # zeppelin = gaussian_models.G2Zeppelin()
    # ball = gaussian_models.G1Ball()
    # bundle = BundleModel([stick, zeppelin])
    # bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
    # bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    # bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)
    # model = modeling_framework.MultiCompartmentSphericalMeanModel(models=[bundle, ball])
    # smt_noddi.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)
    # parameter_vector_init['G1Ball_1_lambda_iso'] = 2.5e-9 * np.ones(parameter_vector_init['partial_volume_0'].shape)
    # parameter_vector_init_orig['G1Ball_1_lambda_iso'] = 2.5e-9 * np.zeros([nx, ny, nz])

    print('data shape = ' + str(data[kpvox, ].shape) + '; mask shape = ' + str(mask[kpvox, ].shape), flush=True)
    print('parameter initial vector shape = ' + str(parameter_vector_init[model.parameter_names[0]].shape), flush=True)
    print('simulated signal shape = ' + str(E_fit[kpvox, ].shape), flush=True)
    print('no. b-values in acq_scheme = ' + str(acq_scheme.bvalues.__len__()))

    #-------------------------------------------------------------------------------------------------------------------
    # hierarchical Bayesian fitting
    print('Running Bayesian fit...', flush=True)
    proc_start = time.time()
    acceptance_rate_tmp, param_conv_tmp, parameter_vector_bayes_tmp, likelihood_stored_tmp, w_stored_tmp \
        = fit_bayes.fit(model, acq_scheme_orig, data[kpvox, :], E_fit[kpvox, :], parameter_vector_init, mask[kpvox], nsteps, burn_in)
    compute_time(proc_start, time.time())
    print('Done\n', flush=True)
    path = os.path.split(path_dmri)
    print('Saving _tmp to ' + path[0] + '/fit_' + model_name + '_nsteps' + str(nsteps) + '_burn' + str(burn_in) + '_w0.05_tmp.npy', flush=True)
    to_save_tmp = {}
    to_save_tmp['nsteps'] = nsteps
    to_save_tmp['burn_in'] = burn_in
    to_save_tmp['parameter_vector_bayes_tmp'] = parameter_vector_bayes_tmp
    to_save_tmp['parameter_vector_init_tmp'] = parameter_vector_init_orig
    np.save(path[0] + '/fit_' + model_name + '_nsteps' + str(nsteps) + '_burn' + str(burn_in) + '_w0.05_tmp1.npy', to_save_tmp, allow_pickle=True, fix_imports=True)
    to_save_tmp['acceptance_rate_tmp'] = acceptance_rate_tmp
    to_save_tmp['param_conv_tmp'] = param_conv_tmp
    to_save_tmp['likelihood_stored_tmp'] = likelihood_stored_tmp
    to_save_tmp['w_stored_tmp'] = w_stored_tmp
    np.save(path[0] + '/fit_' + model_name + '_nsteps' + str(nsteps) + '_burn' + str(burn_in) + '_w0.05_tmp2.npy', to_save_tmp, allow_pickle=True, fix_imports=True)

    # -------------------------------------------------------------------------------------------------------------------

    # reshape and deal with removed nans
    # initialise dictionaries to store output
    print('Initialising dictionaries to store output...', flush=True)
    acceptance_rate = dict.fromkeys(parameter_vector_init.keys(), [])
    param_conv = dict.fromkeys(parameter_vector_init.keys(), [])
    parameter_vector_bayes = dict.fromkeys(parameter_vector_init.keys(), [])
    likelihood_stored = dict.fromkeys(parameter_vector_init.keys(), [])
    w_stored = dict.fromkeys(parameter_vector_init.keys(), [])
    # parameters not including parameter_vector_bayes (this additionally has partial_volume_1)
    for param in acceptance_rate_tmp.keys():
        print(param)
        if model.parameter_cardinality[param] > 1:
            # initialise array to size of original data
            acceptance_rate[param] = np.zeros([nx*ny*nz, 2, nsteps])
            param_conv[param] = np.zeros([nx*ny*nz, 2, nsteps])
            likelihood_stored[param] = np.zeros([nx*ny*nz, 2, nsteps])
            w_stored[param] = np.zeros([nx*ny*nz, 2, int(burn_in/100)])
            # copy bayesian parameters into correct voxels
            acceptance_rate[param][nonzerovox, :, :] = acceptance_rate_tmp[param]
            param_conv[param][nonzerovox, :, :] = param_conv_tmp[param]
            likelihood_stored[param][nonzerovox, :, :] = likelihood_stored_tmp[param]
            w_stored[param][nonzerovox, :, :] = w_stored_tmp[param]
        elif model.parameter_cardinality[param] == 1:
            acceptance_rate[param] = np.zeros([nx*ny*nz, nsteps])
            param_conv[param] = np.zeros([nx*ny*nz, nsteps])
            likelihood_stored[param] = np.zeros([nx*ny*nz, nsteps])
            w_stored[param] = np.zeros([nx*ny*nz, w_stored_tmp[param].shape[1]])  # np.zeros([nx*ny*nz, int(burn_in/100)])
            # copy bayesian parameters into correct voxels
            acceptance_rate[param][nonzerovox, :] = acceptance_rate_tmp[param]
            param_conv[param][nonzerovox, :] = param_conv_tmp[param]
            likelihood_stored[param][nonzerovox, :] = likelihood_stored_tmp[param]
            w_stored[param][nonzerovox, :] = w_stored_tmp[param]
    # only parameter_vector_bayes (this additionally has partial_volume_1)
    for param in parameter_vector_bayes_tmp.keys():
        if model.parameter_cardinality[param] > 1:
            # initialise array to size of original data
            parameter_vector_bayes[param] = np.zeros([nx*ny*nz, 2])
            # copy bayesian parameters into correct voxels
            parameter_vector_bayes[param][nonzerovox, :] = parameter_vector_bayes_tmp[param]
        elif model.parameter_cardinality[param] == 1:
            parameter_vector_bayes[param] = np.zeros([nx*ny*nz])
            # copy bayesian parameters into correct voxels
            parameter_vector_bayes[param][nonzerovox] = parameter_vector_bayes_tmp[param]

    to_save = {}
    to_save['acceptance_rate'] = acceptance_rate
    to_save['param_conv'] = param_conv
    to_save['parameter_vector_bayes'] = parameter_vector_bayes
    to_save['parameter_vector_init'] = parameter_vector_init_orig
    to_save['likelihood_stored'] = likelihood_stored
    to_save['w_stored'] = w_stored
    # to_save['lsq_fit'] = lsq_fit
    to_save['nsteps'] = nsteps
    to_save['burn_in'] = burn_in
    path = os.path.split(path_dmri)
    fname = '/fit_' + model_name + '_nsteps' + str(nsteps) + '_burn' + str(burn_in) + '_w0.01_w0.03.npy'
    np.save(path[0] + fname, to_save, allow_pickle=True, fix_imports=True)
    print('Saving reshaped to ' + path[0] + fname, flush=True)

    '''
    fname = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/fit_smt_noddi_nsteps2000_burn1000_w0.1.npy'
    results = np.load(fname, allow_pickle=True, fix_imports=True)
    acceptance_rate = results.item().get('acceptance_rate')
    param_conv = results.item().get('param_conv')
    parameter_vector_bayes = results.item().get('parameter_vector_bayes')
    parameter_vector_init = results.item().get('parameter_vector_init')
    likelihood_stored = results.item().get('likelihood_stored')
    w_stored = results.item().get('w_stored')
    nsteps = results.item().get('parameter_vector_bayes')
    parameter_vector_bayes = results.item().get('parameter_vector_bayes')
    '''

    '''
    plt.rcParams.update({'font.size': 22})

    fig, axs = plt.subplots(1, len(parameter_vector_init))
    for p in range(len(parameter_vector_init_orig)):
        param = model.parameter_names[p]
        tmp = parameter_vector_init_orig[param]
        # tmp = np.reshape(tmp, (nx, ny))
        axs[p].imshow(tmp)
        axs[p].set_title(model.parameter_names[p])
        make_square_axes(axs[p])
    
    fig, axs = plt.subplots(1, len(parameter_vector_bayes))
    for p in range(len(parameter_vector_bayes)):
        param = model.parameter_names[p]
        print(param)
        tmp = parameter_vector_bayes[param]
        tmp = np.reshape(tmp, (nx, ny))
        axs[p].imshow(tmp)
        axs[p].set_title(model.parameter_names[p])
        make_square_axes(axs[p])
    
    nr = 3
    fig, axs = plt.subplots(nr, len(parameter_vector_bayes)-1)
    lw = 5
    vox = nonzerovox[1]
    nbins = 25
    for p in range(len(parameter_vector_bayes)-1):
        vals = param_conv[model.parameter_names[p]][nonzerovox[vox], burn_in:-1]
        axs[0, p].plot(vals)
        axs[1, p].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
        axs[2, p].plot(acceptance_rate[model.parameter_names[p]][vox, ])
        axs[0, p].set_title(model.parameter_names[p])
        for row in range(nr):
            make_square_axes(axs[row, p])
    '''
    print('Saving mean values as .nii', flush=True)
    for param in parameter_vector_bayes.keys():
        print(param)
        if model.parameter_cardinality[param] > 1:
            img = parameter_vector_bayes[param]
            img = np.reshape(img, (nx, ny, nz, 2))
        elif model.parameter_cardinality[param] == 1:
            img = parameter_vector_bayes[param]
            img = np.reshape(img, (nx, ny, nz))
        print(param)
        img = nib.Nifti1Image(img, np.eye(4), header=hdr[param])
        nib.save(img, os.path.join(path[0], param+'.nii'))
    print('Done.\n', flush=True)

    '''
    #===================================================================================================================
    # print: initialisation, correct value, mean (after burn-in) Bayes-fitted value
    nparams = np.sum(np.array(list(model.parameter_cardinality.values())))
    roi_vals = np.unique(mask)[np.unique(mask) > 0]  # list of unique integers that identify each ROI (ignore 0's)
    roi_nvox = [[xx for xx, x in enumerate(mask == roi_vals[roi]) if x].__len__() for roi in
                range(roi_vals.__len__())]  # number of voxels in each ROI
    to_remove = [roi for roi in range(roi_vals.__len__()) if
                 roi_nvox[roi] < 2 * nparams]  # indices of ROIs with too few voxels
    roi_vals = np.delete(roi_vals, to_remove)
    idx_roi = [xx for xx, x in enumerate(mask == roi_vals[0]) if x]
    vox0 = idx_roi[0]
    vox1 = idx_roi[1]
    vox2 = idx_roi[2]
    vox3 = idx_roi[3]

    # ------------------------------------------------------------------------------------------------------------------
    # filename = '/home/epowell/code/python/dmipy-bayesian/data/shepp_logan_' + str(dims[0]) + 'x' + str(dims[1]) \
    #            + '_snr25_nsteps' + str(nsteps) + '_burn' + str(burn_in) + '.db'
    # save_workspace(filename)

    # ------------------------------------------------------------------------------------------------------------------
    # plot parameter convergence
    plt.rcParams.update({'font.size': 22})
    lw = 5
    fig, axs = plt.subplots(2, 4)

    axs[0, 0].plot(range(nsteps), param_conv['C1Stick_1_lambda_par'][vox0, :], color='seagreen', linewidth=lw)
    axs[0, 0].set_ylabel("D (stick) [$\mu$m/ms]")
    axs[0, 0].set_xlabel("MCMC iteration")
    axs[0, 0].set_ylim([model.parameter_ranges['C1Stick_1_lambda_par'][0] * 1e-9,
                        model.parameter_ranges['C1Stick_1_lambda_par'][1] * 1e-9])
    make_square_axes(axs[0, 0])

    axs[0, 1].plot(range(nsteps), param_conv['G1Ball_1_lambda_iso'][vox1, :], color='steelblue', linewidth=lw)
    axs[0, 1].set_ylabel("D (ball) [$\mu$m/ms]")
    axs[0, 1].set_xlabel("MCMC iteration")
    axs[0, 1].set_ylim([model.parameter_ranges['G1Ball_1_lambda_iso'][0] * 1e-9,
                        model.parameter_ranges['G1Ball_1_lambda_iso'][1] * 1e-9])
    make_square_axes(axs[0, 1])

    axs[0, 2].plot(range(nsteps), param_conv['partial_volume_0'][vox2, :], color='indigo', linewidth=lw)
    axs[0, 2].set_ylabel("f (stick) [a.u.]")
    axs[0, 2].set_xlabel("MCMC iteration")
    axs[0, 2].set_ylim(model.parameter_ranges['partial_volume_0'])
    make_square_axes(axs[0, 2])

    axs[0, 3].plot(range(nsteps), param_conv['C1Stick_1_mu'][vox3, 0, :], color='gold', linewidth=lw)
    axs[0, 3].set_ylabel("stick orientation [rad]")
    axs[0, 3].set_xlabel("MCMC iteration")
    axs[0, 3].set_ylim(model.parameter_ranges['C1Stick_1_mu'][0])
    make_square_axes(axs[0, 3])

    # plot parameter distributions after burn-in period
    nbins = 10
    vals = param_conv['C1Stick_1_lambda_par'][vox0, burn_in:-1] * 1e9  # multiply by 1e9 so gaussian has same scaling
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 0].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 0].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='seagreen', linewidth=lw)
    axs[1, 0].set_ylabel("frequency density")
    axs[1, 0].set_xlabel("D (stick) [$\mu$m/ms]")
    make_square_axes(axs[1, 0])

    vals = param_conv['G1Ball_1_lambda_iso'][vox1, burn_in:-1] * 1e9  # multiply by 1e9 so gaussian has same scaling
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 1].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 1].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='steelblue', linewidth=lw)
    axs[1, 1].set_ylabel("frequency density")
    axs[1, 1].set_xlabel("D (ball) [$\mu$m/ms]")
    make_square_axes(axs[1, 1])

    vals = param_conv['partial_volume_0'][vox2, burn_in:-1]
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 2].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 2].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='indigo', linewidth=lw)
    axs[1, 2].set_ylabel("frequency density")
    axs[1, 2].set_xlabel("f (stick) [a.u.]")
    make_square_axes(axs[1, 2])

    vals = param_conv['C1Stick_1_mu'][vox3, 0, burn_in:-1]
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 3].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 3].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='gold', linewidth=lw)
    axs[1, 3].set_ylabel("frequency density")
    axs[1, 3].set_xlabel("stick orientation [rad]")
    make_square_axes(axs[1, 3])

    # ------------------------------------------------------------------------------------------------------------------
    # plot acceptance rate
    fig, axs = plt.subplots(1, 2)
    axs[0].set_ylabel("Acceptance Rate")
    axs[0].plot(range(nsteps), acceptance_rate['C1Stick_1_lambda_par'][vox0], color='seagreen', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['G1Ball_1_lambda_iso'][vox1, :], color='steelblue', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['partial_volume_0'][vox2, :], color='indigo', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['C1Stick_1_mu'][vox3, 0, :], color='gold', linewidth=lw)
    axs[0].legend(['Dpar', 'Diso', 'fpar', 'mu'])

    # plot likelihood
    axs[1].set_ylabel("Likelihood")
    axs[1].plot(range(nsteps), likelihood_stored['C1Stick_1_lambda_par'][vox0, :], color='seagreen', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['G1Ball_1_lambda_iso'][vox1, :], color='steelblue', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['partial_volume_0'][vox2, :], color='indigo', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['C1Stick_1_mu'][vox3, 0, :], color='gold', linewidth=lw)
    axs[1].legend(['Dpar', 'Diso', 'fpar', 'mu'])

    # ------------------------------------------------------------------------------------------------------------------
    # plot maps: LSQ, Bayes
    plt.rcParams.update({'font.size': 42})
    fig = plt.figure(figsize=(2, 4))
    grid = AxesGrid(fig, 111, nrows_ncols=(2, 4), axes_pad=0, cbar_mode='edge', cbar_location='bottom', cbar_pad=.25)
    cmap_D = copy(mpl.cm.BuPu_r)
    cmap_D.set_bad(color='k')
    cmap_f = copy(mpl.cm.OrRd_r)
    cmap_f.set_bad(color='k')
    cmap_mu = copy(mpl.cm.YlGn_r)
    cmap_mu.set_bad(color='k')
    clims_D = [0e-9, 3e-9]
    clims_f = [0, .75]
    clims_mu = [0, np.pi]
    # remove axes ticks and labels
    for g in range(8):
        grid[g].axes.set_xticklabels([])
        grid[g].axes.set_yticklabels([])
        grid[g].axes.set_xticks([])
        grid[g].axes.set_yticks([])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D),
                        cax=grid[0].cax, orientation='horizontal', label='0-3 $\mu$m$^2$/ms')
    cbar.set_ticks([])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D),
                        cax=grid[1].cax, orientation='horizontal', label='0-3 $\mu$m$^2$/ms')
    cbar.set_ticks([])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_f[0], vmax=clims_f[1]), cmap=cmap_f),
                        cax=grid[2].cax, orientation='horizontal', label='0-0.75 a.u.')
    cbar.set_ticks([])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_mu[0], vmax=clims_mu[1]), cmap=cmap_mu),
        cax=grid[3].cax, orientation='horizontal', label='0-$\pi$ rad')
    cbar.set_ticks([])

    # transform to rotate brains by 90
    t = transforms.Affine2D().rotate_deg(90)

    # LSQ
    im = np.reshape(parameter_vector_lsq['C1Stick_1_lambda_par'], dims)
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[0].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_lsq['G1Ball_1_lambda_iso'], dims)
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[1].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_lsq['partial_volume_0'], dims)
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[2].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_lsq['C1Stick_1_mu'][:, 0], dims)
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[3].imshow(im, vmin=clims_mu[0], vmax=clims_mu[1], cmap=cmap_mu)

    # Bayes
    im = np.reshape(parameter_vector_bayes['C1Stick_1_lambda_par'], dims)
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[4].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_bayes['G1Ball_1_lambda_iso'], dims)
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[5].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_bayes['partial_volume_0'], dims)
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[6].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_bayes['C1Stick_1_mu'][:, 0], dims)
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[7].imshow(im, vmin=clims_mu[0], vmax=clims_mu[1], cmap=cmap_mu)

    grid[0].set_title('D (stick)')
    grid[1].set_title('D (ball)')
    grid[2].set_title('f (stick)')
    grid[3].set_title('orientation (stick)')
    grid[0].set_ylabel('LSQ', rotation=0, labelpad=50)
    grid[4].set_ylabel('Bayesian', rotation=0, labelpad=100)

    '''


if __name__ == '__main__':
    main()
