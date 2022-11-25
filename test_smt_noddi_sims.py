# load some necessary modules
'''
import scipy.stats
from os import listdir
from os.path import join
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import AxesGrid
import math
import shelve
'''

import numpy as np
from copy import copy, deepcopy
import time
from importlib import reload

# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes

import fit_bayes_new 
fit_bayes_new = reload(fit_bayes_new)
# from fit_bayes_new import fit, tform_params  # , dict_to_array, array_to_dict

from useful_functions import suppress_stdout, compute_time, create_spherical_mean_scheme, add_noise
import setup_models

def load_sim_data(path_params_gt, path_rois, path_signals_gt, path_signals_noisy):
    '''
    ['BundleModel_1_C1Stick_1_lambda_par',
     'BundleModel_1_G2Zeppelin_1_lambda_par',
     'BundleModel_1_partial_volume_0',
     'partial_volume_0',
     'partial_volume_1']
    '''
    params_gt = np.load(path_params_gt)
    rois = np.load(path_rois)
    signals_gt = np.load(path_signals_gt)
    signals_noisy = np.load(path_signals_noisy,allow_pickle=True).all()

    return params_gt, rois, signals_gt, signals_noisy


def simulate_data(smt_noddi, acq_scheme):

    # limits on the parameters for wm/gm/csf
    stick_par_wm = [1e-9, 2e-9]
    stick_par_gm = [1e-9, 2e-9]
    stick_par_csf = [1e-9, 2e-9]
    zep_par_wm = [1.5e-9, 2.5e-9]
    zep_par_gm = [1.5e-9, 2.5e-9]
    zep_par_csf = [1.5e-9, 2.5e-9]
    odi_wm = [0.01, 0.3]
    odi_gm = [0.6, 0.9]
    odi_csf = [0, 1]
    f_stick_wm = [0.7, 0.9]
    f_stick_gm = [0.6, 8]
    f_stick_csf = [0.8, 1]
    f_bundle_wm = [0.7, 0.9]
    f_bundle_gm = [0.8, 1]
    f_bundle_csf = [0, 0.2]

    # number of voxels for each tissue type
    n_wm = 500
    n_gm = 300
    n_csf = 100

    # roi_mask = np.concatenate((np.ones((n_wm,)),2*np.ones((n_gm,)),3*np.ones((n_csf,))))
    roi_mask = np.concatenate((np.ones((n_wm,)),2*np.ones((n_gm,)),np.zeros((n_csf,))))

    stick_par = np.concatenate((np.random.uniform(low=stick_par_wm[0],high=stick_par_wm[1], size = n_wm),
                          np.random.uniform(low=stick_par_gm[0],high=stick_par_gm[1], size = n_gm),
                          np.random.uniform(low=stick_par_csf[0],high=stick_par_csf[1], size = n_csf)))

    zep_par = np.concatenate((np.random.uniform(low=zep_par_wm[0],high=zep_par_wm[1], size = n_wm),
                          np.random.uniform(low=zep_par_gm[0],high=zep_par_gm[1], size = n_gm),
                          np.random.uniform(low=zep_par_csf[0],high=zep_par_csf[1], size = n_csf)))

    odi = np.concatenate((np.random.uniform(low=odi_wm[0],high=odi_wm[1], size = n_wm),
                          np.random.uniform(low=odi_gm[0],high=odi_gm[1], size = n_gm),
                          np.random.uniform(low=odi_csf[0],high=odi_csf[1], size = n_csf)))                

    f_stick = np.concatenate((np.random.uniform(low=f_bundle_wm[0],high=f_bundle_wm[1], size = n_wm),
                          np.random.uniform(low=f_bundle_gm[0],high=f_bundle_gm[1], size = n_gm),
                          np.random.uniform(low=f_bundle_csf[0],high=f_bundle_csf[1], size = n_csf)))

    f_bundle = np.concatenate((np.random.uniform(low=f_bundle_wm[0],high=f_bundle_wm[1], size = n_wm),
                          np.random.uniform(low=f_bundle_gm[0],high=f_bundle_gm[1], size = n_gm),
                          np.random.uniform(low=f_bundle_csf[0],high=f_bundle_csf[1], size = n_csf)))

    f_free = 1 - f_bundle

    # put into a big parameter vector that can be passed into simulate_signal
    parameters_smt_noddi = smt_noddi.parameters_to_parameter_vector(BundleModel_1_partial_volume_0=f_stick,
                                                                    partial_volume_0=f_bundle,
                                                                    partial_volume_1=f_free)
    
    signals = smt_noddi.simulate_signal(acq_scheme,parameters_smt_noddi)
    
    return signals, parameters_smt_noddi, roi_mask


def check_lsq_fit(model, parameters_lsq_dict):
    for param in model.parameter_names:  
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                idx = parameters_lsq_dict[param][:, card] <= model.parameter_ranges[param][card][0] * model.parameter_scales[param]
                parameters_lsq_dict[param][idx, card] = (model.parameter_ranges[param][card][0] + np.finfo(float).eps) * model.parameter_scales[param][card]
                idx = parameters_lsq_dict[param][:, card] >= model.parameter_ranges[param][card][1] * model.parameter_scales[param]
                parameters_lsq_dict[param][idx, card] = (model.parameter_ranges[param][card][1] - np.finfo(float).eps) * model.parameter_scales[param][card]
        elif model.parameter_cardinality[param] == 1:
            idx = parameters_lsq_dict[param] <= model.parameter_ranges[param][0] * model.parameter_scales[param]
            parameters_lsq_dict[param][idx] = (model.parameter_ranges[param][0] + np.finfo(float).eps) * model.parameter_scales[param]
            idx = parameters_lsq_dict[param] >= model.parameter_ranges[param][1] * model.parameter_scales[param]
            parameters_lsq_dict[param][idx] = (model.parameter_ranges[param][1] - np.finfo(float).eps) * model.parameter_scales[param]
   
    return parameters_lsq_dict

    
def main():

    nsteps = 1000
    burn_in = 500
    nupdates = 20

    # setup acquisition scheme
    acq_scheme_ful = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_scheme_smt = create_spherical_mean_scheme(acq_scheme_ful)
    
    # set up model
    smt_noddi = setup_models._smt_noddi()

    # generate ground truth parameters and corresponding (spherical mean) signals
    signals_gt, parameters_smt_noddi, mask = simulate_data(smt_noddi, acq_scheme_ful)
    
    # add noise
    signals_snr10 = add_noise(signals_gt, snr=10)
    signals_snr100 = add_noise(signals_gt, snr=100)

    # LSQ fitting
    lsq_fit = smt_noddi.fit(acq_scheme_smt, signals_snr100)
    parameters_lsq_dict = deepcopy(lsq_fit.fitted_parameters)
    parameters_lsq_vect = deepcopy(lsq_fit.fitted_parameters_vector)
    
    # LSQ-predicted signals
    with suppress_stdout():  # suppress annoying output in console
        E_fit = smt_noddi.simulate_signal(acq_scheme_smt, parameters_lsq_vect)
        E_fit = np.squeeze(E_fit)
        
    # check LSQ fits don't hit the bounds; add/subtract eps to any that do
    parameters_lsq_dict = check_lsq_fit(smt_noddi, parameters_lsq_dict)
 
    # hierarchical Bayesian fitting
    proc_start = time.time()
    acceptance_rate, param_conv, parameter_vector_bayes, parameter_vector_init, likelihood_stored, w_stored \
        = fit_bayes_new.fit(smt_noddi, acq_scheme_smt, signals_gt, E_fit, parameters_lsq_dict, mask, nsteps, burn_in, nupdates)
    compute_time(proc_start, time.time())

    '''
    parameter_vector_lsq['G1Ball_1_lambda_iso'] = np.zeros(dim * dim)
    parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'] = np.zeros(dim * dim)
    # revert LSQ diffusivities back to fixed values
    parameter_vector_lsq['G1Ball_1_lambda_iso'][mask_glob] = lambda_iso
    parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][mask_glob] = lambda_par
    parameter_vector_bayes['G1Ball_1_lambda_iso'] = parameter_vector_lsq['G1Ball_1_lambda_iso']
    parameter_vector_bayes['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'] = parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par']

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
    fig, axs = plt.subplots(2, 5)

    axs[0, 0].plot(range(nsteps), param_conv['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][vox0, :], color='seagreen', linewidth=lw)
    axs[0, 0].set_ylabel("D (stick/zeppelin) [$\mu$m/ms]")
    axs[0, 0].set_xlabel("MCMC iteration")
    axs[0, 0].set_ylim([model.parameter_ranges['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][0] * 1e-9,
                        model.parameter_ranges['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][1] * 1e-9])
    make_square_axes(axs[0, 0])

    axs[0, 1].plot(range(nsteps), param_conv['G1Ball_1_lambda_iso'][vox1, :], color='steelblue', linewidth=lw)
    axs[0, 1].set_ylabel("D (ball) [$\mu$m/ms]")
    axs[0, 1].set_xlabel("MCMC iteration")
    axs[0, 1].set_ylim([model.parameter_ranges['G1Ball_1_lambda_iso'][0] * 1e-9,
                        model.parameter_ranges['G1Ball_1_lambda_iso'][1] * 1e-9])
    make_square_axes(axs[0, 1])

    axs[0, 2].plot(range(nsteps), param_conv['partial_volume_0'][vox2, :], color='indigo', linewidth=lw)
    axs[0, 2].set_ylabel("f (ball) [a.u.]")
    axs[0, 2].set_xlabel("MCMC iteration")
    axs[0, 2].set_ylim(model.parameter_ranges['partial_volume_0'])
    make_square_axes(axs[0, 2])

    axs[0, 3].plot(range(nsteps), param_conv['SD1WatsonDistributed_1_partial_volume_0'][vox2, :], color='red', linewidth=lw)
    axs[0, 3].set_ylabel("f (Watson stick) [a.u.]")
    axs[0, 3].set_xlabel("MCMC iteration")
    axs[0, 3].set_ylim(model.parameter_ranges['SD1WatsonDistributed_1_partial_volume_0'])
    make_square_axes(axs[0, 3])

    axs[0, 4].plot(range(nsteps), param_conv['SD1WatsonDistributed_1_SD1Watson_1_odi'][vox3, :], color='gold', linewidth=lw)
    axs[0, 4].set_ylabel("ODI")
    axs[0, 4].set_xlabel("MCMC iteration")
    axs[0, 4].set_ylim(model.parameter_ranges['SD1WatsonDistributed_1_SD1Watson_1_odi'][0])
    make_square_axes(axs[0, 4])

    # plot parameter distributions after burn-in period
    nbins = 15
    vals = param_conv['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][vox0, burn_in:-1] * 1e9  # multiply by 1e9 so gaussian has same scaling
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 0].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 0].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='seagreen', linewidth=lw)
    axs[1, 0].set_ylabel("frequency density")
    axs[1, 0].set_xlabel("D (stickzeppelin) [$\mu$m/ms]")
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
    axs[1, 2].set_xlabel("f (ball) [a.u.]")
    make_square_axes(axs[1, 2])

    vals = param_conv['SD1WatsonDistributed_1_partial_volume_0'][vox2, burn_in:-1]
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 3].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 3].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='red', linewidth=lw)
    axs[1, 3].set_ylabel("frequency density")
    axs[1, 3].set_xlabel("f (Watson stick) [a.u.]")
    make_square_axes(axs[1, 3])

    vals = param_conv['SD1WatsonDistributed_1_SD1Watson_1_odi'][vox3, burn_in:-1]
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 4].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 4].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='gold', linewidth=lw)
    axs[1, 4].set_ylabel("frequency density")
    axs[1, 4].set_xlabel("ODI")
    make_square_axes(axs[1, 4])

    # ------------------------------------------------------------------------------------------------------------------
    # plot acceptance rate
    fig, axs = plt.subplots(1, 2)
    axs[0].set_ylabel("Acceptance Rate")
    axs[0].plot(range(nsteps), acceptance_rate['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][vox0], color='seagreen', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['G1Ball_1_lambda_iso'][vox0, :], color='steelblue', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['partial_volume_0'][vox0, :], color='indigo', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['SD1WatsonDistributed_1_partial_volume_0'][vox0, :], color='red', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['SD1WatsonDistributed_1_SD1Watson_1_odi'][vox0, :], color='gold', linewidth=lw)
    axs[0].legend(['Dpar', 'Diso', 'fball', 'fWatsonstick', 'ODI'])

    # plot likelihood
    axs[1].set_ylabel("Likelihood")
    axs[1].plot(range(nsteps), likelihood_stored['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][vox0, :], color='seagreen', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['G1Ball_1_lambda_iso'][vox0, :], color='steelblue', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['partial_volume_0'][vox0, :], color='indigo', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['SD1WatsonDistributed_1_partial_volume_0'][vox0, :], color='red', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['SD1WatsonDistributed_1_SD1Watson_1_odi'][vox0, :], color='gold', linewidth=lw)
    axs[1].legend(['Dpar', 'Diso', 'fball', 'fWatsonstick', 'ODI'])

    # ------------------------------------------------------------------------------------------------------------------
    # plot maps: LSQ, Bayes, GT
    plt.rcParams.update({'font.size': 42})
    fig = plt.figure(figsize=(3, 5))
    grid = AxesGrid(fig, 111, nrows_ncols=(3, 5), axes_pad=0, cbar_mode='edge', cbar_location='bottom', cbar_pad=.25)
    cmap_D = copy(mpl.cm.BuPu_r)
    cmap_D.set_bad(color='k')
    cmap_f = copy(mpl.cm.OrRd_r)
    cmap_f.set_bad(color='k')
    cmap_mu = copy(mpl.cm.YlGn_r)
    cmap_mu.set_bad(color='k')
    clims_D = [1.5e-9, 3e-9]
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
    im = np.reshape(parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[0].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_lsq['G1Ball_1_lambda_iso'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[1].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_lsq['partial_volume_0'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[2].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_lsq['SD1WatsonDistributed_1_partial_volume_0'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[3].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_lsq['SD1WatsonDistributed_1_SD1Watson_1_odi'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[4].imshow(im, vmin=clims_mu[0], vmax=clims_mu[1], cmap=cmap_mu)

    # Bayes
    im = np.reshape(parameter_vector_bayes['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[5].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_bayes['G1Ball_1_lambda_iso'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[6].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_bayes['partial_volume_0'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[7].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_bayes['SD1WatsonDistributed_1_partial_volume_0'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[8].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_bayes['SD1WatsonDistributed_1_SD1Watson_1_odi'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[9].imshow(im, vmin=clims_mu[0], vmax=clims_mu[1], cmap=cmap_mu)

    # GT
    im = np.reshape(parameter_vector['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[10].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector['G1Ball_1_lambda_iso'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[11].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector['partial_volume_0'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[12].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector['SD1WatsonDistributed_1_partial_volume_0'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[13].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector['SD1WatsonDistributed_1_SD1Watson_1_odi'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[14].imshow(im, vmin=clims_mu[0], vmax=clims_mu[1], cmap=cmap_mu)

    grid[0].set_title('D (stick/zeppelin)')
    grid[1].set_title('D (ball)')
    grid[2].set_title('f (ball)')
    grid[3].set_title('f (Watson stick)')
    grid[4].set_title('ODI')
    grid[0].set_ylabel('LSQ', rotation=0, labelpad=50)
    grid[5].set_ylabel('Bayesian', rotation=0, labelpad=100)
    grid[10].set_ylabel('GT', rotation=0, labelpad=100)
    '''

if __name__ == '__main__':
    main()
