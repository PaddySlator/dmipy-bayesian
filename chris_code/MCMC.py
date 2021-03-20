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
from dmipy.data import saved_data

# ball stick and spherical mean ball-stick model
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel, MultiCompartmentModel

def Compute_time(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("/nTOTAL OPTIMIZATION TIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    time_string = ("TOTAL TIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    return time_string

def compute_temp_schedule(startTemp, endTemp, MAX_ITER):
        SA_schedule =   []
        it          =   np.arange(MAX_ITER + 1)
        SA_schedule.append([-math.log(endTemp/startTemp)/i for i in it])
        return SA_schedule[0]

def sign_par():
    # acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_scheme, data_hcp = saved_data.wu_minn_hcp_coronal_slice()
    nmeas = len(acq_scheme.bvalues)

    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    ballstick = MultiCompartmentModel(models=[stick,ball])
    return scheme_hcp, data, nmeas, stick, ball, ballstick


def load_data():
    acq_scheme, data, nmeas, stick, ball, ballstick = sign_par()
    # baseline parameters for the simulated ball-stick voxel
    
    E_fit = ballstick.fit(scheme_hcp, data_hcp, mask=data_hcp[..., 0]>0)
    fitted_parameters = E_fit.fitted_parameters

    # plot
    fig, axs = plt.subplots(2, 2, figsize=[10, 10])
    axs = axs.ravel()

    counter = 0
    for name, values in fitted_parameters.items():
        if values.squeeze().ndim != 2:
            continue
        cf = axs[counter].imshow(values.squeeze().T, origin=True, interpolation='nearest')
        axs[counter].set_title(name)
        axs[counter].set_axis_off()
        fig.colorbar(cf, ax=axs[counter], shrink=0.8)
        counter += 1

    mask=data_hcp[..., 0]>0

    # initial voxel-wise estimates from LSQ fits within the mask
    D0_init = fitted_parameters['C1Stick_1_lambda_par'][mask]
    D1_init = fitted_parameters['G1Ball_1_lambda_iso'][mask]
    f0_init = fitted_parameters['partial_volume_0'][mask]
    f1_init = fitted_parameters['partial_volume_1'][mask]
    stick_ori_init = fitted_parameters['C1Stick_1_mu'][mask]

    # flatten params to get [5 x nvox] array
    params_all = np.array([D0_init,
                          D1_init,
                          f0_init,
                          stick_ori_init[:,0],
                          stick_ori_init[:,1]])
    nvox = np.shape(params_all)[1]

    dimx=nvox
    dimy=1

    E_sim = np.zeros((dimx, dimy, nmeas))
    E_sim[:,0,:] = data_hcp[mask]

    E_fit = np.zeros((dimx, dimy, nmeas))
    for x in range(0, dimx):
        for y in range(0,dimy):
            # generate signal
            parameter_vector = ballstick.parameters_to_parameter_vector(
                C1Stick_1_mu=stick_ori_init[x,:],
                C1Stick_1_lambda_par=D0_init[x],
                G1Ball_1_lambda_iso=D1_init[x],
                partial_volume_0=f0_init[x],
                partial_volume_1=1 - f0_init[x])
            E_fit[x, y, :] = ballstick.simulate_signal(acq_scheme, parameter_vector)

    return params_all, E_sim, E_fit, nvox
    
def run_optimization(params_all, E_sim, E_fit, nvox, iter):
    acq_scheme, nmeas, stick, ball, ballstick = sign_par()

    # TO DO: log transform variables (non-orientation only)
    #params_all[:, 0] = np.log(params_all[:, 1])
    #params_all[:, 1] = np.log(params_all[:, 2])
    #params_all[:, 2] = np.log(params_all[:, 3]) - np.log(1 - params_all[:, 3])

    # initialise sigma
    sigma = np.cov(params_all)

    # TO DO: tune weights for Metropolis-Hastings parameter sampling (f, D, D* from Orton, orientations guessed)
    w = [.5e-9, .2e-9, .5, .2, .2]

    nMCMCsteps = iter
    T = compute_temp_schedule(2000, 10**-3, nMCMCsteps)

    Accepted = 1
    Acceptance_rate = []
    Acceptance_rate.append(Accepted)

    tmppar = np.zeros((nvox, nMCMCsteps))
    tmpgibbs = np.zeros((5, nMCMCsteps))

    # NB i (voxel loop) and j (MC loop) in keeping with Orton paper
    for j in range(0, nMCMCsteps):
        print(j)
        it = j+1
        # Gibbs moves to update priors
        # sample mu from multivariate normal distribution defined by current parameter estimates
        m = np.mean(params_all, axis=1)
        V = sigma / nvox
        mu = np.random.multivariate_normal(m, V)
        mu_scaled = mu * [1e9, 1e9, 1, 1, 1]

        # sample sigma from inverse Wishart distribution (using newly updated mu)
        # NB scaled parameters used in calculation of priors in Metropolis-Hastings updates
        phi = np.sum([np.outer(params_all[:, i]-mu,
                            params_all[:, i]-mu)
                    for i in range(0, nvox)], axis=0)
        phi_scaled = np.sum([np.outer((params_all[:, i] - mu)*[1e9, 1e9, 1, 1, 1],
                                    (params_all[:, i] - mu)*[1e9, 1e9, 1, 1, 1])
                            for i in range(0, nvox)], axis=0)
        sigma = scipy.stats.invwishart(scale=phi, df=nvox-5).rvs()
        sigma_scaled = scipy.stats.invwishart(scale=phi_scaled, df=nvox-5).rvs()

        # Metropolis-Hastings parameter updates
        params_all_new = copy.copy(params_all)
        for i in range(0, nvox):
            # TO DO: loop over parameters (Lizzie)
                for p in range(5):
                #p = 0  # for now just look at
                # sample parameter
                params_all_new[p, i] = np.random.normal(params_all[p, i], w[p])
                if i == 0:
                    print([params_all_new[p, i], params_all[p, i]])
                # compute acceptance
                y_i = copy.copy(E_sim[i, ])                                                    # actual measured signal
                g_i = copy.copy(E_fit[i, ])                                                    # model-predicted signal (old params)
                parameter_vector = ballstick.parameters_to_parameter_vector(
                    C1Stick_1_mu=[params_all_new[3, i], params_all_new[4, i]],
                    C1Stick_1_lambda_par=params_all_new[0, i],
                    G1Ball_1_lambda_iso=params_all_new[1, i],
                    partial_volume_0=params_all_new[2, i],
                    partial_volume_1=1-params_all_new[2, i])
                g_i_new = ballstick.simulate_signal(acq_scheme, parameter_vector)   # model-predicted signal (new params)
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
                                                                            mu_scaled, sigma_scaled, allow_singular=1))
                    prior_new_scaled = np.log(scipy.stats.multivariate_normal.pdf(params_all_new[:, i] * [1e9, 1e9, 1, 1, 1],
                                                                                mu_scaled, sigma_scaled, allow_singular=1))
                else:  # orientation parameters (Chris to modify) (scaling here?)
                    prior = 1
                    prior_new = 1
                    prior_scaled = 1
                    prior_new_scaled = 1

                alpha = np.min([0, (likelihood_new + prior_new) - (likelihood + prior)])
                alpha_scaled = np.min([0, (likelihood_new + prior_new_scaled) - (likelihood + prior_scaled)] )
                r = np.log(np.random.uniform(0, 1))

                # reject new parameter value if criteria not met
                if r < alpha_scaled:
                    Accepted +=1
                    params_all[p, i] = copy.copy(params_all_new[p, i])
                    E_fit[i, ] = copy.copy(g_i_new)
                # else:
                #     if Accepted/(it*nvox) < 0.23:
                #         continue
                        # print("Stopping criterion met {}".format(Accepted/(it*nvox)))
                        # return Acceptance_rate

                # for plotting
                if i == 0:
                    tmppar[p, j] = copy.copy(params_all[p, i])
                    tmpgibbs[:, j] = mu

            Acceptance_rate.append(Accepted/(it*nvox) )
            # TO DO: update weights every 100 steps
    return Acceptance_rate
            

def main():
    params_all, E_sim, E_fit, nvox = load_data()
    Proc_start = time.time()
    Acceptance_rate = run_optimization(params_all, E_sim, E_fit, nvox, 200)
    Compute_time(Proc_start,time.time())
    fig, ax = plt.subplots()
    color = 'tab:blue'
    ax.set_ylabel("Acceptance Rate", color=color)
    ax.plot(np.arange(len(Acceptance_rate)), Acceptance_rate, marker=",", color=color)
    ax.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
