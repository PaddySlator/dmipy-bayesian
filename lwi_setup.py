#luminal water imaging example

# load some modules
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.stats
import shelve
import os

# for fitting
from dmipy.core import modeling_framework
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, gaussian_models


# -------------------------------------------------------------------------------------------------------------------- #
# FUNCTIONS


# for biexponential model fitting (just with ball ball for now)
def biexpt2(TE,fa,T2a,T2b):
    return fa*np.exp(-TE/T2a) + (1-fa)*np.exp(-TE/T2b)


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


# -------------------------------------------------------------------------------------------------------------------- #
# BALL BALL MODEL FITTING
# load an example LWI image & plot
datadir = '/home/epowell/data/lwi/'
img = nib.load(datadir + 'inn104rwb.nii')
img = img.get_data()
mask = nib.load(datadir + 'inn104rwb_mask_pjs.nii.gz')
mask = mask.get_data()

#plt.imshow(img[:, :, 2, 0])
#plt.imshow(mask[:, :, 3])

# load the acquisition parameters & put them into dmipy format
gradecho = np.loadtxt(datadir + 'LWI_gradechoinv.txt', delimiter=",")
TE = gradecho[:, 4] * 10e6                              # originals are ms - so these are nanoseconds?
TE = TE - np.min(TE)                                    # normalise the TEs to 0
bvecs = gradecho[:, 0:3]
acq_scheme = acquisition_scheme_from_bvalues(TE,bvecs)  # put b-vals as TE so can fit R2 with ball ball model
acq_scheme.print_acquisition_info                       # bvals_SI, bvecs, delta, Delta, TE_SI

print('Data loaded')

# fit biexponential model (just with ball ball for now) & plot signal decay curves of 1 voxel
ball1 = gaussian_models.G1Ball()
ball2 = gaussian_models.G1Ball()
ball_model = MultiCompartmentModel([ball1])
#ball.parameter_scales['G1Ball_1_lambda_iso'] = 1
ballball_model = MultiCompartmentModel((ball1, ball2))
#ballball.parameter_scales['G1Ball_1_lambda_iso'] = 1
#ballball.parameter_scales['G1Ball_2_lambda_iso'] = 1
normvox = img[100, 100, 3, :] / img[100, 100, 3, 0]
ballfit = ball_model.fit(acq_scheme, normvox, Ns=100)
ballballfit = ballball_model.fit(acq_scheme, normvox)
print(mask[100, 100, 3])
ballballfit.fitted_parameters
#plt.plot(TE, np.squeeze(ballballfit.predict(acq_scheme)))
#plt.plot(TE, np.squeeze(ballfit.predict(acq_scheme)))
#plt.plot(TE, normvox)
print('Biexp model fit with 2 Gaussian balls done')

# -------------------------------------------------------------------------------------------------------------------- #
# WILL DEVINE'S MODEL!

# pr = @(s) s(1).*((s(2).*normpdf(t2_iter',s(3),s(5)))+((1-s(2)).*normpdf(t2_iter',s(4),s(6)))); % set up the shape of the fitting, if we use 'pr(s)' we will get a spectrum of T2 distributions that is made up of two gaussians with parameters 's'
# A = exp(-T2'*(1./t2_iter));
# then to calculate the spectrum it's
# P(i,j,k,:) = pr(S(i,j,k,:));
# and to calculate the signal it's
# C(i,j,k,:) = A*pr(S(i,j,k,:));

# define other parameters
T2min, T2max = 0.001, 0.5
T2grid = np.linspace(T2min, T2max, num=1000)
R2min, R2max = 0.001, 0.5                       # should be in nanoseconds I think (for correct dmipy scaling)
scaling = 1e-09
R2grid = np.linspace(R2min * scaling, R2max * scaling, num=1000)
f = 0.3
mu1 = 1e-11
sigma1 = 1e-12
mu2 = 1e-10
sigma2 = 3e-12

# compute spectra
R2spectra1 = scipy.stats.norm.pdf(R2grid, mu1, sigma1)
R2spectra1 = R2spectra1 / np.sum(R2spectra1)    # normalise here instead of the final spectra?
R2spectra2 = scipy.stats.norm.pdf(R2grid, mu2, sigma2)
R2spectra2 = R2spectra2 / np.sum(R2spectra2)
R2spectra = f * R2spectra1 + (1 - f) * R2spectra2
print('WD model: spectra computed')

# normalise spectra (required so that the dmipy signal output is normalised)
# R2spectra = R2spectra/np.sum(R2spectra)
#plt.plot(R2grid, R2spectra)

# make the signal dictionary ("A" here - I usually use "K")
A = np.zeros((len(TE), len(R2grid)))
for i in range(0, len(R2grid)):
    parameter_vector = ball_model.parameters_to_parameter_vector(G1Ball_1_lambda_iso=R2grid[i])
    A[:, i] = ball_model.simulate_signal(acq_scheme, parameter_vector)
    # A[:,i] = np.exp(-TE * R2grid[i])
print(np.shape(A))
print(np.shape(R2spectra))
#plt.plot(TE, np.matmul(A, R2spectra))

# try and get the ballnormaldist working!
balldist1 = gaussian_models.G1BallNormalDist()
balldist2 = gaussian_models.G1BallNormalDist()
balldist_model = MultiCompartmentModel((balldist1, balldist2))
parameter_vector = balldist_model.parameters_to_parameter_vector(G1BallNormalDist_2_lambda_iso_mean=1e-09,
                                                                 G1BallNormalDist_1_lambda_iso_mean=2e-09,
                                                                 G1BallNormalDist_1_lambda_iso_std=1e-10,
                                                                 G1BallNormalDist_2_lambda_iso_std=1e-10,
                                                                 partial_volume_0=0.7,
                                                                 partial_volume_1=0.3)

S = balldist_model.simulate_signal(acq_scheme, parameter_vector)
#plt.plot(S)

# naive fit
#balldist_fit = balldist_model.fit(acq_scheme, S)
#balldist_fit.fitted_parameters

# initialise the fit with simple biexponential fit
#ballball_fit = ballball_model.fit(acq_scheme, S)
#balldist_model.set_initial_guess_parameter('G1BallNormalDist_1_lambda_iso_mean', ballball_fit.fitted_parameters['G1Ball_2_lambda_iso'])
#balldist_model.set_initial_guess_parameter('G1BallNormalDist_2_lambda_iso_mean', ballball_fit.fitted_parameters['G1Ball_1_lambda_iso'])
#balldist_model.set_initial_guess_parameter('G1BallNormalDist_1_lambda_iso_std', 5e-3)
#balldist_model.set_initial_guess_parameter('G1BallNormalDist_2_lambda_iso_std', 5e-3)
#balldist_model.set_initial_guess_parameter('partial_volume_1', ballball_fit.fitted_parameters['partial_volume_0'])
#balldist_model.set_initial_guess_parameter('partial_volume_0', ballball_fit.fitted_parameters['partial_volume_1'])
#balldist_model.fit(acq_scheme, S)

#balldist_fit.fitted_parameters

# try fitting ball dist on whole image (use the LWI loaded earlier)
# setup
balldist_fit = balldist_model.fit(acq_scheme, img, mask=mask)
#PUT THE DICTIONARY AS AN INPUT LIKE B-VALUES I.E. IN THE ACQUISITION SCHEME, OR AS AN ADDITIONAL OPTION?

save_workspace(os.getcwd() + '/lwi.db')
