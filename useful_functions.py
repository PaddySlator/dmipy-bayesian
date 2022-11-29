import numpy as np
import math
import shelve
from contextlib import contextmanager
import sys
import os

from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues


# to temporarily suppress output to console
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Make axes square
def make_square_axes(ax):
    ax.set_aspect(1 / ax.get_data_ratio())


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


# cartesian coords to polar angles
def cart2mu(xyz):
    shape = xyz.shape[:-1]
    mu = np.zeros(np.r_[shape, 2])
    r = np.linalg.norm(xyz, axis=-1)
    mu[..., 0] = np.arccos(xyz[..., 2] / r)  # theta
    mu[..., 1] = np.arctan2(xyz[..., 1], xyz[..., 0])
    mu[r == 0] = 0, 0
    return mu


def create_spherical_mean_scheme(acq_scheme):
    acq_scheme_smt = acq_scheme.spherical_mean_scheme
    # create fake gradient directions
    grad_dirs = np.tile([1,1,1] / np.linalg.norm([1,1,1]), [np.unique(acq_scheme.bvalues).shape[0], 1])
    # grad_dirs = np.tile([0,0,0], [np.unique(acq_scheme.bvalues).shape[0], 1])
    acq_scheme_smt = acquisition_scheme_from_bvalues(np.unique(acq_scheme.bvalues), grad_dirs, acq_scheme.delta[0], acq_scheme.Delta[0])
    # acq_scheme_smt.gradient_directions = grad_dirs
    
    return acq_scheme_smt


# add noise
def add_noise(data, snr=50):
    data_real = data + np.random.normal(scale=1/snr, size=np.shape(data))
    data_imag = np.random.normal(scale=1/snr, size=np.shape(data))
    data_noisy = np.sqrt(data_real**2 + data_imag**2)

    return data_noisy
    