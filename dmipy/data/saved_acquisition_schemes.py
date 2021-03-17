import numpy as np
from os.path import join
import os
import pkg_resources
from ..core.acquisition_scheme import (
    acquisition_scheme_from_bvalues,
    acquisition_scheme_from_gradient_strengths,
    acquisition_scheme_from_schemefile)
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
import zipfile

_GRADIENT_TABLES_PATH = pkg_resources.resource_filename(
    'dmipy', 'data/gradient_tables'
)
DATA_PATH = pkg_resources.resource_filename(
    'dmipy', 'data'
)


__all__ = [
    'wu_minn_hcp_acquisition_scheme',
    'duval_cat_spinal_cord_2d_acquisition_scheme'
]


def wu_minn_hcp_acquisition_scheme():
    "Returns DmipyAcquisitionScheme of Wu-Minn HCP project."
    _bvals = np.loadtxt(
        join(_GRADIENT_TABLES_PATH,
             'bvals_hcp_wu_minn.txt')
    ) * 1e6
    _gradient_directions = np.loadtxt(
        join(_GRADIENT_TABLES_PATH,
             'bvecs_hcp_wu_minn.txt')
    )
    _delta = 0.0106
    _Delta = 0.0431
    return acquisition_scheme_from_bvalues(
        _bvals, _gradient_directions, _delta, _Delta)


def duval_cat_spinal_cord_2d_acquisition_scheme():
    "Returns 2D DmipyAcquisitionScheme of cat spinal cord data."
    scheme_name = 'tanguy_cat_spinal_cord/2D_qspace.scheme'
    scheme = np.loadtxt(join(DATA_PATH, scheme_name), skiprows=3)

    bvecs = scheme[:, :3]
    bvecs[np.linalg.norm(bvecs, axis=1) == 0.] = np.r_[1., 0., 0.]
    G = scheme[:, 3]
    Delta = scheme[:, 4]
    delta = scheme[:, 5]
    TE = scheme[:, 6]

    return acquisition_scheme_from_gradient_strengths(
        G, bvecs, delta, Delta, TE)


def duval_cat_spinal_cord_3d_acquisition_scheme():
    "Returns 3D DmipyAcquisitionScheme of cat spinal cord data."
    scheme_name = 'tanguy_cat_spinal_cord/3D_qspace.scheme'
    scheme = np.loadtxt(join(DATA_PATH, scheme_name), skiprows=3)

    bvecs = scheme[:, :3]
    bvecs[np.linalg.norm(bvecs, axis=1) == 0.] = np.r_[1., 0., 0.]
    G = scheme[:, 3]
    Delta = scheme[:, 4]
    delta = scheme[:, 5]
    TE = scheme[:, 6]
    # there's very small differences in TE between shells that we ignore
    TE[:] = 0.0472
    return acquisition_scheme_from_gradient_strengths(
        G, bvecs, delta, Delta, TE, min_b_shell_distance=20e6)


def de_santis_generated_acquisition_scheme():
    "Returns 2D DmipyAcquisitionScheme of de Santis."
    scheme_name = 'de_santis_camino/deSantis_scheme.scheme'
    scheme = np.loadtxt(join(DATA_PATH, scheme_name), skiprows=2)

    bvecs = scheme[:, :3]
    bvecs[np.linalg.norm(bvecs, axis=1) == 0.] = np.r_[1., 0., 0.]
    G = scheme[:, 3]
    Delta = scheme[:, 4]
    delta = scheme[:, 5]
    TE = scheme[:, 6]

    return acquisition_scheme_from_gradient_strengths(
        G, bvecs, delta, Delta, TE)


def isbi2015_white_matter_challenge_scheme():
    "Returns 35-shell ISBI 2015 challenge DmipyAcquisitionScheme."
    isbi_data_path = join(DATA_PATH, 'isbi2015_white_matter_challenge')
    filename = 'isbi_schemefile.txt'
    return acquisition_scheme_from_schemefile(join(isbi_data_path, filename))


def panagiotaki_verdict_acquisition_scheme():
    "Returns acquisition scheme for VERDICT tumor characterization."
    verdict_data_path = join(DATA_PATH, 'panagiotaki_verdict')
    if not os.path.exists(verdict_data_path):
        os.makedirs(verdict_data_path)

    url = "http://camino.cs.ucl.ac.uk/uploads/Tutorials/"
    schemename = "VC_DTIDW.scheme.zip"
    response = urlopen(join(url, schemename))
    with open(join(verdict_data_path, schemename), 'wb') as f:
        f.write(response.read())
    with zipfile.ZipFile(join(verdict_data_path, schemename)) as zip:
        zip.extract("VC_DTIDW.scheme", path=verdict_data_path)

    return acquisition_scheme_from_schemefile(
        join(verdict_data_path, "VC_DTIDW.scheme"), b0_threshold=5e6)
