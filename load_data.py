
##-- Building scheme file from built-in dmipy HCP b-values and b-vector .txt file

# load the necessary modules
from dmipy.core import modeling_framework
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from os.path import join
import numpy as np

# the HCP acquisition parameters are saved in the following toolbox path:
acquisition_path = modeling_framework.GRADIENT_TABLES_PATH

# we can then load the parameters themselves and convert them to SI units:
bvalues = np.loadtxt(join(acquisition_path, 'bvals_hcp_wu_minn.txt'))  # given in s/mm^2
bvalues_SI = bvalues * 1e6  # now given in SI units as s/m^2
gradient_directions = np.loadtxt(join(acquisition_path, 'bvecs_hcp_wu_minn.txt'))  # on the unit sphere

# The delta and Delta times we know from the HCP documentation in seconds
delta = 0.0106  
Delta = 0.0431 

# The acquisition scheme used in the toolbox is then created as follows:
acq_scheme = acquisition_scheme_from_bvalues(bvalues_SI, gradient_directions, delta, Delta)


##-- Reading the DWI nifti image
from dipy.io.image import load_nifti
image_path='/Users/christopherparker/Documents/Projects/bayes-dmipy/CD-MRI Paper/TestRetestData/103818_1/data.nii.gz'
data, affine, img = load_nifti(image_path, return_img=True)

# plotting an axial slice
import matplotlib.pyplot as plt
axial_middle = data.shape[2] // 2
plt.figure('Axial slice')
plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
plt.show()

