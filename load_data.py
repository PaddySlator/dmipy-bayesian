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

