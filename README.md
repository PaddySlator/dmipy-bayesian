# dmipy-bayesian 
Extension of the dmipy software package to enable Bayesian hierarchical model fitting of microstructural models to diffusion MRI data. The Bayesian hierarchical model assumes a Gaussian prior over a user-defined region (or regions) of interest. This Gaussian prior is estimated from the data, along with the posterior distributions for the microstructure model parameters in each voxel. See (Orton MRM 2014) and (Powell CDMRI 2021) for more details.

The main function is fit_bayes.py - which implements the Bayesian hierarchical model fit. Note that the input/output format is not *exactly* the same as for the dmipy fitting functions - that's something that we'll be working on in future. The example at "bayesian-fitting-HCP-example.ipynb" should be enough to get started.

This code is a work in progress! Please check back regularly for updates, and if you find any bugs please let us know! 

Don't hesistate to get in touch with any questions, suggestions, etc...

Authors:
Elizabeth Powell (e-powell@ucl.ac.uk)
Matteo Battocchio (matteo.battocchio@univr.it)
Chris Parker (christopher.parker@ucl.ac.uk)
Paddy Slator (p.slator@ucl.ac.uk)

Centre for Medical Image Computing, UCL

If you use this code, please consider citing the following papers

Bayesian hierachical model fitting
- **Original implemenation (IVIM model)**: Matthew R. Orton, David J. Collins, Dow-Mu Koh and Martin O. Leach, "Improved intravoxel incoherent motion analysis of diffusion weighted imaging by data driven Bayesian modeling", *Magnetic Resonance in Medicine* 71:411–420, 2014. https://doi.org/10.1002/mrm.24649
- **Implementation for general microstructure model**: Elizabeth Powell, Matteo Battocchio, Christopher S. Parker, and Paddy J. Slator, "Generalised Hierarchical Bayesian Microstructure Modelling for Diffusion MRI", *In: Cetin-Karayumak S. et al. (eds) Computational Diffusion MRI. CDMRI 2021. Lecture Notes in Computer Science, vol 13006. Springer, Cham. https://doi.org/10.1007/978-3-030-87615-9_4*

Underlying dmipy code
- **Primary Reference**: Rutger Fick, Demian Wassermann and Rachid Deriche, "The Dmipy Toolbox: Diffusion MRI Multi-Compartment Modeling and Microstructure Recovery Made Easy", *Frontiers in Neuroinformatics* 13 (2019): 64.
- **Github Repository**: Rutger Fick, Rachid Deriche, & Demian Wassermann. (2019, October 15). *The Dmipy Toolbox: Diffusion MRI Multi-Compartment Modeling and Microstructure Recovery Made Easy (Version 1.0)*. Zenodo. http://doi.org/10.5281/zenodo.3490325

Everything below this line is from the the original dmipy README file...
--------------------------------------------------------------------------------------
[![Build Status](https://travis-ci.org/AthenaEPI/dmipy.svg?branch=master)](https://travis-ci.org/AthenaEPI/dmipy)
[![codecov](https://codecov.io/gh/AthenaEPI/dmipy/branch/master/graph/badge.svg)](https://codecov.io/gh/AthenaEPI/dmipy)
[![Coverage Status](https://coveralls.io/repos/github/AthenaEPI/dmipy/badge.svg)](https://coveralls.io/github/AthenaEPI/dmipy)
[![Documentation Status](https://readthedocs.org/projects/dmipy/badge/?version=latest)](http://dmipy.readthedocs.io/en/latest/?badge=latest)


# Dmipy: Diffusion Microstructure Imaging in Python

The Dmipy software package facilitates the **reproducible estimation of diffusion MRI-based microstructure features**. It does this by taking a completely modular approach to Microstructure Imaging. Using Dmipy you can design, fit, and recover the parameters of any multi-compartment microstructure model in usually less than 10 lines of code. Created models can be used to simulate and fit data for any PGSE-based dMRI acquisition, including single shell, multi-shell, multi-diffusion time and multi-TE acquisition schemes. Dmipy's main features include:

**Complete Freedom in Model Design and Optimization**
- Any combination of tissue models (e.g. Gaussian, Cylinder, Sphere) and axon bundle representation (e.g. orientation-dispersed/diameter-distributed) can be combined into a multi-compartment model.
- Any appropriate model can be orientation-dispersed and/or axon diameter-distributed.
- Any predefined or custom parameter constraints or relations can be imposed.
- Free choice of global optimizer to fit your model to the data (Brute-Force or Stochastic).
- Fit the spherical mean of any multi-compartment model to the spherical mean of the data.
- Generalized multi-compartment constrained spherical deconvolution.

**Human Connectome Project Data Interface**
Dmipy enables you to directly download any HCP subject data using your own credentials.

**Numba-Accelerated, Multi-Core processing**
Dmipy takes heavy advantage of Python's *pathos* multi-core processing and *numba* function compilation.

**Documentation on Tissue and Microstructure Models**
We include documentation and illustrations of all tissue models and parameter distributions, as well as example implementations and results on HCP data for Ball and Stick, Ball and Racket, NODDI-Watson/Bingham, AxCaliber, Spherical Mean models and more.

**Dipy Compatibility**
Dmipy is designed to be complementary for Dipy users. Dipy gradient tables can be directly used in Dmipy models, Dipy models can be used to give initial parameter guesses for Dmipy optimization, and Dmipy models that estimate Fiber Orientation Distributions (FODs) can be visualized and used for tractography in Dipy.

Dmipy allows the user to do Microstructure Imaging research at the highest level, while the package automatically takes care of all the coding architecture that is needed to fit a designed model to a data set. The Dmipy documentation can be found at http://dmipy.readthedocs.io/. If you use Dmipy for your research publications, we kindly request you cite this package at the reference at the bottom of this page.

## Installation
You can install dmipy using pypi by typing in your terminal
- python3 -m pip install dmipy

or you can manually
- clone repository
- python setup.py install

See solutions to [common issues](https://github.com/AthenaEPI/dmipy/blob/master/common_issues.md)
## Dependencies
Recommended to use Anaconda Python distribution.
- numpy >= 1.13
- scipy
- [dipy](http://nipy.org/dipy/)
- [cvxpy](http://www.cvxpy.org/)
- boto (optional for HCP-AWS interface)
- [pathos](https://pypi.python.org/pypi/pathos) (optional for multi-core processing)
- [numba](https://numba.pydata.org/) (optional for faster functions)

## Getting Started
To get a feeling for how to use Dmipy, we provide a few tutorial notebooks:
- [Setting up an acquisition scheme](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_setting_up_acquisition_scheme.ipynb)
- [Simulating and fitting data using a simple Stick model](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_simulating_and_fitting_using_a_simple_model.ipynb)
- [Combining biophysical models into a Microstructure model](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_combining_biophysical_models_into_microstructure_model.ipynb)
- [Creating a dispersed and/or distributed axon bundle representation](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_distributed_model_representations.ipynb)
- [Imposing parameter links and constraints](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_imposing_parameter_links.ipynb)
- [Parameter Cascading: Using a simple model to initialize a complex one](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_parameter_cascading_and_simulating_nd_datasets.ipynb)
- [Generalized Multi-Tissue Modeling](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_generalized_multi_tissue_modeling.ipynb)

## Explanations and Illustrations of Dmipy Contents
### Biophysical Models and Distributions
- [Cylinder Models (Axons, e.g. [Assaf et al. 2004])](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_cylinder_models.ipynb)
- [Sphere Models (Tumor cells, e.g. [Panagiotaki et al. 2014])](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_sphere_models.ipynb)
- [Parameter Distribution Models (Axon Diameter Distribution, e.g. [Assaf et al. 2008])](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_diameter_distributions.ipynb)
- [Gaussian Models (Extra-axonal, e.g. [Behrens et al. 2003])](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_gaussian_models.ipynb)
- Tissue Response Function Models and Estimation (WM/GM/CSF, e.g. [Jeurissen et al. 2014])
- [Spherical Distribution Models (Axon Dispersion, e.g. [Kaden et al. 2007])](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_watson_bingham.ipynb)
- [Spherical Mean of any Compartment Model](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_spherical_mean_models.ipynb)
### Global Multi-Compartment Optimizers
- [Brute Force to Gradient Descent (Brute2Fine)](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_brute_force_optimization.ipynb)
- [Stochastic (MIX) [Farooq et al. 2016]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_stochastic_mix_optimization.ipynb)
### Constrained Spherical Deconvolution Optimizers
- [Generalized Multi-Shell Multi-Compartment CSD [Tournier et al. 2007, Jeurissen et al. 2014]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_generalized_csd_optimizer.ipynb)
## Dmipy implementations of Microstructure Models in Literature
Dmipy uses HCP data to illustrate microstructure model examples. To reproduce these examples, dmipy provides a direct way to download HCP data (using your own AWS credentials) in the [HCP tutorial](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_human_connectome_project_aws.ipynb).
### Single Bundle Models
- [Ball and Stick [Behrens et al. 2003]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_ball_and_stick.ipynb)
- [Ball and Racket [Sotiropoulos et al. 2012]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_ball_and_racket.ipynb)
- [NODDI-Watson [Zhang et al. 2012]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_noddi_watson.ipynb)
- [NODDI-Bingham [Tariq et al. 2016]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_noddi_bingham.ipynb)
- [Multi-Tissue NODDI [Frigo et al. 2020]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_multi_tissue_noddi.ipynb)
- [SMT-NODDI [Cabeen et al. 2019]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_smt_noddi.ipynb)
- [AxCaliber [Assaf et al. 2008]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_axcaliber.ipynb)
- [AxCaliber with Extra-Axonal Diffusion Time-Dependence [De Santis et al. 2016]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_axcaliber_temporal_zeppeline.ipynb)

### Crossing Bundle Models
- [Ball and Sticks [Behrens et al. 2003]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_ball_and_sticks.ipynb)
- [NODDI in Crossings (NODDIx) [Farooq et al. 2016]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_mix_microstructure_imaging_in_crossings.ipynb)

### Tumor Models
- [IVIM [Le Bihan et al. 1988, Gurney-Champion et al. 2018]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_ivim.ipynb)
- [VERDICT [Panagiotaki et al. 2014]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_verdict.ipynb)
### Spherical Mean-Based Models
Any Spherical Mean model can also estimate parametric FODs.
- [Spherical Mean Technique [Kaden et al. 2015]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_spherical_mean_technique.ipynb)
- [Multi-Compartment Microscopic Diffusion Imaging [Kaden et al. 2016]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_multi_compartment_spherical_mean_technique.ipynb)
### Spherical Deconvolution-Based Models
Constrained spherical deconvolution (CSD) models are primarily used for Fiber Orientation Distribution (FOD) estimation. Multi-Tissue CSD models improve FOD estimation by separating WM/GM/CSF signal contributions using multiple tissue response functions.
- [Multi-Shell Multi-Compartment CSD [model-based version of Jeurissen et al. 2014]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_multi_compartment_constrained_spherical_deconvolution.ipynb)
- [Multi-Shell Multi-Tissue CSD with Unsupervised 3-Tissue Response Function Estimation [Jeurissen 2014, Dhollander 2016a]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_multi_tissue_constrained_spherical_deconvolution.ipynb)
- [Single-Shell (1 shell + b0) Multi-Tissue CSD [Dhollander et al. 2016b]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_single_shell_three_tissue_csd.ipynb)

## How to contribute to Dmipy
Dmipy's design is completely modular and can easily be extended with new models, distributions or optimizers. To contribute, view our [contribution guidelines](https://github.com/AthenaEPI/dmipy/blob/master/contribution_guidelines.rst).
## How to cite Dmipy
- **Primary Reference**: Rutger Fick, Demian Wassermann and Rachid Deriche, "The Dmipy Toolbox: Diffusion MRI Multi-Compartment Modeling and Microstructure Recovery Made Easy", *Frontiers in Neuroinformatics* 13 (2019): 64.
- **Github Repository**: Rutger Fick, Rachid Deriche, & Demian Wassermann. (2019, October 15). *The Dmipy Toolbox: Diffusion MRI Multi-Compartment Modeling and Microstructure Recovery Made Easy (Version 1.0)*. Zenodo. http://doi.org/10.5281/zenodo.3490325

[Package Acknowledgements](https://github.com/AthenaEPI/dmipy/blob/master/Acknowledgements.md)
