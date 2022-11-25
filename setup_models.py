# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes

# ball stick and spherical mean ball-stick model
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core import modeling_framework
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel, MultiCompartmentModel
from dmipy.distributions.distribute_models import SD1WatsonDistributed, BundleModel
# from dmipy.utils import spherical_mean


# ball & stick
def _ballstick():
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    ballstick = MultiCompartmentModel(models=[stick, ball])
    return ballstick


# noddi
def _noddi():
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    zeppelin = gaussian_models.G2Zeppelin()

    watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
    # set tortuous parameters
    watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par',
                                                   'partial_volume_0')
    watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)
    # put the model together
    noddi = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
    # fix free water diffusivity
    noddi.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

    return noddi


# spherical mean technique
def _smt():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    bundle = BundleModel([stick, zeppelin])
    bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
    bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    smt = modeling_framework.MultiCompartmentSphericalMeanModel(models=[bundle])

    return smt


# ball, stick, zeppelin; spherical mean (ecap currently can't fit, memory errors, 22/07/22)
def _bsz_sm():
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    bsz = MultiCompartmentModel(models=[stick, ball, zeppelin])

    # spherical mean model with constraints
    bsz_sm = MultiCompartmentSphericalMeanModel(models=[stick, ball, zeppelin])
    bsz_sm.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    bsz_sm.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

    return bsz


# smt-noddi
def _smt_noddi():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    ball = gaussian_models.G1Ball()

    bundle = BundleModel([stick, zeppelin])
    bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
    bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

    smt_noddi = modeling_framework.MultiCompartmentSphericalMeanModel(models=[bundle, ball])
    smt_noddi.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

    return smt_noddi


# select model from ones already defined (above)
def _setup_model(model_to_fit):
    if model_to_fit == 'ballstick':
        model = _ballstick()
    elif model_to_fit == 'noddi':
        model = _noddi()
    elif model_to_fit == 'smt':
        model = _smt()
    elif model_to_fit == 'bsz_sm':
        model = _bsz_sm()
    elif model_to_fit == 'smt_noddi':
        model = _smt_noddi()

    return model
