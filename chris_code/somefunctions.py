##--- Some functions I intend to complete for the new version
def AngleByDistance(theta_phi_current,distances):
    # problem:
    # uniformly spaced polar angles don't map to uniformly spaced orientations
    # this can prevent the angle priors and acceptance probabilities from working as intended
    # to fix this we should sample the angle priors from a distance (on surface of unit sphere) distribution instead
    # this function:
    # calculates 'theta_phi_new' for distance moved on surface of unit sphere, given current orientation 'theta_phi_current'
    # 'distances' is vector of length 2 with the distances in the [polar (longitude), azimuth (lattitude)] directions
    return theta_phi_new

def DistanceByAngle(theta_phi_current,theta_phi_change):
    # opposite function to newAngleByDistance()
    # might be useful, not sure
    return distances

##--- Some functions I worked on during the Hackathon
def accept_fun(theta_prop, theta_current, y, N, lambda_next, nu_next):
    # only used without orientation parameters!!
    g_prop = ballstick.simulate_signal(acq_scheme, theta_prop)
    prop_Lik = (p.dot(y, y) - np.dot(y, g_prop) ** 2 / np.dot(g_prop, g_prop)) ** (N / 2)
    prop_Pri = (1 / np.linalg.det(2 * np.pi * lambda_next) ** (1 / 2)) * np.exp(
        (-1 / 2) * np.multi_dot(theta_proposed - nu_next, np.linalg.inv(lambda_next),
                                theta_proposed - nu_next))  # eq. 7 Orten
    prop_Posterior = prop_Lik * prop_Pri  # posterior distribution for a voxel!
    g_current = ballstick.simulate_signal(acq_scheme, theta_current)
    current_Lik = (p.dot(y, y) - np.dot(y, g_current) ** 2 / np.dot(g_current, g_current)) ** (N / 2)
    current_Pri = (1 / np.linalg.det(2 * np.pi * lambda_next) ** (1 / 2)) * np.exp(
        (-1 / 2) * np.multi_dot(theta_current - nu_next, np.linalg.inv(lambda_next), theta_current - nu_next))
    current_Posterior = current_Lik * current_Pri
    accept_prob = min(1, prop_Posterior / current_Posterior)
    return accept_prob


def accept_fun_withorient(theta_prop, theta_current, y, N, lambda_next, nu_next, constraints_mu_one, constraints_mu_two):
    # for calculating acceptance probability with orientation parameters

    # hyper-priors for non-orientation params
    lambda_next_nonorient = lambda_next[2:, 2:]
    nu_next_nonorient = nu_next[2:]

    # calculating a non-zero posterior if within constraints
    mu_one_proposed = theta_proposed[0]
    mu_two_proposed = theta_proposed[1]
    if mu_one_proposed < constraints_mu_one[0] or mu_one_proposed > constraints_mu_one[1]:
        prop_Posterior = 0
    elif mu_two_proposed < constraints_mu_two[0] or mu_two_proposed > constraints_mu_two[1]:
        prop_Posterior = 0
    else
        g_prop = ballstick.simulate_signal(acq_scheme, theta_prop)
        prop_Lik = (p.dot(y, y) - np.dot(y, g_prop) ** 2 / np.dot(g_prop, g_prop)) ** (N / 2)
        prop_Pri_orient = 1  # joint prior on both orientation parameters - some constant from a uniform distribution i.e. effectively no prior information included (see end pg. 3 Paddy, middle pg. 2)
        prop_Pri_nonorient = (1 / np.linalg.det(2 * np.pi * lambda_next_nonorient) ** (1 / 2)) * np.exp(
            (-1 / 2) * np.multi_dot(theta_proposed_nonorient - nu_next_nonorient, np.linalg.inv(lambda_next_nonorient),
                                    theta_proposed_nonorient - nu_next_nonorient))  # joint prior probability of non-orientation parameters (eq. 7 Orten)
        prop_Pri = prop_Pri_nonorient * prop_Pri_orient  # joint prior distribution over all parameters for a voxel (see middle pg. 2 Paddy)
        prop_Posterior = prop_Lik * prop_Pri  # joint posterior distribution (over parameters) for a voxel

    # assume current orientation params are always within the constraints
    g_curr = ballstick.simulate_signal(acq_scheme, theta_current)
    curr_Lik = (p.dot(y, y) - np.dot(y, g_curr) ** 2 / np.dot(g_curr, g_curr)) ** (N / 2)
    curr_Pri_orient = 1  # same as above
    curr_Pri_nonorient = (1 / np.linalg.det(2 * np.pi * lambda_next_nonorient) ** (1 / 2)) * np.exp(
        (-1 / 2) * np.multi_dot(theta_curr_nonorient - nu_next_nonorient, np.linalg.inv(lambda_next_nonorient),
                                theta_curr_nonorient - nu_next_nonorient))
    curr_Pri = curr_Pri_nonorient * curr_Pri_orient
    curr_Posterior = curr_Lik * curr_Pri

    accept_prob = min(1, prop_Posterior / curr_Posterior)
    return accept_prob
