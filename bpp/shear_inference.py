"""
Hierarchical inference of weak lensing shear

Inputs: posterior galaxy model parameter samples for a selection of galaxies

Outputs: posterior samples of lensing shear (and eventually, magnification)
(assumes a constant shear for all galaxies for now)

Hierarchical model (simplest toy version):

                sigma_g
                  |
                  |
                  V
               (g1,g2)   sigma_e
                  |         |
                  |         |
                  V         V
                e_obs <-- e_int
                  |
                  |
                  V      
sigma_noise --> pixel data

We are given samples of `e_obs` from previous processing steps. Our goal is to 
infer the shear parameters `(g1,g2)`, assumed constant over the sky through 
importance sampling given an asserted value for the standard deviation, 
`sigma_g`, in a zero-mean Gaussian prior for each shear component. Along the 
way, we also infer the standard deviation, `sigma_e`, of the unlensed galaxy 
ellipticity component distribution, also assumed to be a Gaussian. 

To allow for generalization of this model in the future, we adopt the following
naming scheme in this script,

    gal_params: e_obs (and other parameters such as flux in the future)
    gal_params_int: e_int (and other parameters in the future)
    gal_dist: sigma_e (and other distribution parameterizations in the future)
    lens_params: g1, g2 (and convergence, kappa, in the future)
    lens_dist: sigma_g (Gaussian process distribution in the future)
"""
import jax.numpy as jnp


def unshear(e, g, weak_shear=True):
    """Unshear complex ellipticity e given complex reduced shear g

    Args:
        e (complex): ellipticity
        g (comples): reduced shear
    
    Returns:
        e_int (complex): unsheared ellipticity
    """
    d = e - g
    if weak_shear:
        return d
    else: # do not assume weak shear
        return d / (1.0 - e * g.conj()) # Bartelmann & Schneider (4.12)


def prior_gaussian_shear(lens_dist):
    """A Gaussian functional form for the final prior over g1, g2

    This is a function of lens_dist through sigma_g. 
    Returns P(g1, g2 | sigma_g) as a Gaussian.
    """
    sigma_g_sq = lens_dist[0]**2
    lognorm = jnp.log(2*jnp.pi*sigma_g_sq)
    def shear_prior(g1,g2):
        return -0.5 * (g1**2 + g2**2) / sigma_g_sq - lognorm
    return shear_prior

def prior_gaussian_ellipticity(gal_dist):
    """A Gaussian functional form for the final prior over e1,e2

    This is a function of gal_dist through sigma_e.
    Returns P(e_int_1, e_int_2 | sigma_e)

    Args:
        gal_dist (list): Hyperparameters for the intrinsic galaxy image 
            properties
    """
    sigma_e_sq = gal_dist[0]**2
    lognorm = jnp.log(2.0*jnp.pi * sigma_e_sq)
    def ellipticity_prior(gal_params, lens_params):
        g = lens_params[0] + 1j*lens_params[1]
        e = gal_params[0] + 1j*gal_params[1]
        e_int = unshear(e, g)
        return -0.5 * (e_int*e_int.conj() / sigma_e_sq) - lognorm
    return ellipticity_prior


def ln_likelihood(gal_params,lens_params, gal_dist, prior_forms, interim_priors):
    """
    Evaluate the natural logarithm of the likelihood of galaxy images given 
    lensing shear parameters

    This is evaluated as a 'pseudo-marginal' likelihood where the paramteres
    of individual galaxy images are approximately marginalized via Monte Carlo
    simulation.

    Args:
        gal_params: Array of posterior samples for galaxy image model parameters.
        gal_dist: List of galaxy distribution hyperparameters (sigma_e).
        prior_forms: List of functionals of hyperparameters `gal_dist` 
            which each yield a probability density function of galaxy 
            parameters (e1, e2).
        interim_priors: Array of interim prior values for each of the input 
            posterior samples (i.e., what prior was used for the galaxy image
            parameters in the previous processing step?)
    """
    # Construct prior functions for given shear parameters
    priors = [p(gal_dist) for p in prior_forms]
    # Evaluate the shear-dependent priors on the galaxy image parameter samples
    # This is: Pr(e_obs_1, e_obs_2 | g1, g2, sigma_e)
    probs = [p(gal_params, lens_params) for p in priors]
    # Sum importance sampling weights for each sample for a given galaxy to
    # evaluate the marginal likelihood for that galaxy. 
    # Use `mean` instead of `sum` to keep the numerical value smaller - we 
    # never care about the absolute normalization anyway.
    wts = jnp.mean(probs / interim_priors, axis=1)
    # The marginal likelihood for all galaxies is the product of marginal
    # likelihoods for each individual galaxy (or the sum of the 
    # log-likelihoods)
    out = jnp.log(wts).sum()
    return out


def ln_shear_posterior(gal_params, lens_params, gal_dist, lens_dist, prior_forms, interim_priors):
    """
    Evaluate the natural logarithm of the marginal posterior on shear parameters
    """
    lnlike = ln_likelihood(gal_params, lens_params, gal_dist, prior_forms, interim_priors)
    lnprior = jnp.log(prior_gaussian_shear(lens_dist)(lens_params))
    return lnlike + lnprior


def sample_posterior():
    """
    Run MCMC to generate samples from the shear posterior
    """
    pass

def main():
    # Load input samples

    # Sample from the shear posterior

    # Save output shear samples
    pass


if __name__ == "__main__":
    main()
