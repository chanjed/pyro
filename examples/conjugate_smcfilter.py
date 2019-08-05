import argparse
import logging

import torch
from torch.testing import assert_allclose

import pyro
import pyro.distributions as dist
from pyro.infer import SMCFilter

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)

"""
This file demonstrates how to use the SMCFilter algorithm with
a conjugate model.

"""
# class StateSpaceModel:

#     def __init__(self, prior_alpha, prior_beta, input_size, state_size, obs_size):
#         self.prior_alpha = prior_alpha
#         self.prior_beta = prior_beta
#         self.input_size = input_size
#         self.state_size = state_size
#         self.obs_size = obs_size

#     def init(self, initial):
#         assert(initial.shape == self.state_size)
#         self.t = 0
#         self.z = initial
#         self.y = None

#         # Draw globals
#         # Draw for each state dimension
#         self.prec_x = pyro.sample("prec_x", dist.Gamma(self.prior_alpha, self.prior_beta).expand([self.state_size])) # batch_shape == (state_size), event_shape == ()
#         self.W_x = pyro.sample("W_x", dist.Normal(torch.zeros(self.state_size), # batch_shape = (input_size, state_size), event_shape=()
#                                                   torch.sqrt(1./(self.prec_x*self.input_size))).expand([self.input_size, self.state_size]))

#         self.prec_z = pyro.sample("prec_z", dist.Gamma(self.prior_alpha, self.prior_beta).expand([self.state_size])) # batch_shape == (state_size), event_shape == ()
#         self.W_z = pyro.sample("W_z", dist.Normal(torch.zeros(self.state_size), # batch_shape = (state_size, state_size), event_shape=()
#                                                   torch.sqrt(1./(self.prec_z*self.state_size))).expand([self.state_size, self.state_size]))

#         self.prec_y = pyro.sample("prec_y", dist.Gamma(self.prior_alpha, self.prior_beta).expand([self.obs_size])) # batch_shape == (obs_size), event_shape == ()
#         self.W_y = pyro.sample("W_y", dist.Normal(torch.zeros(self.obs_size), # batch_shape = (state_size, obs_size), event_shape=()
#                                                   torch.sqrt(1./(self.prec_x*self.state_size))).expand([self.state_size, self.obs_size]))
#         # TODO: can implement allowing for multiple alpha/betas for each layer.
#         # TODO: do we want these guys in the batch or event bit

#     def step(self, x=None, y=None):
#         self.t += 1
#         # TODO: figure out the correct variance here
#         self.z = pyro.sample("z_{}".format(self.t),
#                              dist.Normal(self.z.matmul(self.W_z) + self.x.matmul(self.W_x), 
#                                          torch.sqrt(1./self.prec_x + 1./self.prec_z))) #.to_event(1))
#         self.y = pyro.sample("y_{}".format(self.t),
#                              dist.Normal(self.z.matmul(self.W_y), torch.sqrt(1./self.prec_y)),
#                              obs=y)

#         return self.z, self.y


class ConjugateStateSpaceModel:

    def __init__(self, prior_xz, prior_zz, prior_zy, latent_dim, obs_dim, has_inputs=True):
        self.has_inputs = has_inputs
        self.prior_xz = prior_xz
        self.prior_zz = prior_zz
        self.prior_zy = prior_zy

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

    def init(self, initial):
        # initial must have particles as the first dimension otherwise it will break.
        self.t = 0
        self.z = initial
        self.y = None

        if self.has_inputs:
            self.summary_xz = dist.NIGNormalRegressionSummary(*self.prior_xz)
        assert(self.prior_zz[0].shape[-2:] == (self.latent_dim, self.latent_dim))
        assert(self.prior_zz[1].shape[-3:] == (self.latent_dim, self.latent_dim, self.latent_dim))
        assert(self.prior_zz[2].shape[-1] == (self.latent_dim))
        assert(self.prior_zz[3].shape[-1] == (self.latent_dim))
        self.summary_zz = dist.NIGNormalRegressionSummary(*self.prior_zz)
        assert(self.prior_zy[0].shape[-2:] == (self.obs_dim, self.latent_dim))
        assert(self.prior_zy[1].shape[-3:] == (self.obs_dim, self.latent_dim, self.latent_dim))
        assert(self.prior_zy[2].shape[-1] == (self.obs_dim))
        assert(self.prior_zy[3].shape[-1] == (self.obs_dim))
        self.summary_zy = dist.NIGNormalRegressionSummary(*self.prior_zy)

    def step(self, x=None, y=None):
        print("STEPPING")
        print("TIMESTEP: ", self.t)
        # TODO: Figure out if non-batching over summary update works
        old_z = self.z
        if self.has_inputs and x is not None:
            raise RuntimeError("Not yet implemented")
            z_x = pyro.sample("z_x_{}".format(self.t),
                              self._get_posterior_predictive(self.summary_xz, x))
            self.summary_xz.update(z_x[..., None, :], x[..., None, :]) # (particles x 1 x features_dim) for x and (particles x 1 x obs_dim) for z
        else:
            z_x = 0.

        z_z = pyro.sample("z_z_{}".format(self.t),
                          self._get_posterior_predictive(self.summary_zz, old_z))
        self.z = z_x + z_z # (particles x z_dim)
        self.summary_zz.update(self.z[..., None, :], old_z[..., None, :])

        self.y = pyro.sample("y_{}".format(self.t), 
                             self._get_posterior_predictive(self.summary_zy, self.z),
                             obs=y)
        self.summary_zy.update(self.y[..., None, :], self.z[..., None, :])

        self.t += 1

    def forecast(self, x=None):
        print("FORECASTING")
        if self.has_inputs and x is not None:
            raise RuntimeError("Not yet verified")
            z_x = pyro.sample("z_x_{}".format(self.t),
                              self._get_posterior_predictive(self.summary_xz, x))
        else:
            z_x = 0.
        z_z = pyro.sample("z_z_{}".format(self.t),
                          self._get_posterior_predictive(self.summary_zz, self.z))
        z = z_x + z_z

        y = pyro.sample("y_{}".format(self.t), 
                        self._get_posterior_predictive(self.summary_zy, z))


    @staticmethod
    def _get_posterior_predictive(summary, features):
        a = summary.covariance.inverse()
        b = summary.covariance.inverse().matmul(summary.mean.unsqueeze(-1)).squeeze(-1)
       
        _features = features[..., None, None, :]
        df = 2. * summary.shape
        loc = _features.matmul(summary.precision.inverse()).matmul(summary.precision_times_mean.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        term1 = (summary.reparametrized_rate - 0.5*summary.precision_times_mean.unsqueeze(-2).matmul(summary.precision.inverse())
                 .matmul(summary.precision_times_mean.unsqueeze(-1)).squeeze(-1).squeeze(-1))/summary.shape
        term1_correct = summary.rate/summary.shape
        # term2_correct = 1. + _features.matmul(summary.covariance).matmul(_features.transpose(-2,-1)).squeeze(-1).squeeze(-1)
        term2 = 1. + _features.matmul(summary.precision.inverse()).matmul(_features.transpose(-2,-1)).squeeze(-1).squeeze(-1)
        scalesquared = term1 * term2 
        # scalesquared_correct = term1_correct * term2_correct
        assert(torch.all(term1_correct > 0.))
        assert(torch.all(term1 > 0.))
        assert(torch.all(term2 > 0.))
        assert(torch.all(scalesquared > 0.))
        
        return dist.StudentT(df, loc, torch.sqrt(scalesquared)) # (summary.obs_dim)



class ConjugateStateSpaceModel_Guide:

    def __init__(self, model, has_inputs=True):
        pass

    def init(self, initial):
        return None

    def step(self, x=None, y=None):
        return None


def generate_data(args):
    ys = []
    for t in range(args.num_timesteps):
        ys.append(torch.sin(torch.tensor([5.*t + 3.])) + 0.5 + torch.distributions.Normal(100., 3.).sample())

    return ys

def get_forecast(values, logweights):
    print("Values: ", values)
    print("Logweights: ", logweights)
    return {name: dist.Empirical(value, logweights)
                    for name, value in values.items()
                    if name.startswith('y')}


def main(args):
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    prior_zz = (torch.zeros((args.latent_dim, args.latent_dim)), 
                torch.eye(args.latent_dim).expand((args.latent_dim, args.latent_dim, args.latent_dim)),
                torch.tensor(3.).expand(args.latent_dim), 
                torch.tensor(1.).expand(args.latent_dim))
    prior_zy = (torch.zeros((1, args.latent_dim)), 
                torch.eye(args.latent_dim).expand((1, args.latent_dim, args.latent_dim)) + 0.5, 
                torch.tensor([3.]), 
                torch.tensor([1.]))
    model = ConjugateStateSpaceModel(None, prior_zz, prior_zy, args.latent_dim, 1, has_inputs=args.has_inputs)
    guide = ConjugateStateSpaceModel_Guide(model, has_inputs=args.has_inputs)

    smc = SMCFilter(model, guide, num_particles=args.num_particles, max_plate_nesting=1)

    logging.info('Generating data')
    ys = generate_data(args)

    logging.info('Performing inference')
    forecasts = {}
    smc.init(initial=torch.zeros(args.latent_dim).expand((args.num_particles, args.latent_dim)))
    for y in ys:
        forecasts.update(get_forecast(*smc.forecast()))
        smc.step(y=y)

    # logging.info('Marginals')
    # empirical = smc.get_empirical()
    # for t in range(args.num_timesteps):
    #     z = empirical["z_z_{}".format(t)]
    #     if args.has_inputs:
    #         raise RuntimeError("Not yet implemented")
    #         z += empirical["z_x_{}".format(t)]
    #     logging.info("{}\t{}\t{}\t{}".format(t, zs[t], z.mean, z.variance))

    logging.info("=============================")
    logging.info('Forecasts\tTruth\tPred_Mean\tPred_Variance')
    for t in range(args.num_timesteps):
        y_emp = forecasts["y_{}".format(t)]
        logging.info("{}\t{}\t{}\t{}".format(t, ys[t], y_emp.mean, y_emp.variance))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conjugate State Space Model w/ SMC Filtering Inference")
    parser.add_argument("-n", "--num-timesteps", default=50, type=int)
    parser.add_argument("-p", "--num-particles", default=100, type=int)
    parser.add_argument("-d", "--latent-dim", default=5, type=int)
    parser.add_argument("--has-inputs", default=False, type=bool)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)
