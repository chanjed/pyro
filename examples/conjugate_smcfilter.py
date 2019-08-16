import argparse
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch.testing import assert_allclose

import pyro
import pyro.distributions as dist
from pyro.infer import SMCFilter, FastOnlineSMCFilter

import csv
import os

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)

"""
This file demonstrates how to use the SMCFilter algorithm with
a conjugate model.

"""

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
        self.z = pyro.sample("zz_-1", dist.Delta(initial)) # Hack to ease downdating otherwise history doesnt know about initial
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
            z_x = pyro.sample("zx_{}".format(self.t),
                              self._get_posterior_predictive(self.summary_xz, x))
            self.summary_xz.update(z_x[..., None, :], x[..., None, :]) # (particles x 1 x features_dim) for x and (particles x 1 x obs_dim) for z
        else:
            z_x = 0.

        z_z = pyro.sample("zz_{}".format(self.t),
                          self._get_posterior_predictive(self.summary_zz, old_z))
        # z_z = pyro.sample("zz_{}".format(self.t),
        #                   dist.Delta(torch.arange(1.*self.latent_dim)))
        self.z = z_x + z_z # (particles x z_dim)
        old_rate = self.summary_zz.rate
        self.summary_zz.update(self.z[..., None, :], old_z[..., None, :])

        print("Min Weights: ", torch.abs(self.summary_zz.mean).min())
        print("Max Weights: ", torch.abs(self.summary_zz.mean).max())
        print("Min Cov Val: ", torch.abs(self.summary_zz.covariance).min())
        print("Max Cov Val: ", torch.abs(self.summary_zz.covariance).max())
        print("Min Prec Val: ", torch.abs(self.summary_zz.precision).min())
        print("Max Prec Val: ", torch.abs(self.summary_zz.precision).max())
        print("Min Cov Diag: ", torch.abs(self.summary_zz.covariance.diag_embed()).min())
        print("Max Cov Diag: ", torch.abs(self.summary_zz.covariance.diag_embed()).max())
        print("Min Prec Diag: ", torch.abs(self.summary_zz.precision.diag_embed()).min())
        print("Max Prec Diag: ", torch.abs(self.summary_zz.precision.diag_embed()).max())
        # print("Min Cov Eig: ", min([self.summary_zz.covariance[i,0,:,:].symeig().eigenvalues.min() for i in range(1000)]))
        # print("Max Cov Eig: ", max([self.summary_zz.covariance[i,0,:,:].symeig().eigenvalues.max() for i in range(1000)]))
        # print("Min Prec Eig: ", min([self.summary_zz.precision[i,0,:,:].symeig().eigenvalues.min() for i in range(1000)]))
        # print("Max Prec Eig: ", max([self.summary_zz.precision[i,0,:,:].symeig().eigenvalues.max() for i in range(1000)]))
        print("Min diff rate: ", (self.summary_zz.rate - old_rate).min())
        assert(torch.all(self.summary_zz.rate + 1.e-6 >= old_rate))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.y = pyro.sample("y_{}".format(self.t), 
                             self._get_posterior_predictive(self.summary_zy, self.z),
                             obs=y)
        old_rate = self.summary_zy.rate
        self.summary_zy.update(self.y[..., None, :], self.z[..., None, :])

        print("Min Weights: ", torch.abs(self.summary_zy.mean).min())
        print("Max Weights: ", torch.abs(self.summary_zy.mean).max())
        print("Min Cov Val: ", torch.abs(self.summary_zy.covariance).min())
        print("Max Cov Val: ", torch.abs(self.summary_zy.covariance).max())
        print("Min Prec Val: ", torch.abs(self.summary_zy.precision).min())
        print("Max Prec Val: ", torch.abs(self.summary_zy.precision).max())
        print("Min Cov Diag: ", torch.abs(self.summary_zy.covariance.diag_embed()).min())
        print("Max Cov Diag: ", torch.abs(self.summary_zy.covariance.diag_embed()).max())
        print("Min Prec Diag: ", torch.abs(self.summary_zy.precision.diag_embed()).min())
        print("Max Prec Diag: ", torch.abs(self.summary_zy.precision.diag_embed()).max())
        print("Min Cov Eig: ", min([self.summary_zy.covariance[i,0,:,:].symeig().eigenvalues.min() for i in range(10)]))
        print("Max Cov Eig: ", max([self.summary_zy.covariance[i,0,:,:].symeig().eigenvalues.max() for i in range(10)]))
        print("Min Prec Eig: ", min([self.summary_zy.precision[i,0,:,:].symeig().eigenvalues.min() for i in range(10)]))
        print("Max Prec Eig: ", max([self.summary_zy.precision[i,0,:,:].symeig().eigenvalues.max() for i in range(10)]))
        print("Min diff rate: ", (self.summary_zy.rate - old_rate).min())
        print("Min reparam rate: ", self.summary_zy.reparametrized_rate.min())
        print("Max reparam rate: ", self.summary_zy.reparametrized_rate.max())
        print("Min rate: ", self.summary_zy.rate.min())
        print("Max rate: ", self.summary_zy.rate.max())
        assert(torch.all(self.summary_zy.rate + 1.e-6 >= old_rate))
        print("=================================")
        self.t += 1

    def resample(self, index):
        if self.has_inputs:
            self.summary_xz = self.summary_xz[index]
        print("RESAMPLING...")
        self.summary_zz = self.summary_zz[index]
        print("zy")
        self.summary_zy = self.summary_zy[index]
        self.z = self.z[index]

    def forecast(self, x=None):
        if self.has_inputs and x is not None:
            z_x = pyro.sample("zx_{}".format(self.t),
                              self._get_posterior_predictive(self.summary_xz, x))
        else:
            z_x = 0.
        z_z = pyro.sample("zz_{}".format(self.t),
                          self._get_posterior_predictive(self.summary_zz, self.z))
        z = z_x + z_z

        y = pyro.sample("y_{}".format(self.t), 
                        self._get_posterior_predictive(self.summary_zy, z))

    def downdate(self, history):
        if self.has_inputs:
            raise Exception("Not yet implemented..history needs to keep track of x's too.")
        zs = history.get_downdate_values("zz")
        if zs is not None:
            print("VALUES:", history._values["zz"][:,0,...])
            print("Summary mean:", self.summary_zz.mean)
            # print("OBS:", history._obs["y"].shape)
            # print("LOG PROBS:", history._log_probs["y"].shape)
            # print("INITTIAL:", history._initial["zz"].shape)
            # print("obs_to_dd", zs)
            # print("features_to_dd", history.get_downdate_values("zz", -1))
            # print("num_forget", history.get_num_forget())
            print("zs shape", zs)
            print("features_shape", history.get_downdate_values("zz", -1))
            # print("forget shape", history.get_num_forget().shape)
            old_rate = self.summary_zz.reparametrized_rate
            self.summary_zz.downdate(zs, history.get_downdate_values("zz", -1), history.get_num_forget())
            # print("Min forget: ", history.get_num_forget().min())
            # print("Max forget: ", history.get_num_forget().max())
            # print("Min forget: ", history.get_num_forget().argmin())
            # print("Max forget: ", history.get_num_forget().argmax())
            # print("Mean: ", self.summary_zz.mean)
            # print("Cov: ", self.summary_zz.covariance)
            # print("Shape: ", self.summary_zz.shape)
            print("Downdated reparam'd rate: ", old_rate - self.summary_zz.reparametrized_rate)
            old_rate = self.summary_zy.rate
            self.summary_zy.downdate(history.get_downdate_obs("y"), zs, history.get_num_forget())
            print("Downdated rate: ", old_rate - self.summary_zy.rate)

    @staticmethod
    def _get_posterior_predictive(summary, features):
        _features = features[..., None, None, :]
        df = 2. * summary.shape
        loc = _features.matmul(summary.precision.inverse()).matmul(summary.precision_times_mean.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        term1 = (summary.reparametrized_rate - 0.5*summary.precision_times_mean.unsqueeze(-2).matmul(summary.precision.inverse())
                 .matmul(summary.precision_times_mean.unsqueeze(-1)).squeeze(-1).squeeze(-1))/summary.shape
        term2 = 1. + _features.matmul(summary.precision.inverse()).matmul(_features.transpose(-2,-1)).squeeze(-1).squeeze(-1)
        scalesquared = term1 * term2
        assert(torch.all(term1 > 0.))
        assert(torch.all(term2 > 0.))
        assert(torch.all(scalesquared > 0.))
        
        return dist.StudentT(df, loc, torch.sqrt(scalesquared)) # (summary.obs_dim)



class ConjugateStateSpaceModel_Guide:

    def __init__(self, model, has_inputs=True):
        pass

    def init(self, initial):
        pass

    def step(self, x=None, y=None):
        pass


def generate_data(args):
    denoised_ys = []
    ys = []
    for t in range(args.num_timesteps):
        # clean_y = 10.*torch.sin(torch.tensor([0.2*t + 3.])) + 0.05*t
        if t < 500:
            clean_y = 10.*torch.sin(torch.tensor([0.2*t + 3.])) + 0.05*t
        else:
            clean_y = 20.*torch.sin(torch.tensor([0.2*t + 8.])) + 0.1*(t-500.) + 0.05*500.
        denoised_ys.append(clean_y)
        ys.append(clean_y + torch.distributions.Normal(0., 2.0).sample())

    return ys, denoised_ys

def get_nile_data(args):
    """STEP 1: Read in the nile data from nile.txt"""
    raw_data = []
    with open("/Users/jeffrey.chan/Documents/bocpdms/Data/nile.txt") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            raw_data += row

    raw_data_float = []
    for entry in raw_data:
        raw_data_float.append(float(entry))
    raw_data = raw_data_float

    """STEP 2: Format the nile data so that it can be processed with a Detector
    object and instantiations of ProbabilityModel subclasses"""
    T = int(len(raw_data) / 2)
    data = np.array(raw_data).reshape(T, 2)
    dates = data[:, 0]
    river_height = data[:, 1]
    mean, variance = np.mean(river_height), np.var(river_height)

    """STEP 3: Standardize in order to be able to compare with GP-approaches"""
    standardised_river_height = (river_height - mean) / np.sqrt(variance)

    return [torch.tensor([i]) for i in standardised_river_height]


def plot(true_xs, true_ys, predictions, nile=False):
    # xs: (num_timesteps)
    # ys: (num_timesteps)
    # predictions: num_timesteps x 3

    # Grab predictions
    # plot results
    fig, ax = plt.subplots(1, 1)

    ax.plot(true_xs, true_ys, 'kx', label='Observations')
    if not nile:
        # dom = np.arange(0, true_xs[-1], 0.1)
        dom1 = np.arange(0, 500., 0.1)
        ran1 = 10.*np.sin(0.2*dom1 + 3.) + 0.05*dom1
        dom2 = np.arange(500., true_xs[-1], 0.1)
        ran2 = 20.*np.sin(0.2*dom2 + 8.) + 0.1*(dom2-500.) + 0.05*500.
        dom = np.hstack([dom1, dom2])
        ran = np.hstack([ran1, ran2])
        ax.plot(dom, ran, 'red', ls='solid', lw=2.0, label="Truth")

    # plot 90% confidence level of predictions
    ax.fill_between(true_xs.flatten(), predictions[:, 1].flatten(), predictions[:, 2].flatten(), color='lightblue')
    # plot mean prediction
    ax.plot(true_xs, predictions[:, 0], 'blue', ls='solid', lw=2.0, label='Mean Prediction')
    # TODO: allow trajectoriees to be ploted

    ax.set(xlabel="Time", ylabel="Y", title="Mean predictions with 90% CI")

    plt.plot()
    plt.show()
    plt.close()


def get_forecast(values, logweights):
    return {name: dist.Empirical(value, logweights)
                    for name, value in values.items()
                    if name.startswith('y')}

def extract_mean_nll(log_probs):
    assert(len(log_probs) == 1)
    for name, log_prob in log_probs.items():
        log_max = log_prob.max(0).values
        # log[(1/N)\sum exp(log(p_i) - log_max)]
        mean_nll = float(torch.mean(torch.logsumexp(log_prob - log_max, 0) - np.log(log_prob.size(0)) + log_max))

    return mean_nll


def main(args):
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    # Define the prior, the model object, and inference object
    prior_zz = (torch.zeros((args.latent_dim, args.latent_dim)), 
                (torch.eye(args.latent_dim).expand((args.latent_dim, args.latent_dim, args.latent_dim)) + 0.5),
                #5.*
                torch.tensor(3.).expand(args.latent_dim), 
                0.01*
                torch.tensor(1.).expand(args.latent_dim))
    prior_zy = (torch.zeros((1, args.latent_dim)), 
                (torch.eye(args.latent_dim).expand((1, args.latent_dim, args.latent_dim)) + 0.5), 
                50.*
                torch.tensor([3.]), 
                1.*
                torch.tensor([1.]))
    model = ConjugateStateSpaceModel(None, prior_zz, prior_zy, args.latent_dim, 1, has_inputs=args.has_inputs)
    guide = ConjugateStateSpaceModel_Guide(model, has_inputs=args.has_inputs)

    smc = FastOnlineSMCFilter(model, guide, num_particles=args.num_particles, max_plate_nesting=1, 
                              max_timesteps=args.num_timesteps, forget_prob=0.5)

    # Generate data
    logging.info('Generating data')
    if not args.nile:
        ys, clean_ys = generate_data(args)
    else:
        ys = get_nile_data(args)

    # Perform inference and forecasting
    logging.info('Performing inference')
    forecasts = {}
    smc.init(initial=torch.zeros(args.latent_dim).expand((args.num_particles, args.latent_dim)))
    for y in ys:
        forecasts.update(get_forecast(*smc.forecast()))
        smc.step(y=y)

    # Log predictions and plot
    logging.info("=============================")
    logging.info('Forecasts\tTruth\tPred_Mean\tPred_Variance\t0.05 CI\t0.95 CI')
    emps = []
    # nll = 0.
    mse = np.zeros(len(ys))
    for t in range(len(ys)):
        y_emp = forecasts["y_{}".format(t)]

        mse[t] = (y_emp.mean - ys[t])**2
        # nll -= y_emp.log_prob(ys[t])
        samples = np.array(y_emp.enumerate_support()).flatten()
        samples = np.sort(samples)
        conf_left = samples[int(np.round(args.num_particles*0.05)) - 1]
        conf_right = samples[int(np.round(args.num_particles*0.95)) - 1]
        emps.append(np.array([y_emp.mean, conf_left, conf_right]))

        if not args.nile:
            logging.info("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(t, ys[t], clean_ys[t], y_emp.mean, y_emp.variance, conf_left, conf_right))
        else:
            logging.info("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(t, ys[t], y_emp.mean, y_emp.variance, conf_left, conf_right, mse[t]))

    log_probs, log_weights = smc.get_log_probs()
    assert(torch.all(log_weights == 0.)) # If not then our nll computation is wrong
    # print(log_probs)
    mean_nll = extract_mean_nll(log_probs)

    logging.info("MSE:\t{}".format(np.mean(mse)))
    logging.info("Last 400 MSE:\t{}".format(np.mean(mse[-400:])))
    logging.info("NLL:\t{}".format(mean_nll))
    plot(np.arange(len(ys)), np.array(ys), np.array(emps), nile=args.nile)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conjugate State Space Model w/ SMC Filtering Inference")
    parser.add_argument("-n", "--num-timesteps", default=50, type=int)
    parser.add_argument("-p", "--num-particles", default=100, type=int)
    parser.add_argument("-d", "--latent-dim", default=5, type=int)
    parser.add_argument("--has-inputs", default=False, type=bool)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--nile", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
