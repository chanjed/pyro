import pytest
import torch
from torch.testing import assert_allclose
import numpy as np
import pdb
import pyro
import pyro.distributions as dist
from pyro.distributions import Bernoulli, Beta, BetaBernoulliSummary, Gamma, Normal, NIGNormalRegressionSummary


def test_betabern_smoke():
    summary = BetaBernoulliSummary(2., 4.)
    multiple_batch_summary = BetaBernoulliSummary(torch.ones(3), torch.ones(3))

    obs = torch.rand((1, 1))
    batch_obs = torch.rand((5, 1))
    multiple_batch_obs1 = torch.rand((3, 5, 1))
    multiple_batch_obs2 = torch.rand((3, 1, 1))

    summary.update(obs)
    summary.update(batch_obs)
    multiple_batch_summary.update(multiple_batch_obs1)
    multiple_batch_summary.update(multiple_batch_obs2)


def test_betabern_asymptotics():
    summary = BetaBernoulliSummary(torch.Tensor([2.]), torch.Tensor([4.]))
    obs = Bernoulli(probs=0.3).sample(sample_shape=torch.Size([100, 1]))

    summary.update(obs)
    assert_allclose(Beta(summary.alpha, summary.beta).mean, 0.3, rtol=0.0, atol=0.05)
    assert_allclose(Beta(summary.alpha, summary.beta).variance, 0.0, rtol=0.0, atol=0.1)


@pytest.mark.parametrize("features_dim", [1, 2])
@pytest.mark.parametrize("obs_dim", [1, 3])
@pytest.mark.parametrize("batch_dim", [1, 4])
def test_nignorm_smoke(features_dim, obs_dim, batch_dim):
    summary_null = NIGNormalRegressionSummary(5., 6., 3., 1.)
    summary_mixed = NIGNormalRegressionSummary(torch.tensor([5.]), 6., torch.tensor(3.), 1.)
    summary_nobatch = NIGNormalRegressionSummary(torch.zeros(features_dim).expand((obs_dim, features_dim)),
                                                 torch.eye(features_dim).expand((obs_dim, features_dim, features_dim)),
                                                 torch.tensor(3.),
                                                 torch.tensor(1.).expand(obs_dim))
    summary_bcast = NIGNormalRegressionSummary(torch.zeros(features_dim).expand((1, features_dim)),
                                               torch.eye(features_dim).expand((1, features_dim, features_dim)),
                                               torch.tensor(3.),
                                               torch.tensor(1.).expand(obs_dim))
    summary_batch = NIGNormalRegressionSummary(torch.zeros(features_dim).expand((batch_dim, obs_dim, features_dim)),
                                               torch.eye(features_dim)
                                               .expand((batch_dim, obs_dim, features_dim, features_dim)),
                                               torch.tensor(3.).expand(obs_dim),
                                               torch.tensor(1.).expand(batch_dim, obs_dim))
    summary_b_bcast = NIGNormalRegressionSummary(torch.zeros(features_dim).expand((batch_dim, 1, features_dim)),
                                                 torch.eye(features_dim)
                                                 .expand((batch_dim, 1, features_dim, features_dim)),
                                                 torch.tensor(3.).expand(1),
                                                 torch.tensor(1.).expand(batch_dim, 1))
    summary_list = [summary_null, summary_mixed, summary_nobatch, summary_bcast, summary_batch, summary_b_bcast]

    features = torch.rand((1, features_dim))
    obs = torch.rand((1, obs_dim))
    features_batch = torch.rand((5, features_dim))
    obs_batch = torch.rand((5, obs_dim))

    for s in summary_list:
        s.update(obs, features)
        s.update(obs_batch, features_batch)
        (s.mean, s.covariance, s.shape, s.rate)

    # TODO: test swapping between obs_dim is not allowed


def test_nignorm_prior():
    true_mean = torch.tensor([2., 1.])
    true_covariance = torch.tensor([[5., 2.], [2., 5.]])
    true_shape = 3.
    true_rate = 1.
    summary = NIGNormalRegressionSummary(true_mean, true_covariance, true_shape, true_rate)

    assert_allclose(summary.mean, true_mean)
    assert_allclose(summary.covariance, true_covariance)
    assert_allclose(summary.shape, true_shape)
    assert_allclose(summary.rate, true_rate)


def test_conversion():
    true_mean = torch.tensor([2., 1., 0.]).expand((4, 2, 3))
    true_covariance = torch.eye(3).expand((4, 2, 3, 3))
    true_shape = 3.
    true_rate = 1.
    summary = NIGNormalRegressionSummary(true_mean, true_covariance, true_shape, true_rate)
    assert_allclose(summary.precision_times_mean,
                    summary.covariance.inverse().matmul(summary.mean.unsqueeze(-1)).squeeze(-1))
    assert_allclose(summary.precision, summary.covariance.inverse())
    assert_allclose(summary.reparametrized_rate, summary.rate + 0.5
                    * summary.mean.unsqueeze(-2).matmul(summary.covariance.inverse())
                    .matmul(summary.mean.unsqueeze(-1)).squeeze(-1).squeeze(-1))
    summary.update(torch.rand((100, 2)), torch.rand((100, 3)))
    assert(summary.mean is not None)
    assert_allclose(summary.precision_times_mean,
                    summary.covariance.inverse().matmul(summary.mean.unsqueeze(-1)).squeeze(-1))
    assert_allclose(summary.precision, summary.covariance.inverse())
    assert_allclose(summary.reparametrized_rate, summary.rate + 0.5
                    * summary.mean.unsqueeze(-2).matmul(summary.covariance.inverse())
                    .matmul(summary.mean.unsqueeze(-1)).squeeze(-1).squeeze(-1))

    features = torch.rand((4, 3))
    features = features[..., None, None, :]
    loc = features.matmul(summary.precision.inverse()).matmul(summary.precision_times_mean
                                                              .unsqueeze(-1)).squeeze(-1).squeeze(-1)
    term1 = (summary.reparametrized_rate - 0.5*summary.precision_times_mean.unsqueeze(-2)
             .matmul(summary.precision.inverse())
             .matmul(summary.precision_times_mean.unsqueeze(-1)).squeeze(-1).squeeze(-1))/summary.shape
    term2 = 1. + features.matmul(summary.precision.inverse()).matmul(features.transpose(-2, -1)).squeeze(-1).squeeze(-1)

    canonical_loc = features.matmul(summary.mean.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    canonical_term1 = summary.rate/summary.shape
    canonical_term2 = 1. + features.matmul(summary.covariance).matmul(features
                                                                      .transpose(-2, -1)).squeeze(-1).squeeze(-1)
    assert_allclose(loc, canonical_loc)
    assert_allclose(term1, canonical_term1)
    assert_allclose(term2, canonical_term2)


def test_nignorm_asymptotics():
    # test the likelihood being correct
    # include conversions between forms
    weights = torch.tensor([2., 1.])
    variance = 10.
    noise = Normal(0., np.sqrt(variance))
    features = torch.rand((10000, 2))
    obs = features.matmul(weights).unsqueeze(-1) + noise.sample(sample_shape=torch.Size([10000, 1]))

    summary = NIGNormalRegressionSummary(torch.tensor([0.5, 0.5]),
                                         torch.tensor([[3., 0.5], [0.5, 3.]]), 1.1, 10.)
    summary.update(obs, features)

    assert_allclose(summary.mean, weights, rtol=0.0, atol=0.1)
    assert_allclose(summary.covariance, torch.zeros((2, 2)), rtol=0.0, atol=0.1)
    assert_allclose(1./Gamma(summary.shape, summary.rate).mean, variance, rtol=0.0, atol=0.1)
    assert_allclose(Gamma(summary.shape, summary.rate).variance, 0., rtol=0.0, atol=0.1)

def _get_posterior_predictive(summary, features):
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

def test_vanilla_asymptotics():
    weights = torch.tensor([2., 1.])
    variance = 10.
    noise = Normal(0., np.sqrt(variance))
    summary = NIGNormalRegressionSummary(torch.tensor([0.5, 0.5]),
                                             torch.tensor([[1., 0.5], [0.5, 1.]]), 3., 1.)
    for i in range(100):
        features = torch.rand((1, 2))
        obs = features.matmul(weights).unsqueeze(-1) + noise.sample(sample_shape=torch.Size([1, 1]))
        summary.update(obs, features)
        assert(summary.rate > 0.)

def test_nig_downdate():
    mean = torch.tensor([0.5, 0.5])
    cov = torch.tensor([[1., 0.5], [0.5, 1.]])
    shape = torch.tensor(3.)
    rate = torch.tensor(1.)
    summary = NIGNormalRegressionSummary(mean, cov, shape, rate)
    features = torch.rand((51,2))
    for i in range(50):
        summary.update(features[i+1][None,...],features[i][None,...])
    summary.downdate(features[1:], features[:-1], num_forget=torch.tensor(50))

    assert(torch.abs(summary.mean - mean).max() < 1.e-6)
    assert(torch.abs(summary.covariance - cov).max() < 1.e-6)
    assert(torch.abs(summary.shape - shape).max() < 1.e-6)
    assert(torch.abs(summary.rate - rate).max() < 1.e-6)
    # assert(torch.abs(summary.precision_times_mean - cov.inverse().times(mean)).max() < 1.e-6)
    # assert(torch.abs(summary.precision - cov.inverse()).max() < 1.e-6)
    # assert(torch.abs(summary.reparametrized_rate -).max() < 1.e-6)



# def test_numerical():
#     batches = 2
#     prior_mean = torch.tensor(0.).expand((batches,1,1))
#     prior_covariance = torch.tensor(1.).expand((batches,1,1,1))
#     prior_shape = torch.tensor(3.).expand((batches,1))
#     prior_rate = torch.tensor(1.).expand((batches,1))
#     summary = NIGNormalRegressionSummary(prior_mean, prior_covariance, prior_shape, prior_rate)
#     x = torch.zeros((batches,1))
#     for i in range(100000):
#         y = _get_posterior_predictive(summary, x).sample()
#         summary.update(y[...,None,:],x[...,None,:])
#         print("Time: ", i)
#         assert_allclose(summary.mean, ((prior_mean + prior_covariance.squeeze(-1)*x.unsqueeze(-1)*y.unsqueeze(-1))/(1. + prior_covariance.squeeze(-1)*(x.unsqueeze(-1))**2)), rtol=1.e-1, atol=0.0)
#         assert_allclose(summary.covariance, (1./((1./prior_covariance) + (x.unsqueeze(-1).unsqueeze(-1)**2))), rtol=1.e-1, atol=0.0)
#         assert_allclose(summary.shape, prior_shape + 0.5, rtol=1.e-1, atol=0.0)
#         assert_allclose(summary.rate, (prior_rate + (y - prior_mean.squeeze(-1)*x)**2/(2.*(1.+prior_covariance.squeeze(-1).squeeze(-1)*x**2))), rtol=1.e-1, atol=0.0)
#         # if torch.abs(summary.covariance).min() < 1.e-8:
#         #     pdb.set_trace()
#         if torch.any(summary.rate + 1e-2 < prior_rate):
#             pdb.set_trace()
#         prior_mean = summary.mean
#         prior_covariance = summary.covariance
#         prior_shape = summary.shape
#         prior_rate = summary.rate
#         x = y.clone()


# def test_forward_backward():
#     prior_zz = (torch.zeros((3, 2, 2)), 
#                 torch.eye(2).expand((3, 2, 2, 2)),
#                 torch.tensor(3.).expand(3,2), 
#                 torch.tensor(1.).expand(3,2))
#     summary = NIGNormalRegressionSummary(*prior_zz)
#     x = torch.zeros(3,2)

#     for t in range(5):
#         print("timestep: ", t)
#         sigma2 = 1./dist.Gamma(summary.shape, summary.rate).expand((100000, 3, 2)).sample()
#         # print("Sigma shape: ", sigma2.shape)
#         weights = dist.MultivariateNormal(summary.mean, covariance_matrix=sigma2[...,None,None]*summary.covariance).sample()
#         # print("Weights shape: ", weights.shape)
#         noise = dist.Normal(0., torch.sqrt(sigma2)).sample()
#         # print("Noise shape:", noise.shape)
#         # print("Z shape:", z.shape)
#         y = weights.matmul(x.unsqueeze(-1)).squeeze(-1) + noise

#         student_t = _get_posterior_predictive(summary, x)
#         y_marginal = pyro.sample("y", student_t.expand((100000, 3, 2)))
#         print("x: ", x.shape)
#         print("y_marginal: ", y_marginal.shape)
#         summary.update(y_marginal[...,None,:], x[...,None,:])
#         sigma2_marginal = 1./dist.Gamma(summary.shape, summary.rate).expand((100000, 3, 2)).sample()
#         # print("Sigma shape: ", sigma2_marginal.shape)
#         weights_marginal = dist.MultivariateNormal(summary.mean, covariance_matrix=sigma2_marginal[...,None,None]*summary.covariance).sample()
#         # print("Weights shape: ", weights_marginal.shape)
#         noise_marginal = dist.Normal(0., torch.sqrt(sigma2_marginal)).sample()
#         print("y: ", y.shape)
#         assert_allclose(y.mean([0,1]), y_marginal.mean([0,1]), rtol=0.0, atol=0.1)
#         assert_allclose(y.var([0,1]), y_marginal.var([0,1]), rtol=0.0, atol=0.1)
#         print("Initial variance: ", sigma2.var([0,1]))
#         print("Updated variance: ", sigma2_marginal.var([0,1]))
#         assert_allclose(sigma2.mean([0,1]), sigma2_marginal.mean([0,1]), rtol=0.0, atol=0.1)
#         assert_allclose(sigma2.var([0,1]), sigma2_marginal.var([0,1]), rtol=0.0, atol=0.1)
#         assert_allclose(weights.mean(0), weights_marginal.mean(0), rtol=0.0, atol=0.1)
#         assert_allclose(weights.var(0), weights_marginal.var(0), rtol=0.0, atol=0.1)
#         x = y_marginal[0,:,:]




# def test_nignorm_posterior_predictive_check():
#     latent_dim = 2
#     prior_zz = (torch.zeros((3, latent_dim, latent_dim)), 
#                 torch.eye(latent_dim).expand((3, latent_dim, latent_dim, latent_dim)),
#                 torch.tensor(3.).expand(3,latent_dim), 
#                 torch.tensor(1.).expand(3,latent_dim))
#     summary = NIGNormalRegressionSummary(*prior_zz)
#     z = torch.zeros(3, latent_dim)
#     old_rate = summary.rate

#     for t in range(50):
#         print("t: ", t)
#         sigma2 = 1./dist.Gamma(summary.shape, summary.rate).expand((50000, 3, latent_dim)).sample()
#         print("Sigma shape: ", sigma2.shape)
#         weights = dist.MultivariateNormal(summary.mean, covariance_matrix=sigma2[...,None,None]*summary.covariance).sample()
#         print("Weights shape: ", weights.shape)
#         noise = dist.Normal(0., torch.sqrt(sigma2)).sample()
#         print("Noise shape:", noise.shape)
#         print("Z shape:", z.shape)
#         y = weights.matmul(z.unsqueeze(-1)).squeeze(-1) + noise
#         student_t = _get_posterior_predictive(summary, z)
#         assert_allclose(student_t.mean, y.mean(0), rtol=0.0, atol=0.1)
#         assert_allclose(student_t.variance, y.var(0), rtol=0.0, atol=0.1)
#         new_z = pyro.sample("z", student_t)
#         summary.update(new_z[...,None,:], z[...,None,:])
#         # assert(torch.all(old_rate < summary.rate))
#         old_rate = summary.rate
#         z = new_z

# def test_nignorm_posterior_predictive_asymptotics():
#     batch = 1
#     latent_dim = 10
#     prior_zz = (torch.zeros((batch, latent_dim, latent_dim)), 
#                 torch.eye(latent_dim).expand((batch, latent_dim, latent_dim, latent_dim)),
#                 torch.tensor(3.).expand(batch,latent_dim), 
#                 torch.tensor(1.).expand(batch,latent_dim))
#     summary = NIGNormalRegressionSummary(*prior_zz)
#     z = torch.zeros(batch, latent_dim)
#     old_rate = torch.tensor(1.).expand(batch,latent_dim)

#     for t in range(1000):
#         print("t: ", t)
#         pyro.set_rng_seed(t*10)
#         new_z = pyro.sample("z", _get_posterior_predictive(summary, z))
#         print("Z_Z Mean: ",  _get_posterior_predictive(summary, z).mean)
#         print("Z_Z Var: ",  _get_posterior_predictive(summary, z).variance)
#         print("Min Cov Eig: ",   min([summary.covariance[i,0,:,:].symeig().eigenvalues.min() for i in range(batch)]))
#         print("Max Prec Eig: ",  min([summary.precision[i,0,:,:].symeig().eigenvalues.max() for i in range(batch)]))
#         summary.update(new_z[...,None,:], z[...,None,:])
#         print("Min diff rate: ", (summary.rate - old_rate).min())
#         assert(torch.all(summary.rate > old_rate))
#         old_rate = summary.rate.clone()
#         z = new_z.clone()
