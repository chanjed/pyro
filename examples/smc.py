from __future__ import absolute_import, division, print_function

import argparse

import pyro
import pyro.distributions as dist
import pyro.infer.smcfilter as smcfilter

import torch


class SimpleHarmonicModel:

    def __init__(self, process_noise, measurement_noise):
        self.A = torch.tensor([[0., 1.],
                               [-1., 0.]])
        self.B = torch.tensor([0.5, 1])
        self.sigma_z = torch.tensor(process_noise)
        self.sigma_y = torch.tensor(measurement_noise)

    def init(self, initial):
        self.t = 0
        self.z = initial
        self.y = None

    def step(self, y=None):
        self.t += 1
        self.z = pyro.sample("z_{}".format(self.t),
                             dist.Normal(self.z.matmul(self.A), self.B*self.sigma_z).to_event(1))
        self.y = pyro.sample("y_{}".format(self.t),
                             dist.Normal(self.z[..., 0], self.sigma_y),
                             obs=y)

        return self.z, self.y


class SimpleHarmonicModel_Guide:

    def __init__(self, model):
        self.model = model

    def init(self, initial):
        self.t = 0
        self.z = initial

    def step(self, y=None):
        self.t += 1

        # Proposal distribution
        self.z = pyro.sample("z_{}".format(self.t),
                             dist.Normal(self.z.matmul(self.model.A), torch.tensor([1., 1.])).to_event(1))


def generate_data(args):
    model = SimpleHarmonicModel(args.process_noise, args.measurement_noise)

    initial = torch.tensor([1., 0.])
    model.init(initial=initial)
    zs = [initial]
    ys = [None]
    for t in range(args.num_timesteps):
        z, y = model.step()
        zs.append(z)
        ys.append(y)

    return zs, ys


def main(args):
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    model = SimpleHarmonicModel(args.process_noise, args.measurement_noise)
    guide = SimpleHarmonicModel_Guide(model)

    smc = smcfilter.SMCFilter(model, guide, num_particles=args.num_particles, max_plate_nesting=0)

    zs, ys = generate_data(args)
    smc.init(initial=torch.tensor([1., 0.]))
    for y in ys[1:]:
        smc.step(y)

    empirical = smc.get_empirical()
    for t in range(1, args.num_timesteps):
        z = empirical["z_{}".format(t)]
        print("{}\t{}\t{}\t{}".format(t, zs[t], z.mean, z.variance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Harmonic Oscillator w/ SMC Filtering Inference")
    parser.add_argument("-n", "--num-timesteps", default=50, type=int)
    parser.add_argument("-p", "--num-particles", default=100, type=int)
    parser.add_argument("--process-noise", default=1., type=float)
    parser.add_argument("--measurement-noise", default=1., type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)
