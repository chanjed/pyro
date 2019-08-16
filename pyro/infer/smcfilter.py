import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.poutine.util import prune_subsample_sites


class SMCFilter(object):
    """
    :class:`SMCFilter` is the top-level interface for filtering via sequential
    monte carlo.

    The model and guide should be objects with two methods: ``.init()`` and
    ``.step()``. These two methods should have the same signature as :meth:`init`
    and :meth:`step` of this class. These methods are intended to be called first
    with :meth:`init`, then with :meth:`step` repeatedly.

    :param object model: probabilistic model defined as a function
    :param object guide: guide used for sampling defined as a function
    :param int num_particles: The number of particles used to form the
        distribution.
    :param int max_plate_nesting: Bound on max number of nested
        :func:`pyro.plate` contexts.
    """
    # TODO: Add window kwarg that defaults to float("inf")
    def __init__(self, model, guide, num_particles, max_plate_nesting):
        self.model = model
        self.guide = guide
        self.num_particles = num_particles
        self.max_plate_nesting = max_plate_nesting

        # Equivalent to an empirical distribution.
        self._values = {}
        self._log_probs = {}
        self._log_weights = torch.zeros(self.num_particles)

    def init(self, *args, **kwargs):
        """
        Perform any initialization for sequential importance resampling.
        Any args or kwargs are passed to the model and guide
        """
        self.particle_plate = pyro.plate("particles", self.num_particles, dim=-1-self.max_plate_nesting)
        with poutine.block(), self.particle_plate:
            guide_trace = poutine.trace(self.guide.init).get_trace(*args, **kwargs)
            model = poutine.replay(self.model.init, guide_trace)
            model_trace = poutine.trace(model).get_trace(*args, **kwargs)

        self._update(model_trace, guide_trace)
        # self._update_weights(model_trace, guide_trace)
        # self._update_values(model_trace)
        # self._maybe_importance_resample()

    def step(self, *args, **kwargs):
        """
        Take a filtering step using sequential importance resampling updating the
        particle weights and values while resampling if desired.
        Any args or kwargs are passed to the model and guide
        """
        with poutine.block(), self.particle_plate:
            guide_trace = poutine.trace(self.guide.step).get_trace(*args, **kwargs)
            model = poutine.replay(self.model.step, guide_trace)
            model_trace = poutine.trace(model).get_trace(*args, **kwargs)

        self._update(model_trace, guide_trace)
        # self._update_weights(model_trace, guide_trace)
        # self._update_values(model_trace)
        self._maybe_importance_resample()

    def resample(self, index):
        self._values = {name: value[index].contiguous() for name, value in self._values.items()}
        self._log_probs = {name: log_prob[index].contiguous() for name, log_prob in self._log_probs.items()}
        self._log_weights.fill_(0.)

    def forecast(self, *args, **kwargs):
        """
        Take a forecasting step using 
        """
        with poutine.block(), self.particle_plate:
            model_trace = poutine.trace(self.model.forecast).get_trace(*args, **kwargs)

        return self._extract_samples(model_trace), self._log_weights


    def get_values_and_log_weights(self):
        """
        Returns the particles and its (unnormalized) log weights.
        :returns: the values and unnormalized log weights.
        :rtype: tuple of dict and floats where the dict is a key of name of latent to value of latent.
        """
        return self._values, self._log_weights

    def get_empirical(self):
        """
        :returns: a marginal distribution over every latent variable.
        :rtype: a dictionary with keys which are latent variables and values
            which are :class:`~pyro.distributions.Empirical` objects.
        """
        return {name: dist.Empirical(value, self._log_weights)
                for name, value in self._values.items()}

    def get_log_probs(self):
        return self._log_probs, self._log_weights

    def _update(self, model_trace, guide_trace):
        self._update_weights(model_trace, guide_trace)
        self._values.update(self._extract_samples(trace))
        self._update_log_probs(model_trace)

    @torch.no_grad()
    def _update_weights(self, model_trace, guide_trace):
        # w_t <-w_{t-1}*p(y_t|z_t) * p(z_t|z_t-1)/q(z_t)

        model_trace = prune_subsample_sites(model_trace)
        guide_trace = prune_subsample_sites(guide_trace)

        model_trace.compute_log_prob()
        guide_trace.compute_log_prob()

        for name, guide_site in guide_trace.nodes.items():
            if guide_site["type"] == "sample":
                model_site = model_trace.nodes[name]
                log_p = model_site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                log_q = guide_site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                self._log_weights += log_p - log_q

        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                log_p = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                self._log_weights += log_p

        self._log_weights -= self._log_weights.max()

    @torch.no_grad()
    def _update_log_probs(self, model_trace):
        # w_t <-w_{t-1}*p(y_t|z_t) * p(z_t|z_t-1)/q(z_t)
        model_trace = prune_subsample_sites(model_trace).compute_log_prob()

        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                log_p = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                self._log_probs.update({name: log_p})

    @staticmethod
    def _extract_samples(trace):
        return {name: site["value"]
                for name, site in trace.nodes.items()
                if site["type"] == "sample"
                if not site["is_observed"]
                if type(site["fn"]).__name__ != "_Subsample"}

    def _maybe_importance_resample(self):
        if torch.rand(1) > 1.0:  # TODO check perplexity
            self._importance_resample()

    def _importance_resample(self):
        # TODO: Turn quadratic algo -> linear algo by being lazier
        index = dist.Categorical(logits=self._log_weights).sample(sample_shape=(self.num_particles,))
        self.resample(index)
        if hasattr(self.model, "resample"):
            self.model.resample(index)
        if hasattr(self.guide, "resample"):
            self.guide.resample(index)


class FastOnlineSMCFilter(SMCFilter):

    def __init__(self, model, guide, num_particles, max_plate_nesting, max_timesteps, forget_prob, steps_until_downdate=20):
        SMCFilter.__init__(self, model, guide, num_particles, max_plate_nesting)
        self.t = -1 # Hack to handle init
        self.steps_until_downdate = steps_until_downdate
        self.history = NegativeBinomialHistory(num_particles, max_timesteps, forget_prob)

    def resample(self, index):
        self.history.resample(index)
        self._log_weights.fill_(0.)

    def get_values_and_log_weights(self):
        """
        Returns the particles and its (unnormalized) log weights.
        :returns: the values and unnormalized log weights.
        :rtype: tuple of dict and floats where the dict is a key of name of latent to value of latent.
        """
        return self.history._values, self.history._log_weights

    def get_empirical(self):
        """
        :returns: a marginal distribution over every latent variable.
        :rtype: a dictionary with keys which are latent variables and values
            which are :class:`~pyro.distributions.Empirical` objects.
        """
        return {name: dist.Empirical(value, self.history._log_weights)
                for name, value in self.history._values.items()}

    def get_log_probs(self):
        return {prefix: log_prob[:,:self.t,...] for prefix, log_prob in self.history._log_probs.items()}, self._log_weights

    def _update(self, model_trace, guide_trace):
        self.history.update_history(model_trace, self.t)
        self._update_weights(model_trace, guide_trace)
        if self.t >= self.steps_until_downdate:
            print("DOWNDATE IS HAPPENING")
            self.history.sample_downdate(self.t)
            self._downdate_weights(self.history)
            self.model.downdate(self.history)
        self.t += 1

    def _downdate_weights(self, history):
        # Get weights from log_probs
        for prefix in self.history._log_probs.keys():
            assert(self._log_weights.shape == self.history.get_downdate_log_probs(prefix).sum(1).shape)
            self._log_weights -= self.history.get_downdate_log_probs(prefix).sum(1)

        self._log_weights -= self._log_weights.max()

class NegativeBinomialHistory:
    def __init__(self, num_particles, max_timesteps, forget_prob):
        self.num_particles = num_particles
        self.max_timesteps = max_timesteps
        self.forget_prob = forget_prob
        self._total_forgotten = torch.zeros(self.num_particles).int()
        self._values = {}
        self._obs = {}
        self._log_probs = {}
        self._initial = {}
        self.downdate_sampler = torch.distributions.NegativeBinomial(torch.tensor(1.), probs=torch.tensor(self.forget_prob)).expand((self.num_particles,))

    def get_num_forget(self):
        return self._num_forget

    def get_downdate_values(self, prefix, offset=0):
        # TODO: Can probably clean this up to one method using getattr
        if self.mask is None:
            return None
        assert(self.mask.dim() == 2)
        if self._window_min + offset == -1:
            # Add initial in here.
            windowed_vals = self._values[prefix][:, self._window_min + offset + 1:self._window_max + offset, ...]
            windowed_vals = torch.cat([self._initial[prefix][:, None, ...], windowed_vals], 1)
        else:
            windowed_vals = self._values[prefix][:, self._window_min + offset:self._window_max + offset, ...]
        return (windowed_vals.transpose(1,-1).transpose(0,-2) * self.mask).transpose(0,-2).transpose(1,-1)

    def get_downdate_obs(self, prefix, offset=0):
        if self.mask is None:
            return None
        assert(self.mask.dim() == 2)
        windowed_obs = self._obs[prefix][None, self._window_min + offset:self._window_max + offset, ...]
        return  (windowed_obs.transpose(1,-1).transpose(0,-2) * self.mask).transpose(0,-2).transpose(1,-1)

    def get_downdate_log_probs(self, prefix, offset=0):
        if self.mask is None:
            return None
        assert(self.mask.dim() == 2)
        windowed_log_probs = self._log_probs[prefix][:, self._window_min + offset:self._window_max + offset, ...]
        return  (windowed_log_probs.transpose(1,-1).transpose(0,-2) * self.mask).transpose(0,-2).transpose(1,-1)

    # def sample_downdate(self, t):
    #     # Maybe need bookkeeping of whats already been downdated?
    #     self._num_forget = self.downdate_sampler.sample()
    #     self._num_forget = torch.max(torch.min(self._num_forget, torch.tensor(t - 1.)), torch.tensor(0.)).type(self._total_forgotten.type()) # Leave at least one obs and don't allow negatives. 
    #     self._window_min = self._total_forgotten.min()
    #     self._window_max = (self._total_forgotten + self._num_forget).max()
    #     self.mask = self._length_to_mask(self._total_forgotten + self._num_forget - self._window_min) - self._length_to_mask(self._total_forgotten - self._window_min, max_len = self._window_max - self._window_min)
    #     assert(self.mask.size(0) == self.num_particles)
    #     assert(self.mask.size(1) == self._window_max - self._window_min)
    #     if self.mask.size(1) == 0:
    #         self.mask = None
    #     # update total forgotten
    #     self._total_forgotten += self._num_forget

    def sample_downdate(self, t):
        # Maybe need bookkeeping of whats already been downdated?
        self._num_forget = torch.tensor(1).expand(self.num_particles).type(self._total_forgotten.type())
        # self._num_forget = torch.max(torch.min(self._num_forget, torch.tensor(t - 1.)), torch.tensor(0.)).type(self._total_forgotten.type()) # Leave at least one obs and don't allow negatives. 
        self._window_min = self._total_forgotten.min()
        self._window_max = (self._total_forgotten + self._num_forget).max()
        self.mask = self._num_forget.float().unsqueeze(-1)
        # self.mask = self._length_to_mask(self._total_forgotten + self._num_forget - self._window_min) - self._length_to_mask(self._total_forgotten - self._window_min, max_len = self._window_max - self._window_min)
        # assert(self.mask.size(0) == self.num_particles)
        # assert(self.mask.size(1) == self._window_max - self._window_min)
        # if self.mask.size(1) == 0:
        #     self.mask = None
        # update total forgotten
        self._total_forgotten += self._num_forget

    @staticmethod
    def _length_to_mask(length, max_len=None, dtype=None):
        """length: B.
        return B x max_len
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1 #Length shape should be 1 dimensional.
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype)
        return mask.float()

    def resample(self, index):
        self._values = {name: value[index].contiguous() for name, value in self._values.items()}
        self._log_probs = {name: log_prob[index].contiguous() for name, log_prob in self._log_probs.items()}
        self._initial = {name: initial[index].contiguous() for name, initial in self._initial.items()}
        self._total_forgotten = self._total_forgotten[index].contiguous()


    def update_history(self, trace, t):
        if t == -1:
            self._set_initial(trace)
        if t >= 0:
            self._update_values(trace, t)
            self._update_log_probs(trace, t)
            self._update_obs(trace, t)

    def _set_initial(self, trace):
        samples = {name: site["value"]
                   for name, site in trace.nodes.items()
                   if site["type"] == "sample"
                   if not site["is_observed"]
                   if type(site["fn"]).__name__ != "_Subsample"}

        for name, initial in samples.items():
            assert("_" in name)
            prefix, time = tuple(name.split("_"))
            assert(int(time) == -1)
            self._initial.update({prefix: initial})

    def _update_values(self, trace, t):
        # self._values: prefix -> tensor(particles x time x dim)
        samples = {name: site["value"]
                   for name, site in trace.nodes.items()
                   if site["type"] == "sample"
                   if not site["is_observed"]
                   if type(site["fn"]).__name__ != "_Subsample"}

        for name, value in samples.items():
            assert(t != -1)
            assert("_" in name)
            prefix, time = tuple(name.split("_"))
            assert(t == int(time))
            assert(t < self.max_timesteps)
            if t == 0:
                self._values.update({prefix: torch.zeros((self.num_particles, self.max_timesteps)
                                                         + value.shape[1:])})
                print("VALUE 0:", value)
            assert(prefix in self._values.keys())
            self._values[prefix][:,t,...] = value


    def _update_log_probs(self, trace, t):
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()

        for name, site in trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                assert(t != -1)
                assert("_" in name)
                prefix, time = tuple(name.split("_"))
                assert(t == int(time))
                assert(t < self.max_timesteps)
                log_p = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                if t == 0:
                    self._log_probs.update({prefix: torch.zeros((self.num_particles, self.max_timesteps) 
                                           + log_p.shape[1:])})
                self._log_probs[prefix][:, t, ...] = log_p

    def _update_obs(self, trace, t):
        samples = {name: site["value"]
                    for name, site in trace.nodes.items()
                    if site["type"] == "sample"
                    if site["is_observed"]
                    if type(site["fn"]).__name__ != "_Subsample"}

        for name, obs in samples.items():
            assert(t != -1)
            assert("_" in name)
            prefix, time = tuple(name.split("_"))
            assert(t == int(time))
            assert(t < self.max_timesteps)
            if t == 0:
                self._obs.update({prefix: torch.zeros(obs.shape).expand(self.max_timesteps, -1)})
            assert(prefix in self._obs.keys())
            assert(obs.size(0) != self.num_particles)
            self._obs[prefix][t,...] = obs
