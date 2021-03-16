import warnings
import logging
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])

def get_percentile_val(fn, x0, percentile):
    try:
        y0 = sorted([fn(x) for x in x0])
        index = int(percentile * len(y0))
        return y0[index], True
    except:
        logging.exception('Error in get_percentile_val().')
        return 0, False

def acq_sigma(gp, y_max, bounds, random_state, percentile, n_iter=1000):

    min_acq = None

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = get_percentile_val(lambda x: gp.predict(x.reshape(1, -1), return_std=True)[1],
                       x_try.reshape(1, -1),
                       percentile)
                    #    bounds=bounds,
                    #    method="L-BFGS-B")

        # See if success
        if not res[1]:
            continue

        # Store it if better than previous minimum(maximum).
        if min_acq is None or res[0] <= min_acq:
            min_acq = res[0]
    return min_acq

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0, kappa_min=0):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay
        self._kappa_min = kappa_min

        self.xi = xi

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay and self.kappa > self._kappa_min:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)

class MultiUtilityFunction(UtilityFunction):
    def __init__(self, kind, kappa, alpha, xi, source_bo_list, kappa_decay=1, kappa_decay_delay=0,
                    alpha_decay=1, alpha_decay_delay=0, alpha_min=0, kappa_min=0, power=1):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay
        self._kappa_min = kappa_min
        self.alpha = alpha
        self._alpha_min = alpha_min
        self._alpha_decay = alpha_decay
        self._alpha_decay_delay = alpha_decay_delay
        self._pow = power
        self.source_gp_list = [bo._gp for bo in source_bo_list]

        self.xi = xi

        self._iters_counter = 0

        if kind not in ['multi_ucb_weighted', 'multi_flat', 'alternate', 'multi_ucb', 'ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of 'multi_ucb_weighted'," \
                  "'multi_flat', 'alternate', 'multi_ucb', 'ucb', 'ei', 'poi'.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'multi_ucb':
            return self._multi_ucb(x, target_gp=gp, source_gp_list=self.source_gp_list, kappa=self.kappa, alpha=self.alpha, power=self._pow)
        if self.kind == 'multi_flat':
            return self._multi_flat_state(x, target_gp=gp, kappa=self.kappa, alpha=self.alpha, power=self._pow)
        if self.kind == 'multi_ucb_weighted':
            return self._multi_ucb_weighted(x, target_gp=gp, source_gp_list=self.source_gp_list, kappa=self.kappa)
        if self.kind == 'alternate':
            return self._alternate_multi(x, target_gp=gp, source_gp_list=self.source_gp_list, kappa=self.kappa)
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _multi_ucb(x, target_gp, source_gp_list, kappa, alpha, power=1):
        target_mean, target_std = target_gp.predict(x, return_std=True)
        source_mean_sum = 0
        for source_gp in source_gp_list:
            source_mean_sum += source_gp.predict(x, return_std=False)
        source_mean_avg = source_mean_sum / len(source_gp_list)
        return target_mean + source_mean_avg + ((target_mean - source_mean_avg + (alpha * target_std )) * np.exp(-np.power(kappa * target_std, power)))

    @staticmethod
    def _multi_flat_state(x, target_gp, kappa, alpha, power=1):
        target_mean, target_std = target_gp.predict(x, return_std=True)
        return target_mean + 0.5 + ((target_mean - 0.5 + (alpha * target_std )) * np.exp(-np.power(kappa * target_std, power)))

    @staticmethod
    def _multi_ucb_weighted(x, target_gp, source_gp_list, kappa):
        target_mean, target_std = target_gp.predict(x, return_std=True)
        source_mean_sum = 0
        inverse_std_sum = 0
        for source_gp in source_gp_list:
            mean, stdev = source_gp.predict(x, return_std=True)
            source_mean_sum += mean / stdev
            inverse_std_sum += 1 / stdev
        source_mean_avg = source_mean_sum / inverse_std_sum
        return target_mean + source_mean_avg - ((source_mean_avg - target_mean) * np.exp(-(target_std * kappa)))

    @staticmethod
    def _alternate_multi(x, target_gp, source_gp_list, kappa):
        target_mean, target_std = target_gp.predict(x, return_std=True)
        source_mean_sum = 0
        for source_gp in source_gp_list:
            source_mean_sum += source_gp.predict(x, return_std=False)
        source_mean_avg = source_mean_sum / len(source_gp_list)
        return target_mean + source_mean_avg - ((source_mean_avg - target_mean) * (np.exp(-(target_std * kappa)) + (kappa * target_std)))

    def update_params(self, gp, y_max, bounds, random_state, percentile):
        self._iters_counter += 1
        if type(self._kappa_decay) == str:
            if self._kappa_decay == 'dynamic':
                self.kappa = 1 / acq_sigma(gp=gp,
                                        y_max=y_max,
                                        bounds=bounds,
                                        random_state=random_state,
                                        percentile=percentile)

        if type(self._kappa_decay) == float:
            if self._iters_counter > self._kappa_decay_delay and self.kappa > self._kappa_min:
                self.kappa *= self._kappa_decay

        if self._iters_counter > self._alpha_decay_delay and self.alpha > self._alpha_min:
            self.alpha *= self._alpha_decay

def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)
