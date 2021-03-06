import warnings
import random

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, MultiUtilityFunction, acq_max, ensure_rng

from copy import deepcopy
from dataclasses import dataclass, field
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)

@dataclass
class data:
    bestResult: list[float] = field(default_factory=list)
    bestPoints: list[dict] = field(default_factory=list)
    cost: list[int] = field(default_factory=list)

    def add_points(self, newBestResult, newBestPoint, newCost, cumulativeCost=True):
        self.bestResult.append(newBestResult)
        self.bestPoints.append(newBestPoint)
        if cumulativeCost and len(self.cost) > 0:
            self.cost.append(self.cost[-1] + newCost)
        else:
            self.cost.append(newCost)

class BayesianOptimization(Observable):
    """
    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    f: function
        Function to be maximized.

    pbounds: dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.

    random_state: int or numpy.random.RandomState, optional(default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provieded it is used.
        When set to None, an unseeded random state is generated.

    verbose: int, optional(default=2)
        The level of verbosity.

    bounds_transformer: DomainTransformer, optional(default=None)
        If provided, the transformation is applied to the bounds.

    Methods
    -------
    probe()
        Evaluates the function on the given points.
        Can be used to guide the optimizer.

    maximize()
        Tries to find the parameters that yield the maximum value for the
        given function.

    set_bounds()
        Allows changing the lower and upper searching bounds
    """
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, cost=0):
        self._random_state = ensure_rng(random_state)

        self._data = data()
        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        self.cost = cost
        if self._bounds_transformer:
            try:
                self._bounds_transformer.initialize(self._space)
            except (AttributeError, TypeError):
                raise TypeError('The transformer must be an instance of '
                                'DomainTransformer')

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, newData):
        self._data = newData

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.data.add_points(self.max['target'], self.max['params'], self.cost)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """
        Evaluates the function on the given points. Useful to guide the optimizer.

        Parameters
        ----------
        params: dict or list
            The parameters where the optimizer will evaluate the function.

        lazy: bool, optional(default=True)
            If True, the optimizer will evaluate the points when calling
            maximize(). Otherwise it will evaluate it at the moment.
        """
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promising point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 kappa_min=0,
                 xi=0.0,
                 **gp_params):
        """
        Probes the target space to find the parameters that yield the maximum
        value for the given function.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        acq: {'ucb', 'ei', 'poi'}
            The acquisition method used.
                * 'ucb' stands for the Upper Confidence Bounds method
                * 'ei' is the Expected Improvement method
                * 'poi' is the Probability Of Improvement criterion.

        kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
                Higher value = favors spaces that are least explored.
                Lower value = favors spaces where the regression function is the
                highest.

        kappa_decay: float, optional(default=1)
            `kappa` is multiplied by this factor every iteration.

        kappa_decay_delay: int, optional(default=0)
            Number of iterations that must have passed before applying the decay
            to `kappa`.

        xi: float, optional(default=0.0)
            [unused]
        """
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay,
                               kappa_min=kappa_min)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

            self.data.add_points(self.max['target'], self.max['params'], self.cost)

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        """Set parameters to the internal Gaussian Process Regressor"""
        self._gp.set_params(**params)

    def get_new_bounds(self, new_size):
        new_bounds = {}
        for coord in self.max['params']:
            new_bounds[coord] = (self.max['params'][coord] - new_size, self.max['params'][coord] + new_size)
        return new_bounds



class TargetBayesianOptimization(BayesianOptimization):

    def __init__(self, f, pbounds, source_bo_list, random_state=None, verbose=2,
                bounds_transformer=None, cost=1):
        BayesianOptimization.__init__(self, f, pbounds, random_state, verbose,
                bounds_transformer, cost)
        self.source_bo_list = source_bo_list

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='multi_ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 kappa_min=0,
                 alpha=2,
                 alpha_decay=1,
                 alpha_decay_delay=0,
                 alpha_min=0,
                 power=1,
                 xi=0.0,
                 feedback=0.,
                 **gp_params):
        """
        Probes the target space to find the parameters that yield the maximum
        value for the given function.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        acq: {'ucb', 'ei', 'poi'}
            The acquisition method used.
                * 'ucb' stands for the Upper Confidence Bounds method
                * 'ei' is the Expected Improvement method
                * 'poi' is the Probability Of Improvement criterion.

        kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
                Higher value = favors spaces that are least explored.
                Lower value = favors spaces where the regression function is the
                highest.

        kappa_decay: float, optional(default=1)
            `kappa` is multiplied by this factor every iteration.

        kappa_decay_delay: int, optional(default=0)
            Number of iterations that must have passed before applying the decay
            to `kappa`.

        xi: float, optional(default=0.0)
            [unused]
        """
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = MultiUtilityFunction(kind=acq,
                               kappa=kappa,
                               alpha=alpha,
                               xi=xi,
                               source_bo_list = self.source_bo_list,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay,
                               kappa_min=kappa_min,
                               alpha_decay=alpha_decay,
                               alpha_decay_delay=alpha_decay_delay,
                               alpha_min=alpha_min,
                               power=power)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params(gp=self._gp,
                            y_max=self._space.target.max(),
                            bounds=self._space.bounds,
                            random_state=self._random_state,
                            percentile=0.000)

                x_probe = self.suggest(util)
                iteration += 1
            self.probe(x_probe, lazy=False)
            extra_cost = 0
            if random.random() < feedback:
                for bo in self.source_bo_list:
                    bo.probe(x_probe, lazy=False)
                    extra_cost += bo.cost

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))
            self.data.add_points(self.max['target'], self.max['params'], self.cost + extra_cost)

        self.dispatch(Events.OPTIMIZATION_END)

    def transferData(self, other_list):
        self.data = deepcopy(other_list[0].data)
        for i in range(len(self.data.cost)):
            self.data.cost[i] *= len(other_list)
        for i in range(len(self.data.bestResult)):
            self.data.bestResult[i] = self._space.target_func(**(self.data.bestPoints[i]))
