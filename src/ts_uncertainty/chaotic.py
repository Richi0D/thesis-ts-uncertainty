"""Chaotic univariate time series generator based on dysts.

Each sample takes one coordinate x_t of a chaotic flow from the dysts library (Lorenz,
Rossler, ...), z-normalised with the statistics of the context window, and builds the
clean series

    y_t = level + slope * t + amplitude * x_t

split into a context window of length T and a forecast horizon of length H. The system,
the observed coordinate, the sampling rate (points per dominant period) and the initial
condition (random perturbation + burn-in onto the attractor) are drawn per sample.

Noise, spikes and observation shifts (level, trend, noise, amplitude) are added afterwards
by an Injector (ts_uncertainty.injection). Only shifts of the dynamics themselves are made
here, because they can't be applied to a finished series:

    timescale  the system runs faster / slower: sampling step scaled by a factor
    parameter  one system parameter is perturbed, the trajectory continues from the
               current state under the new equations (new attractor geometry)

The shift episodes are drawn as in the Injector (one persistent episode [tau, end) or a
recurring one). Parameter shifts are rejected and redrawn if the trajectory diverges or
collapses onto a fixed point.

Integration uses scipy's LSODA (the dysts default Radau with rtol 1e-12 is ~100x slower).
Install numba so the dysts right-hand sides are JIT-compiled.
"""

import copy
import itertools
import time
import warnings
from collections.abc import Iterator, Sequence

import numpy as np
from dysts import flows
from scipy.integrate import solve_ivp

from ts_uncertainty.injection import (
    check_episode_ranges,
    draw_episodes,
    scale_factor,
    signed,
)
from ts_uncertainty.sample import Sample, stack_samples

DYNAMICS_SHIFT_TYPES = ("timescale", "parameter")

# Continuous dysts flows that integrate reliably with LSODA, checked with check_systems()
# (T=2048, H=320). Left out: CellCycle, MacArthur, SprottF, SprottG, SprottH, SprottJerk, SprottL, StickSlipOscillator
# (integrator hangs or fails, or > 3 s per sample). None = all dysts flows.
DEFAULT_SYSTEMS: tuple[str, ...] | None = (
    "Aizawa", "AnishchenkoAstakhov", "Arneodo", "ArnoldBeltramiChildress", "ArnoldWeb",
    "AtmosphericRegime", "BeerRNN", "BelousovZhabotinsky", "BickleyJet", "Blasius",
    "BlinkingRotlet", "BlinkingVortex", "Bouali", "Bouali2", "BurkeShaw", "CaTwoPlus",
    "CaTwoPlusQuasiperiodic", "CellularNeuralNetwork", "Chen", "ChenLee", "Chua",
    "CircadianRhythm", "CoevolvingPredatorPrey", "Colpitts", "Coullet", "Dadras", "DequanLi",
    "DoubleGyre", "DoublePendulum", "Duffing", "ExcitableCell", "Finance", "FluidTrampoline",
    "ForcedBrusselator", "ForcedFitzHughNagumo", "ForcedVanDerPol", "GenesioTesi",
    "GlycolyticOscillation", "GuckenheimerHolmes", "Hadley", "Halvorsen", "HastingsPowell",
    "HenonHeiles", "HindmarshRose", "Hopfield", "HyperBao", "HyperCai", "HyperJha", "HyperLorenz",
    "HyperLu", "HyperPang", "HyperQi", "HyperRossler", "HyperWang", "HyperXu", "HyperYan",
    "HyperYangChen", "InteriorSquirmer", "IsothermalChemical", "ItikBanksTumor", "JerkCircuit",
    "KawczynskiStrizhak", "Laser", "LidDrivenCavityFlow", "LiuChen", "Lorenz", "Lorenz84",
    "Lorenz96", "LorenzBounded", "LorenzCoupled", "LorenzStenflo", "LuChen", "LuChenCheng",
    "MooreSpiegel", "MultiChua", "NewtonLiepnik", "NoseHoover", "NuclearQuadrupole",
    "OscillatingFlow", "PanXuZhou", "PehlivanWei", "Qi", "QiChen", "RabinovichFabrikant",
    "RayleighBenard", "RikitakeDynamo", "Rossler", "Rucklidge", "Sakarya", "SaltonSea",
    "SanUmSrisuchinwong", "ShimizuMorioka", "SprottA", "SprottB", "SprottC", "SprottD", "SprottE",
    "SprottI", "SprottJ", "SprottK", "SprottM", "SprottMore", "SprottN", "SprottO", "SprottP",
    "SprottQ", "SprottR", "SprottS", "SprottTorus", "SwingingAtwood", "Thomas", "ThomasLabyrinth",
    "Torus", "Tsucs2", "TurchinHanski", "VallisElNino", "WangSun", "WindmiReduced", "YuWang",
    "YuWang2", "ZhouChen",
)


class _Invalid(Exception):
    """Trajectory unusable (integration failed, diverged or collapsed): redraw."""


class ChaoticGenerator:
    """Infinite generator of clean chaotic time series from dysts flows.

    Args:
        context_length: T, number of observed time steps.
        horizon: H, number of time steps to forecast.
        systems: dysts flow names to draw from; None = DEFAULT_SYSTEMS or all dysts flows.
        dim: observed coordinate; None draws one per sample, an int picks it (clipped to the
            system dimension).
        pts_per_period_range: range of time steps per dominant period of the system; small
            values make the series look more irregular, large values smoother.
        burn_in_periods: dominant periods integrated and discarded so the trajectory starts on
            the attractor.
        ic_noise: relative perturbation of the dysts default initial condition.
        level_range: range of the constant level.
        slope_range: range of the linear trend slope per time step (default: no trend).
        amplitude_range: range of the scale of the z-normalised chaotic signal.
        shift: apply a dynamics shift at a random change point.
        shift_types: dynamics shift types to draw from, subset of DYNAMICS_SHIFT_TYPES.
        shift_position_range: range of the change point as a fraction of T + H.
        shift_scale_range: range of the scale factor for timescale shifts; with probability
            0.5 the reciprocal is used, so the system speeds up or slows down.
        shift_param_range: range of the relative change of one system parameter (sign is random).
        shift_recurring_prob: probability that a drawn shift is recurring.
        shift_duration_range: range of the length of each recurring episode as a fraction of T.
        rtol, atol: integrator tolerances.
        max_tries: redraws per sample before giving up.
        max_solve_seconds: wall-clock budget per integration; slower draws (e.g. a parameter
            shift that makes the system stiff) are rejected and redrawn.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        context_length: int,
        horizon: int,
        systems: Sequence[str] | None = None,
        dim: int | None = None,
        pts_per_period_range: tuple[float, float] = (20.0, 60.0),
        burn_in_periods: float = 20.0,
        ic_noise: float = 0.1,
        level_range: tuple[float, float] = (-1.0, 1.0),
        slope_range: tuple[float, float] = (0.0, 0.0),
        amplitude_range: tuple[float, float] = (0.5, 2.0),
        shift: bool = False,
        shift_types: Sequence[str] = DYNAMICS_SHIFT_TYPES,
        shift_position_range: tuple[float, float] = (0.5, 1.0),
        shift_scale_range: tuple[float, float] = (2.0, 4.0),
        shift_param_range: tuple[float, float] = (0.1, 0.3),
        shift_recurring_prob: float = 0.0,
        shift_duration_range: tuple[float, float] = (0.1, 0.25),
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_tries: int = 20,
        max_solve_seconds: float = 5.0,
        seed: int | None = None,
    ):
        if unknown := set(shift_types) - set(DYNAMICS_SHIFT_TYPES):
            raise ValueError(f"unknown shift types {unknown}, choose from {DYNAMICS_SHIFT_TYPES}")
        check_episode_ranges(shift_recurring_prob, shift_duration_range)
        if shift and not shift_types:
            raise ValueError("shift=True needs at least one shift type")

        if systems is None:
            systems = DEFAULT_SYSTEMS if DEFAULT_SYSTEMS is not None else all_flows()
        if unknown := [s for s in systems if not hasattr(flows, s)]:
            raise ValueError(f"unknown dysts systems {unknown}")

        self.context_length = context_length
        self.horizon = horizon
        self.systems = tuple(systems)
        self.dim = dim
        self.pts_per_period_range = pts_per_period_range
        self.burn_in_periods = burn_in_periods
        self.ic_noise = ic_noise
        self.level_range = level_range
        self.slope_range = slope_range
        self.amplitude_range = amplitude_range
        self.shift = shift
        self.shift_types = tuple(shift_types)
        self.shift_position_range = shift_position_range
        self.shift_scale_range = shift_scale_range
        self.shift_param_range = shift_param_range
        self.shift_recurring_prob = shift_recurring_prob
        self.shift_duration_range = shift_duration_range
        self.rtol = rtol
        self.atol = atol
        self.max_tries = max_tries
        self.max_solve_seconds = max_solve_seconds
        self.rng = np.random.default_rng(seed)
        self._models: dict[str, flows.DynSys] = {}

    def __iter__(self) -> Iterator[Sample]:
        return self

    def __next__(self) -> Sample:
        return self.sample()

    # ------------------------------------------------------------------ sampling

    def sample(
        self,
        system: str | None = None,
        shift_point: float | None = None,
        shift_type: str | None = None,
        recurring: bool | None = None,
        shift_episodes: Sequence[tuple[float, float]] | None = None,
    ) -> Sample:
        """Draw one clean sample; info holds system, dim and pts_per_period.

        Passing any shift argument forces a dynamics shift, even if the generator has shift=False.

        Args:
            system: dysts flow name; drawn from systems if None (also when retrying).
            shift_point: change point as a fraction of T + H in [0, 1].
            shift_type: one of DYNAMICS_SHIFT_TYPES; drawn from shift_types if None.
            recurring: True forces a recurring shift, False a single persistent one.
            shift_episodes: explicit (start, end) episodes as fractions of T + H.
        """
        last_error = None
        for _ in range(self.max_tries):
            try:
                return self._sample(system, shift_point, shift_type, recurring, shift_episodes)
            except _Invalid as e:
                last_error = e
        raise RuntimeError(f"no valid sample after {self.max_tries} tries, last error: {last_error}")

    def _sample(self, system, shift_point, shift_type, recurring, shift_episodes) -> Sample:
        rng = self.rng
        T = self.context_length
        length = T + self.horizon
        t = np.arange(length, dtype=np.float64)

        name = system if system is not None else str(rng.choice(self.systems))
        model = self._model(name)

        episodes: list[tuple[int, int]] = []
        active = np.zeros(length, dtype=bool)
        forced = shift_point is not None or shift_type is not None or recurring or shift_episodes is not None
        if self.shift or forced:
            episodes = draw_episodes(
                rng, T, self.horizon, self.shift_position_range, self.shift_recurring_prob, self.shift_duration_range,
                shift_point=shift_point, recurring=recurring, shift_episodes=shift_episodes,
            )
            for start, end in episodes:
                active[start:end] = True
            if shift_type is None:
                # systems without a scalar parameter can't take a parameter shift
                types = [s for s in self.shift_types if s != "parameter" or self._shiftable(model)]
                if not types:
                    raise ValueError(f"{name}: no applicable shift types, pass shift_type")
                shift_type = str(rng.choice(types))
            elif shift_type not in DYNAMICS_SHIFT_TYPES:
                raise ValueError(f"shift_type {shift_type} not in {DYNAMICS_SHIFT_TYPES}, use an Injector for the others")

        # ---- chaotic trajectory
        ppp = rng.uniform(*self.pts_per_period_range)
        step = np.full(length, model.period / ppp)  # integration time between consecutive samples

        shift_info: dict = {}
        params = model.param_list
        shifted_params = None
        if shift_type == "timescale":
            factor = scale_factor(rng, self.shift_scale_range)
            step = np.where(active, step * factor, step)
            shift_info = {"factor": factor}
        elif shift_type == "parameter":
            shifted_params, shift_info = self._perturb_params(model)

        x0 = self._initial_state(model, params)
        traj = self._integrate_piecewise(model, x0, step, params, shifted_params, active)
        if hasattr(model, "_postprocessing"):
            traj = np.stack(model._postprocessing(*traj.T), axis=-1)

        d = int(rng.integers(traj.shape[1])) if self.dim is None else min(self.dim, traj.shape[1] - 1)
        x = traj[:, d]
        self._check(x, active, shift_type)
        # z-normalise with the context statistics before the first shift (as a model would see it)
        ref = x[: min(T, episodes[0][0])] if episodes and episodes[0][0] >= T // 4 else x[:T]
        x = (x - ref.mean()) / ref.std()

        y = rng.uniform(*self.level_range) + rng.uniform(*self.slope_range) * t + rng.uniform(*self.amplitude_range) * x
        y = y.astype(np.float32)
        return Sample(
            context=y[:T],
            target=y[T:],
            shift_type=shift_type,
            shift_episodes=tuple(episodes),
            shift_info=shift_info,
            info={"system": name, "dim": d, "pts_per_period": float(ppp)},
        )

    def sample_batch(self, batch_size: int) -> dict[str, np.ndarray | None]:
        """Stack batch_size clean samples, see stack_samples; plus 'system' (B,)."""
        samples = [self.sample() for _ in range(batch_size)]
        return stack_samples(samples) | {"system": np.array([s.info["system"] for s in samples])}

    # ------------------------------------------------------------------ dynamics

    def _model(self, name: str):
        if name not in self._models:
            model = getattr(flows, name)()
            model.rhs(np.asarray(model.ic, dtype=np.float64), 0.0)  # trigger numba compilation outside the time budget
            self._models[name] = model
        return self._models[name]

    def _rhs(self, model, params):
        """Right-hand side with the given parameter values (param_list order = sorted keys).

        Works for systems implementing _rhs (uses param_list) and for systems overriding rhs
        (use the parameter attributes): both are set on a shallow copy of the model.
        """
        if params is not model.param_list:
            model = copy.copy(model)
            keys = sorted(model.params.keys())
            model.params = dict(zip(keys, params))
            model.__dict__.update(model.params)
            model.param_list = list(params)

        deadline = time.perf_counter() + self.max_solve_seconds
        n_calls = 0

        def f(t, X):
            nonlocal n_calls
            n_calls += 1
            if n_calls % 1000 == 0 and time.perf_counter() > deadline:
                raise _Invalid(f"{model.name}: integration exceeded {self.max_solve_seconds} s (stiff?)")
            return np.asarray(model.rhs(X, t), dtype=np.float64)

        return f

    def _solve(self, model, params, x0, t_eval):
        """Integrate from (t_eval[0], x0), return states at t_eval (len(t_eval), D)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_ivp(
                self._rhs(model, params), (t_eval[0], t_eval[-1]), x0, t_eval=t_eval,
                method="LSODA", rtol=self.rtol, atol=self.atol,
            )
        if sol.status != 0 or sol.y.shape[1] != len(t_eval) or not np.isfinite(sol.y).all():
            raise _Invalid(f"{model.name}: integration failed ({sol.message})")
        return sol.y.T

    def _initial_state(self, model, params) -> np.ndarray:
        """Perturbed default initial condition, burned in onto the attractor."""
        ic = np.asarray(model.ic, dtype=np.float64)
        ic = ic * (1 + self.ic_noise * self.rng.standard_normal(ic.shape)) + 1e-3 * self.rng.standard_normal(ic.shape)
        t_burn = self.burn_in_periods * model.period
        return self._solve(model, params, ic, np.array([0.0, t_burn]))[-1]

    def _integrate_piecewise(self, model, x0, step, params, shifted_params, active) -> np.ndarray:
        """Integrate over the sample grid, switching to shifted_params while active (if given).

        Sample k is at time sum(step[:k]); a timescale shift just changes the step sizes.
        """
        length = len(step)
        times = np.concatenate([[0.0], np.cumsum(step[:-1])])
        if shifted_params is None:
            return self._solve(model, params, x0, times)
        # segments of constant regime
        bounds = np.flatnonzero(np.diff(active.astype(int))) + 1
        edges = [0, *bounds.tolist(), length]
        out = np.empty((length, len(x0)))
        state, t_prev = x0, times[0]
        for a, b in itertools.pairwise(edges):
            p = shifted_params if active[a] else params
            if a == 0:
                seg = self._solve(model, p, state, times[a:b]) if b > 1 else state[None]
            else:
                seg = self._solve(model, p, state, np.concatenate([[t_prev], times[a:b]]))[1:]
            out[a:b] = seg
            state, t_prev = seg[-1], times[b - 1]
        return out

    @staticmethod
    def _shiftable(model) -> list[int]:
        """Indices into param_list of scalar, non-zero parameters."""
        keys = sorted(model.params.keys())
        return [i for i, k in enumerate(keys) if np.ndim(model.params[k]) == 0 and model.params[k] != 0]

    def _perturb_params(self, model) -> tuple[list, dict]:
        """Perturb one scalar, non-zero parameter by a relative amount from shift_param_range."""
        keys = sorted(model.params.keys())  # param_list is ordered by sorted keys
        candidates = self._shiftable(model)
        if not candidates:
            raise ValueError(f"{model.name} has no scalar parameter, a parameter shift is not possible")
        i = int(self.rng.choice(candidates))
        params = list(model.param_list)
        old = float(params[i])
        params[i] = old * (1 + signed(self.rng, self.shift_param_range))
        return params, {"param": keys[i], "old": old, "new": float(params[i])}

    def _check(self, x: np.ndarray, active: np.ndarray, shift_type: str | None):
        """Reject diverging or collapsed trajectories, also inside a parameter shift."""
        T = self.context_length
        ctx_std = x[:T].std()
        if not np.isfinite(x).all() or ctx_std < 1e-6 * (np.abs(x[:T]).mean() + 1e-12):
            raise _Invalid("collapsed or non-finite trajectory")
        if shift_type == "parameter" and active.any():
            normal, shifted = x[~active], x[active]
            spread = normal.max() - normal.min()
            if np.abs(shifted - normal.mean()).max() > 10 * spread:
                raise _Invalid("parameter shift diverged")
            # collapse onto a fixed point: last stretch of the shifted regime is (nearly) constant
            tail = shifted[-max(len(shifted) // 3, 2):]
            if len(shifted) > 20 and tail.std() < 1e-3 * normal.std():
                raise _Invalid("parameter shift collapsed onto a fixed point")


def all_flows() -> tuple[str, ...]:
    """Names of all continuous-time chaotic ODE systems in dysts (delay equations excluded)."""
    from dysts.base import DynSysDelay
    from dysts.systems import get_attractor_list

    return tuple(n for n in get_attractor_list() if not issubclass(getattr(flows, n), DynSysDelay))


def check_systems(
    context_length: int = 2048, horizon: int = 320, n: int = 3, max_seconds: float = 5.0, seed: int = 0
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Try every dysts flow; return (usable systems, {rejected system: reason}).

    A system is usable if n samples (with a parameter shift, the hardest case) can be drawn
    and one sample takes at most max_seconds. Use the result as `systems` / DEFAULT_SYSTEMS.
    """
    import time

    ok, bad = [], {}
    for name in all_flows():
        gen = ChaoticGenerator(context_length, horizon, systems=[name], max_tries=5, seed=seed)
        try:
            t0 = time.perf_counter()
            for i in range(n):
                gen.sample(system=name, shift_type="parameter" if i == 0 else None)
            dt = (time.perf_counter() - t0) / n
            if dt > max_seconds:
                bad[name] = f"slow ({dt:.1f}s per sample)"
            else:
                ok.append(name)
        except Exception as e:  # noqa: BLE001
            bad[name] = f"{type(e).__name__}: {e}"[:200]
    return tuple(ok), bad