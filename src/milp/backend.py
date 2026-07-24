"""MILP backend abstraction layer for PuLP-based exact solvers (CBC / CPLEX)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - depends on environment
    import pulp
except Exception:  # pragma: no cover
    pulp = None


class _PulpGRB:
    """GRB-style status and variable-type constants used by existing MILP code."""

    BINARY = "Binary"
    INTEGER = "Integer"
    CONTINUOUS = "Continuous"

    MINIMIZE = 1
    MAXIMIZE = -1

    OPTIMAL = 2
    FEASIBLE = 1
    SUBOPTIMAL = 12
    TIME_LIMIT = 9
    INTERRUPTED = 11


@dataclass
class _PulpParams:
    """Solver parameter container compatible with the previous project's Param pattern."""

    OutputFlag: int = 0
    TimeLimit: int | float = 3600
    MIPGap: float = 0.0
    Threads: int = 1


class _PulpModel:
    """Minimal model object exposing the methods used by the project."""

    def __init__(self, name: str):
        if pulp is None:
            raise RuntimeError("PULP is not installed. Install `pulp` to enable CBC backend.")

        self._problem = pulp.LpProblem(name=name, sense=pulp.LpMinimize)
        self._objective = None
        self.Params = _PulpParams()
        self.Status = 0
        self.SolCount = 0
        self.ObjVal = math.nan
        self.MIPGap = math.nan
        self.BestBound = math.nan

    def addVar(
        self,
        vtype: str | None = None,
        lb: float = 0.0,
        ub: float | None = None,
        name: str = "",
    ):
        if pulp is None:
            raise RuntimeError("PULP is not installed")
        cat = pulp.LpContinuous
        if vtype == _PulpGRB.BINARY:
            cat = pulp.LpBinary
        elif vtype == _PulpGRB.INTEGER:
            cat = pulp.LpInteger
        return pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=cat)

    def addConstr(self, expr: Any, name: str | None = None) -> None:
        if name is not None:
            self._problem += expr, str(name)
        else:
            self._problem += expr

    def setObjective(self, expr: Any, sense: int) -> None:
        if sense == _PulpGRB.MINIMIZE:
            self._problem.sense = pulp.LpMinimize
        elif sense == _PulpGRB.MAXIMIZE:
            self._problem.sense = pulp.LpMaximize
        else:
            self._problem.sense = pulp.LpMinimize
        self._objective = expr
        self._problem.setObjective(expr)

    def optimize(self) -> None:
        if pulp is None:
            raise RuntimeError("PULP is not installed")
        time_limit = self.Params.TimeLimit if self.Params.TimeLimit > 0 else None
        threads = self.Params.Threads if self.Params.Threads and self.Params.Threads > 0 else None
        gap = self.Params.MIPGap if self.Params.MIPGap and self.Params.MIPGap > 0 else None
        solver = None
        if _ACTIVE_PULP_SOLVER == "cplex":
            if hasattr(pulp, "CPLEX_PY"):
                try:
                    solver = pulp.CPLEX_PY(
                        msg=bool(self.Params.OutputFlag),
                        timeLimit=time_limit,
                        threads=threads,
                        gapRel=gap,
                    )
                except TypeError:
                    solver = pulp.CPLEX_PY(
                        msg=bool(self.Params.OutputFlag),
                        timeLimit=time_limit,
                    )
            elif hasattr(pulp, "CPLEX_CMD"):
                try:
                    solver = pulp.CPLEX_CMD(
                        msg=bool(self.Params.OutputFlag),
                        timeLimit=time_limit,
                        options=([] if gap is None else [f"mip tolerances mipgap {gap}"]),
                    )
                except TypeError:
                    solver = pulp.CPLEX_CMD(msg=bool(self.Params.OutputFlag))
            else:
                raise RuntimeError("cplex backend requested but PuLP CPLEX interfaces are unavailable.")
        else:
            # NOTE:
            # On some Windows environments with bundled CBC (2.10.3), passing
            # the `-threads` flag can trigger severe stalls even on tiny MIPs.
            # We therefore do not pass threads to CBC here.
            cbc_threads = None
            solver = pulp.PULP_CBC_CMD(
                msg=bool(self.Params.OutputFlag),
                timeLimit=time_limit,
                threads=cbc_threads,
                gapRel=gap,
            )
        self._problem.solve(solver)

        # Prefer native CPLEX status/metrics when available, because PuLP may
        # report "Optimal" for time-limited MIP runs that only found an incumbent.
        if _ACTIVE_PULP_SOLVER == "cplex" and hasattr(self._problem, "solverModel") and self._problem.solverModel is not None:
            cpx = self._problem.solverModel
            try:
                cpx_status = int(cpx.solution.get_status())
            except Exception:
                cpx_status = None

            # Common CPLEX status mapping for MIP.
            if cpx_status in {101, 102}:
                self.Status = _PulpGRB.OPTIMAL
            elif cpx_status in {107, 108, 109, 110, 111, 112}:
                self.Status = _PulpGRB.TIME_LIMIT
            elif cpx_status in {103, 104, 105, 106}:
                self.Status = 0
            else:
                self.Status = 0

            try:
                self.ObjVal = float(cpx.solution.get_objective_value())
                self.SolCount = 1
            except Exception:
                self.ObjVal = math.nan
                self.SolCount = 0

            try:
                self.MIPGap = float(cpx.solution.MIP.get_mip_relative_gap())
            except Exception:
                self.MIPGap = (0.0 if self.Status == _PulpGRB.OPTIMAL else math.nan)
            try:
                self.BestBound = float(cpx.solution.MIP.get_best_objective())
            except Exception:
                self.BestBound = float(self.ObjVal) if self.Status == _PulpGRB.OPTIMAL and self.SolCount > 0 else math.nan
            return

        status_name = pulp.LpStatus[self._problem.status]
        status_lower = status_name.lower()
        if status_lower == "optimal":
            self.Status = _PulpGRB.OPTIMAL
        elif status_lower in {"not solved", "suboptimal"}:
            self.Status = _PulpGRB.TIME_LIMIT
        elif status_lower in {"infeasible", "unbounded", "undefined"}:
            self.Status = 0
        elif status_lower == "integer infeasible":
            self.Status = 0
        else:
            self.Status = 0

        self.SolCount = 1 if any(v.varValue is not None for v in self._problem.variables()) else 0
        if self.SolCount and self._problem.objective is not None:
            self.ObjVal = float(pulp.value(self._problem.objective))
        else:
            self.ObjVal = math.nan

        if self.Status in {_PulpGRB.OPTIMAL, _PulpGRB.FEASIBLE, _PulpGRB.SUBOPTIMAL}:
            self.MIPGap = 0.0
            self.BestBound = float(self.ObjVal) if self.SolCount > 0 else math.nan
        else:
            self.MIPGap = math.nan
            self.BestBound = math.nan


class _PulpBackend:
    """Minimal PuLP-compatible object exposing the same surface used by the code."""

    Model = _PulpModel
    GRB = _PulpGRB

    @staticmethod
    def quicksum(values):
        if pulp is None:
            raise RuntimeError("PULP is not installed.")
        return pulp.lpSum(values)


_ACTIVE_PULP_SOLVER = "pulp_cbc"


def _build_backend(name: str):
    global _ACTIVE_PULP_SOLVER
    key = (name or "pulp_cbc").strip().lower()
    if key in {"pulp", "cbc", "pulp_cbc"}:
        if pulp is None:
            raise RuntimeError("pulp backend requested but pulp is not installed.")
        _ACTIVE_PULP_SOLVER = "pulp_cbc"
        return _PulpBackend()
    if key in {"cplex", "cplex_py"}:
        if pulp is None:
            raise RuntimeError("cplex backend requested but pulp is not installed.")
        _ACTIVE_PULP_SOLVER = "cplex"
        return _PulpBackend()
    if key in {"gurobi", "gurobi_persistent", "scip"}:
        raise RuntimeError(
            f"backend '{key}' is not available in this build. Set milp.solver to 'pulp_cbc' "
            "or add a dedicated backend adapter."
        )
    raise ValueError(f"Unsupported MILP backend: {name}")


_ACTIVE_BACKEND = _build_backend("pulp_cbc")


def set_active_backend(name: str):
    """Select the backend used by subsequent model construction."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = _build_backend(name)


def get_active_backend():
    """Return currently configured backend object."""
    return _ACTIVE_BACKEND


class _BackendProxy:
    """Dynamic attribute proxy so `from .backend import gp` follows active backend."""

    def __getattr__(self, name: str):
        return getattr(get_active_backend(), name)


gp = _BackendProxy()
