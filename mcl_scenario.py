"""Szenario: Reglerwerte (Settings) und das daraus erzeugte Turnier samt Punktedifferenz."""

from dataclasses import dataclass

import numpy as np

import mcl_constants as C
import mcl_margins as MG
import turnier_generator as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    delta: float = C.DEFAULT_DELTA
    k: int = C.DEFAULT_K
    plan: str = "staffeln"
    x: int = C.DEFAULT_X
    p: float = C.DEFAULT_P
    c: float = C.DEFAULT_CYCLE
    stil: str = "ausgewogen"
    q: float = C.DEFAULT_Q
    h: int = C.DEFAULT_H
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Run:
    turnier: T.Turnier
    D: np.ndarray        # Punktedifferenz je Paar (antisymmetrisch)


def make_run(s: Settings) -> Run:
    """Deterministisch aus den Reglerwerten: gleiche Eingabe, gleiches Turnier und gleiche Punktedifferenz."""
    t = T.build_turnier(s.n, s.delta, s.k, s.plan, s.x, s.p, s.c, s.stil, s.seed)
    D = MG.point_differential(t.wins, s.q, s.h, T.SplitMix64(s.seed ^ 0x3A3A))
    return Run(t, D)
