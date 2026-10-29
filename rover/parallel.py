"""Run many simulations across processes, one PyBullet client per worker."""
from __future__ import annotations

import os
from multiprocessing import Pool

_SIM = None


def _init():
    global _SIM
    from .sim import RoverSim
    _SIM = RoverSim()


def _call(args):
    fn, payload = args
    return fn(_SIM, payload)


def pmap(fn, payloads, procs=None):
    """fn(sim, payload) -> result, evaluated in worker processes; yields in order."""
    procs = procs or os.cpu_count() or 1
    with Pool(procs, initializer=_init) as pool:
        yield from pool.imap(_call, [(fn, p) for p in payloads])
