"""pytest plugin: run any g6_clone suite with the stagger_placement rule installed, without editing the op.
    PYTHONPATH=<dir of this file> PLACE_ROOT=<dir of placement.py> PLACE=rule pytest -p stagger_plugin ...
"""
import importlib.util
import os
from pathlib import Path


_MOD = None


def pytest_configure(config):
    global _MOD
    place = os.environ.get("PLACE", "rule")
    spec = importlib.util.spec_from_file_location("stagger_placement", Path(os.environ["PLACE_ROOT"]) / "placement.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"\nSTAGGER_PLUGIN installed {mod.install(place)}", flush=True)
    _MOD = mod


def pytest_unconfigure(config):
    if _MOD is not None:
        seen = sorted(set(_MOD.REPLACED))
        print(f"\nSTAGGER_PLUGIN re-placed {len(_MOD.REPLACED)} plans, distinct {seen}", flush=True)
