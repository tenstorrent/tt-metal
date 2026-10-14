"""pytest plugin: route the real op's tests through this experiment's descriptor with a NoC role swap.
usage: run_safe_pytest.sh <tilize unit test> -p ttnn.operations.tilize.perf_experiments.geo_write_noc.geo_plugin
GEO_PLUGIN_SWAP (default "ly % 4 == 0") selects the swapped cores."""
import importlib
import os


def pytest_configure(config):
    real = importlib.import_module("ttnn.operations.tilize.tilize")
    exp = importlib.import_module("ttnn.operations.tilize.perf_experiments.geo_write_noc.tilize_program_descriptor")
    exp.GEO_SWAP = os.environ.get("GEO_PLUGIN_SWAP", "ly % 4 == 0")
    real.create_program_descriptor = exp.create_program_descriptor
    print(f"GEO_PLUGIN active: swap={exp.GEO_SWAP!r}")
