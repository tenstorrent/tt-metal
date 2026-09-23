"""pytest plugin (hop_aware_coread): run any tilize test against the GRADUATION CANDIDATE.

PYTHONPATH=<this dir> scripts/run_safe_pytest.sh -p hac_graduate_plugin <test> ...
Swaps ttnn.operations.tilize.tilize's create_program_descriptor for graduate/
tilize_program_descriptor.py's (whose KERNEL_DIR is graduate/kernels). Never touches the op files.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_configure(config):
    import ttnn.operations.tilize  # noqa: F401  (imports the op module)

    tilize_mod = sys.modules["ttnn.operations.tilize.tilize"]
    spec = importlib.util.spec_from_file_location(
        "tilize_pd_graduate", os.path.join(HERE, "tilize_program_descriptor.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tilize_mod.create_program_descriptor = mod.create_program_descriptor
    sys.modules["tilize_pd_graduate"] = mod
    print(f"HAC_GRADUATE: create_program_descriptor -> {mod.__file__} (kernels {mod.KERNEL_DIR})")

    # Count programs by co-read mode (HAC_GRADUATE_COUNT=1): positional vs geometric (CO_READ_LISTED).
    if os.environ.get("HAC_GRADUATE_COUNT") == "1":
        import atexit
        import ttnn

        counts = {"listed": 0, "positional": 0, "off": 0}
        orig = mod.ttnn.KernelDescriptor

        def kd(*a, **kw):
            src = str(kw.get("kernel_source", ""))
            if src.endswith("tilize_reader.cpp"):
                ct = list(kw.get("compile_time_args", []))
                listed = any(d[0] == "CO_READ_LISTED" for d in kw.get("defines", []))
                counts["listed" if listed else ("positional" if ct[28] else "off")] += 1
            return orig(*a, **kw)

        ttnn.KernelDescriptor = kd
        atexit.register(lambda: print(f"HAC_GRADUATE_COUNTS {counts}"))
