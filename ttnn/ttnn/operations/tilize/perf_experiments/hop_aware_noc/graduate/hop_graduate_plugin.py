"""pytest plugin (hop_aware_noc): run any tilize test against the GRADUATION CANDIDATE.

PYTHONPATH=<this dir> scripts/run_safe_pytest.sh <test> -p hop_graduate_plugin ...

Overlays graduate/tilize_program_descriptor.py onto the op's own descriptor module (exec into
ttnn.operations.tilize.tilize_program_descriptor's namespace, KERNEL_DIR = graduate/kernels), so a
test's monkeypatch of pd.<KNOB> still reaches the graduate code; rebinds the names tilize.py imported
from it. Never touches the op files.
HOP_GRADUATE_COUNT=1: print how many programs engaged the hop path (writer define present), by store path.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_configure(config):
    import ttnn.operations.tilize  # noqa: F401  (imports the op module)

    pd = sys.modules["ttnn.operations.tilize.tilize_program_descriptor"]
    tilize_mod = sys.modules["ttnn.operations.tilize.tilize"]
    path = os.path.join(HERE, "tilize_program_descriptor.py")
    pd.__dict__["__file__"] = path
    exec(compile(open(path).read(), path, "exec"), pd.__dict__)
    for name in ("create_program_descriptor", "TILE_WIDTH", "PadSpec", "_tile_grid"):
        setattr(tilize_mod, name, getattr(pd, name))
    print(f"HOP_GRADUATE: tilize_program_descriptor overlaid from {path} (kernels {pd.KERNEL_DIR})")

    if os.environ.get("HOP_GRADUATE_COUNT") == "1":
        import atexit

        import ttnn

        counts = {"hop_sub_block": 0, "hop_store_rows": 0, "off": 0}
        orig = ttnn.KernelDescriptor

        def kd(*a, **kw):
            if str(kw.get("kernel_source", "")).endswith("tilize_writer.cpp"):
                names = {d[0] for d in kw.get("defines", [])}
                if "TILIZE_HOP_WRITE_MIN_SAVING" not in names:
                    counts["off"] += 1
                else:
                    counts["hop_sub_block" if "TILIZE_SUB_BLOCK_TILES" in names else "hop_store_rows"] += 1
            return orig(*a, **kw)

        ttnn.KernelDescriptor = kd
        atexit.register(lambda: print(f"HOP_GRADUATE_COUNTS {counts}"))
