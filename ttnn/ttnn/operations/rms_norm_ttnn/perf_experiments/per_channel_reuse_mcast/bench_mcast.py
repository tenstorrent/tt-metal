# Perf experiment `per_channel_reuse_mcast` -- STEP 2: the candidate.
#
#   base        the shipped op, kernels copied byte-for-byte into k_base
#   pd_off      the experiment's descriptor with the multicast switched OFF
#               (proves the copied descriptor + copied kernels are neutral)
#   mcast       one injector per grid COLUMN reads the per-channel slice from
#               DRAM and multicasts it down the column
#   ablate      the per-channel DRAM read deleted -- the UPPER BOUND no
#               multicast can beat.  Output is intentionally wrong (perf only).
import importlib.util
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes"))

import bench_r3  # noqa: E402
import ttnn  # noqa: E402
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD  # noqa: E402

# The package __init__ re-exports the FUNCTION under the submodule's name, so a
# plain `import ... .rms_norm_ttnn as OP` binds the function.  Go through sys.modules.
import importlib  # noqa: E402

OP = importlib.import_module("ttnn.operations.rms_norm_ttnn.rms_norm_ttnn")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PDM = _load("pd_mcast", HERE / "pd_mcast.py")
SHIPPED = OP.create_program_descriptor
SHIPPED_KDIR = PD.KERNEL_DIR


def _use_shipped(kdir):
    OP.create_program_descriptor = SHIPPED
    PD.KERNEL_DIR = kdir


def _use_mcast(on, handshake=True, inj_trim=None, batch=False):
    OP.create_program_descriptor = PDM.create_program_descriptor
    PDM.PC_MCAST_ENABLE = on
    PDM.PC_MCAST_HANDSHAKE = handshake
    PDM.PC_MCAST_INJECTOR_TRIM = inj_trim
    PDM.PC_MCAST_BATCH_READS = batch
    PDM.KERNEL_DIR = HERE / "k_mcast"


VARIANTS = {
    "base": lambda: _use_shipped(HERE / "k_base"),
    "pd_off": lambda: _use_mcast(False),
    "mcast": lambda: _use_mcast(True),
    "mcast_nh": lambda: _use_mcast(True, handshake=False),
    "mcast_wt": lambda: _use_mcast(True, inj_trim=0),
    "mcast_wtnh": lambda: _use_mcast(True, handshake=False, inj_trim=0),
    "mcast_wtb": lambda: _use_mcast(True, inj_trim=0, batch=True),
    "mcast_wtbnh": lambda: _use_mcast(True, handshake=False, inj_trim=0, batch=True),
    "ablate": lambda: _use_shipped(HERE / "k_ablate"),
}

DEFAULT_VARIANTS = ["base", "mcast", "ablate"]
DEFAULT_CASES = ["G4_blk7168", "G3_blk8192", "P1_int1024_g", "P2_int1024_gb", "P6_int7168_g", "G1_w7168_g28"]


def main():
    cases = (os.environ.get("RMS_CASES") or ",".join(DEFAULT_CASES)).split(",")
    labels = (os.environ.get("RMS_VARIANTS") or ",".join(DEFAULT_VARIANTS)).split(",")
    reps = int(os.environ.get("RMS_REPS", "1"))
    res = {}
    device = ttnn.open_device(device_id=0)
    try:
        for _ in range(reps):
            for label in labels:
                VARIANTS[label]()
                for case in cases:
                    ns, p, r = bench_r3.measure(device, case)
                    res.setdefault((case, label), []).append((ns, p, r))
                    print(f"RESULT.raw {case:16s} {label:8s} ns={ns:10.0f} pcc={p:.6f} relrms={r:.3e}", flush=True)
    finally:
        _use_shipped(SHIPPED_KDIR)
        ttnn.close_device(device)

    print("RESULT " + f"{'case':16s}" + "".join(f"{l:>11s}" for l in labels))
    for case in cases:
        row = f"{case:16s}"
        for label in labels:
            row += f"{min(x[0] for x in res[(case, label)]):11.0f}"
        print("RESULT " + row)
    print("RESULT --- speedup vs " + labels[0] + " ---")
    for case in cases:
        b = min(x[0] for x in res[(case, labels[0])])
        row = f"{case:16s}"
        for label in labels:
            row += f"{b / min(x[0] for x in res[(case, label)]):11.3f}"
        pcs = " ".join(f"{l}:{min(x[1] for x in res[(case, l)]):.6f}" for l in labels)
        print("RESULT " + row + "  pcc " + pcs)


if __name__ == "__main__":
    main()
