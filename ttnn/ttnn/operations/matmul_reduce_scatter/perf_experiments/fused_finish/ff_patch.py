"""pytest plugin (-p ff_patch, with this dir on PYTHONPATH): route the real op's program factory to the fused_finish
experiment copy mrs_ff (its kernels; MRS_FF picks the compute variant, default 5 = the final candidate). Every call's
plan (incl. lag_entries) is appended to results/ff_patch_calls.txt so a run proves it went through mrs_ff."""
import os

import mrs_ff.matmul_reduce_scatter_program_descriptor as _mine
import ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter_program_descriptor as _real

_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ff_patch_calls.txt")


def _run(a, w, **kw):
    out = _mine.run(a, w, **kw)
    with open(_LOG, "a") as f:
        f.write(
            f"{tuple(a.shape)} {tuple(w.shape)} sd={kw.get('scatter_dim')} topo={kw.get('topology')} "
            f"links={kw.get('num_links')} MRS_FF={os.environ.get('MRS_FF', '5')} {kw['hooks']._LAST_PLAN}\n"
        )
    return out


_real.run = _run
