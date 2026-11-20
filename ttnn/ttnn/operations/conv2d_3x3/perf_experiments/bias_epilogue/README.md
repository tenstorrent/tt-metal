# bias_epilogue (Perf 17): the bias pass of the K-blocked walk

Idea: stop running the bias add as a separate pass after the K walk of every M-block.

## Layout
- `op_base/`: the working-tree op at HEAD ec60e2fa9f2 (with the uncommitted user changes), plus the
  `ablation_r17/apply_hooks.py` stubs (`CONV2D_3X3_ABLATE`) and a `LAST_GB` blocking record. This is the baseline.
- `op_copy/`: the menu. `kernels/conv2d_3x3_bias_epilogue.hpp` holds the walk with every mode. The descriptor knob
  `BIAS_EPI` picks the mode: 0 off, 1/2 Add, 3 Seed, 4 Seed + exact reload (`UnpackToDestFp32` on the partials CB),
  5 Interleave with `BIAS_EPI_G` row groups per batch. Modes 2..5 also send the no-bias walk through the copy reload.
  It also has the ablation hooks and a `LAST_EPI` path record. One comment line was added at the top of the reader
  only to move its zone source lines off a profiler zone-hash collision.
- `op_final/`: the recommended change applied to the unmodified working-tree op, with no hooks and no knobs:
  `kernels/conv2d_3x3_kblock_walk.hpp` plus the walk branch of `conv2d_3x3_compute.cpp`.
  - `integration.patch` = `diff -ruN <real op> op_final`. Apply it with
    `patch -p1 -d ttnn/ttnn/operations/conv2d_3x3 < integration.patch` (dry-run checked against the working tree).
  - `variants.patch` = `diff -ruN op_base op_copy` (the full menu, for reference).

## Re-run (from the repo root)
- `run.sh <label> <cases> <variants> [ENV=..]` runs `test_bias_epi.py` under `--profile`. The device ns per
  `case|variant` go to `runs/<label>.txt`. PCC, mean / max abs error vs torch, sha1, bit-identity and ulp vs the case's
  first variant go to `runs/<label>.jsonl`.
  - Case syntax: a FLUX bh2x4 idx (31..47) plus optional suffixes `n` (no bias), `r` (ReLU), `t` (TILE input).
    Alternatively an `EXTRA` name (`u512_256`, `u1024_128`, `d512_256s2`, `d1024_128s2`, `fd16`, `s2rm`, `fdk1`) with
    optional `.n` / `.r`.
  - Variant syntax: `base`, `rec` (`op_final`), `epi` (Add), `epi3` / `epi4` (Seed / Seed + exact), `ilv1`, `ilv2`,
    `ilv4`, `ilv8`, `ilvA` (Interleave with G = 1, 2, 4, 8 or the whole block). Append `@fill` to stub the fill
    (`CONV2D_3X3_ABLATE=fill`, `op_base` / `op_copy` only).
  - `BE_ITERS=3` measures 3 calls per variant. `BE_ORDER=alt` reverses the variant order on every other case.
- `run_pan.sh <label> <indices> <variants>` runs `test_panoptic_be.py`. Indices 0..44 are the non-FLUX LOOSE cases;
  23..44 are the `@ LoFi [perf]` Panoptic gate. Indices 45..52 are HEIGHT-sharded ROW_MAJOR cells that take the
  one-K-block walk.
- `safe_run.sh` gives up with exit code 75 (DEFERRED) if the device lock is not acquired within `BE_MAXWAIT` s
  (default 300). That way the caller's time limit never kills a run that is holding the device.
