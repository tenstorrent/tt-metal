# resident_block_count — idea A: buy MORE row-blocks per core with a SMALLER block

Isolated bake-off for the still-open half of design lamp **L-OVERLAP**: the RESIDENT
regime search only ever evaluates `br = min(max_rows, brmax)` (the LARGEST block that
fits) at each ring depth, and never a smaller `br` at the same depth.

## How it is isolated (nothing here touches the shipped op)

* `desc_fork.py` — a byte copy of `rms_norm_ttnn_program_descriptor.py` with ONE
  edit: the RESIDENT block/depth pick is env-driven (search for `IDEA A`).
* `kernels/` — a byte copy of the shipped kernels. `KERNEL_DIR` is
  `Path(__file__).parent / "kernels"`, so the fork picks these up, which also gives
  every variant here a kernel PATH distinct from the op's (the JIT cache key hashes
  the path, not the source content).
* `bench.py` — imports the fork and monkeypatches it into the op module in place of
  `create_program_descriptor`; cases come from `feature_spec.LOOSE_CASES`
  (`group == "perf"`) plus a synthetic regime sweep. `RMS_RBC=shipped` reproduces the
  shipped D41 pick byte for byte, so the baseline is the op's current rule.
* `run.sh "<tag>" [ENV=VAL ...]` — one foreground `tt-probe.sh` session.
* `patch_search.py` — the one-shot source patch that produced `desc_fork.py`'s hook
  (kept for provenance; already applied).

Precision is FIXED per case (`math_fidelity`, `fp32_dest_acc_en`, dtypes straight out
of feature_spec) and identical for every variant. Measured pcc never moved between
variants on any case (focus 0.999985 vs a 0.9995 soft gate).

## Measurement discipline

Device kernel ns is read in-process (`DEVICE KERNEL DURATION [ns]`). Variants are
INNER-looped per case so they see bit-identical tensors, and every list starts with a
`burn` variant plus alternates base/candidate (ABBA): the FIRST measured variant in a
session reads ~1.5-2.2% slow (verified on cases whose plan does not change at all, so
the two variants are the same program), and the burn absorbs that ramp.

```bash
./run.sh focus RMS_CASES=4 RMS_READS=3 RMS_TRACE_BLOCKING=1 \
  RMS_VARIANTS='burn:RMS_RBC=shipped;base:RMS_RBC=shipped;cand:RMS_RBC=ia_bal;base2:RMS_RBC=shipped'
```

## The rules the fork can select (`RMS_RBC=`)

| value | rule |
|---|---|
| `shipped` | the D41 pick: per depth take the largest fitting `br`, then the depth with the most row-blocks, tie-break shallowest |
| `maxblocks` | sweep `br` 1..brmax at every depth, take the most row-blocks (degenerates to `br = 1`) |
| `shallow` | `maxblocks` restricted to the shallowest feasible depth |
| `half` | `br = clamp(rows_min // 2, 1, brmax)` — the finest balanced split that still gives every core two blocks |
| **`ia`** | x read over the NoC → `br = 1`; x resident in L1 (`native_in`) → shipped coarsest. Shallowest depth. |
| **`ia_bal`** | `ia` + on the resident path, keep the shipped BLOCK COUNT but balance it when the balanced block divides the core's rows exactly (`20+12 → 16+16`) |

Also `RMS_RBC_BR` / `RMS_RBC_DEPTH` (hard force, for the grid probes),
`RMS_RBC_DEPTHS` (widen the depth ladder), `RMS_RBC_MAXB` (cap the block count).

## Measured (blackhole p150b, 110-core grid, this session)

Focus `(1,1,8192,1024)` interleaved bf16 HiFi2, aggregated over 6 sessions:
`shipped br=2 depth=3` **83,997 ns** (n=20, sd 600) vs `br=1 depth=2` **82,889 ns**
(n=17, sd 390) → **1.013x**, ~6.7 sigma. Depth is irrelevant once `br=1`
(d2 82,924 / d3 82,961 / d4 83,153 / d5 82,570 / d6 82,713 / d8 82,673).

`br` grids (median of 3-5, `RMS_RBC_BR` forced):

| case | plan | br → ns |
|---|---|---|
| 04 `(1,1,8192,1024)` INTER, rows_max 3 | interleaved | 3→84,173 · 2→83,997 (base) · 1→**82,889** |
| 29 `(1,1,32768,1024)` INTER, rows_max 10 | interleaved | 2→312,500 (base) · 1→311,600 (flat) |
| 30 `(1,1,65536,1024)` INTER, rows_max 19 | interleaved | 2→620,200 (base) · 1→623,371 (flat, spread 5,244) |
| 12 `(1,1,8192,1024)` BLOCK 1024x128, rows 32 | native, brmax 20 | 20→20,372 (base) · **16→19,497** · 11→21,195 · 8→22,361 · 6→26,079 · 4→29,656 · 2→47,394 · 1→40,618 |
| 17 `(1,1,7168,1024)` BLOCK 896x128 gbr, rows 28 | native, brmax 11 | 11→28,880 (base) · 10→29,083 · 7→31,659 · 4→40,251 · 1→69,566 |
| 31 `(1,1,6144,1024)` BLOCK 768x128, rows 24 | native, brmax 24 | 24→14,432 (base) · 12→15,996 · 8→17,299 · 1→30,799 |
| 28 `(1,1,4096,1024)` BLOCK 512x128, rows 16 | native, brmax 16 | 16→10,452 (base) · 8→12,272 |

Rule-level, base vs `ia_bal`: focus 83,997 → 82,889 (1.013x); case 12 20,372 →
19,493 (1.045x); `(1,1,8192,1000)` (PARTIAL_W=8) 84,766 → 83,512 (1.015x); cases
17/28/31 and every rows_max==1 / chunked-width case are byte-identical programs.
