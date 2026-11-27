# Flat expert features to A/B on Galaxy

Branch `mstaletovic/flat-galaxy-features`: the tip of the 3-PR stack (#67449 → #67450 → #67451, at `104c89576d3`) plus
five flat expert features from single-chip perf work. **Every feature is off by default**: with no environment
variables set, this branch is the stack tip. The goal is to measure each feature on the full op (MoE block with the
combine) on Galaxy, alone and together, and keep what helps.

## Switching features

```bash
FLAT_FEATURES=cmm                      # one feature
FLAT_FEATURES=cmm,rowmop,lean          # several (comma list)
FLAT_FEATURES=all                      # all five
FLAT_FEATURES=all FLAT_NOC1=0          # all but one: FLAT_<KEY>=0 / =1 forces a feature over FLAT_FEATURES
```

- Read when the program is built: use **one setting per process** (the program cache keeps what it built first).
- Every program build logs what is on: `flat_routed_expert: features [cmm,rowmop]` (or `[none]`). Check it in the log
  of every run; the variables must reach the process that builds the programs.
- With `noc1` the plan also logs `flat_routed_expert noc1: primaries (x, y) (x, y), helpers (x, y) (x, y)`.
- Host code: `device/flat_features.hpp` (the switch), `device/flat_routed_expert_program_factory.cpp` and
  `device/flat_routed_expert_plan.cpp` (where each is read).

## The features

| key | what | single-chip result (p150b, GLM-5.3 4x4, 16 experts a chip) | applies when |
|---|---|---|---|
| `cmm` | gate/up matmul on `custom_mm` with full [32, 32] in0 tiles | M 320 -2.5%, M 480 -7.2%; GLM 8x4 hot -10%, DSV4.1-Flash hot -10.6%; worst case +1.6% | LoFi only (the log says if it was asked for and not taken) |
| `rowmop` | down matmul a weight block per call on the Blackhole row MOP (`matmul_block_k<true>`) | pseudo combine M 480 -4.3%, 64 experts a chip -3.1% | row-major y only (the combine path) |
| `lean` | combine hand-off without a division or a NoC address build per row | pseudo combine M 320 -2.4% | running with the combine (y handed off in L1) |
| `trailers` | each row's 32 B trailer written by writer `row % 32`, not all by the column-0 writer | pseudo combine M 320 -4%, M 480 -1..-3% | running with the combine |
| `noc1` | x snakes on NOC1, relays placed one hop from the snake heads | -0.7% (M 80) .. -3.4% (M 240), M 480 -1.9%, zipf M 320 -0.8% | any; see the notes |

"Pseudo combine" = a single-chip stand-in for the combine's ports (the y hand-off into port injection rings), not the
real combine: the hand-off features (`lean`, `trailers`, and `rowmop` with row-major y) have not run on a Galaxy yet.

### cmm - gate/up on custom_mm

- Files: `kernels/gate_up_compute.cpp` (option `gate_up_custom_mm`), the custom_mm LLK / API
  (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_{math,unpack_AB}_custom_mm.h`,
  `llk_custom_mm_common.h`, `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/*custom_mm_api.h`,
  `tt_metal/hw/inc/api/compute/experimental/custom_mm.h`): full [32, 32] in0 tiles with `rt_dim`, and an in0 k step.
- Bit-identical to the `matmul_block_k` path it replaces; the gain is the matmul's per-tile rate (18.1 -> 16.7 cycles
  a tile in a one-core bench). Biggest at large M (math-bound).

### rowmop - down matmul on the row MOP

- Files: `kernels/down_compute.cpp` (option `down_row_mop`), `tt_metal/hw/inc/api/compute/matmul.h`
  (`matmul_block_k` gained `in0_k_stride`, default 1: h is k-tile-major, so the in0 step is the row tiles).
- Replaces 64 `matmul_block` calls a row (one a K tile) with one call a weight block.
- Not the down `custom_mm` from the perf branch: that one hung with row-major y (pack_untilize), so it is not here.

### lean - the hand-off's per-row diet

- File: `kernels/y_row_major.hpp` (option `lean_handoff`): the ring slot as a running offset (no `% slots` a row),
  each destination's NoC base computed once, the trailer owner as a counter.
- Why: a row is one small NoC write, so the RISC's per-row instructions are the hand-off's cost (NoC traces on one
  chip: no saturated link; the issuing RISC was the limit).

### trailers - spread over the writers

- Files: `kernels/y_row_major.hpp` (hand-off arg 4, `kArgTrailerWriters`), factory `set_flat_combine_report` sets it
  to the number of y writers (32). The combine's own l1 arg 4 is an unused 0 today; nothing changes on the combine
  side. With W = 0 (off) the column-0 writer writes every trailer, as today.
- Why: a row is taken by the port only once every writer has landed it, and the column-0 writer had twice the writes.

### noc1 - x snakes on NOC1

- The gate/up NCRISC forwards x on NOC1 along a mirrored snake (head on the rectangle's east column, going north,
  then one column west); the gate/up BRISC moves to NOC0 and reads the gate/up weights (taking over from the NCRISC)
  and sends h; each primary injects from its BRISC on NOC1.
- Placement (plan): each primary one NOC1 hop from its snake head (east of the rectangle on its row, or south of its
  south-east corner at the grid's edge), each helper one NOC0 hop from its primary (north, else west); the down
  cores take the remaining cores. On GLM-5.3 4x4 (p150b): primaries (6, 4) (10, 8), helpers (6, 3) (9, 8).
- All three parts were needed on one chip: placement alone with NOC0 snakes was much worse (M 240 1292 vs 962),
  NOC1 snakes with the old relay places +6.7% at M 480.
- Risks on Galaxy: (1) the moved relays displace down cores - if the block's placement relies on specific cores
  being free (combine ports, other ops' cores), check the logged positions; (2) NOC1 now carries x beside the helpers'
  x DRAM reads and the h chains, and on Galaxy also whatever the combine puts on NOC1; (3) the gate/up weights come
  from the BRISC's path, new on Galaxy. Test it last and on its own first.

## Suggested runs

1. Baseline: no variables (log: `features [none]`). Check it matches the stack tip's numbers.
2. Each feature alone: `FLAT_FEATURES=cmm`, `=rowmop`, `=lean`, `=trailers`, `=noc1`.
3. Combined: `FLAT_FEATURES=cmm,rowmop,lean,trailers` (the four low-risk ones), then `all`.
4. For every setting that is kept: the accuracy and determinism tests of the block / combine with the same variables
   (the features are bit-identical or reorder nothing; a PCC change means a bug).
5. Record per case: the block's / combine's device time, and which features were on (from the log line).

## Single-chip checks done on this branch (p150b, flat expert alone)

- Build; with no variables the log says `features [none]` and the timings match the stack tip.
- GLM-5.3 4x4 accuracy + determinism (`test_flat_routed_expert_checks.py -k "glm53 and 4x4 and not perf"`, 8 tests):
  pass with `FLAT_FEATURES=cmm`, `=noc1` and `=all`. `noc1` logs primaries (6, 4) (10, 8), helpers (6, 3) (9, 8) - the
  rule reproduces the hand-tuned placement.
- `rowmop`: row-major y (the flat expert called with `y_row_major=True`, M 320) bit-identical with it off / on.
- `lean` / `trailers` act only on the combine hand-off, which needs a mesh: compiled (the down reader with the
  hand-off enabled, `lean` off and on, with the device compiler) but **not run** - the first Galaxy run is their
  first run. A hang there points at these two first.
- Perf, flat expert alone (no combine), us:

| GLM-5.3 4x4 | none | cmm | noc1 | cmm + noc1 |
|---|---|---|---|---|
| M 160 (5120) | 885.4 | 882.0 | 882.2 | 878.0 (-0.8%) |
| M 240 (7680) | 968.8 | 966.2 | 943.3 | 930.5 (-4.0%) |
| M 320 (10240) | 1061.5 | 1054.5 | 1055.1 | 1026.9 (-3.3%) |
| M 320 zipf | 1262.2 | 1210.0 | 1251.3 | 1199.4 (-5.0%) |

## Measured on one chip and not worth Galaxy time

Per-bank fairness of the weight reads (progress counter, token), h chains on NOC0, every x multicast variant
(primary NOC0 / NOC1, helpers multicasting), dynamic NoC mode, the K split, the hand-off on the down BRISC,
staggered writers, pull grants. Details: `FLAT_GLM16_PERF_WORKLOG.md` on `mstaletovic/custom-mm-full-in0-wip`.

## Bigger levers, not built

- Preloading the first experts' weights during dispatch: emulated upper bound -3.4% .. -5.9% (M 80 -11% with x too).
- Dispatch writing x as bfp8 rows (halves the x DRAM reads).
