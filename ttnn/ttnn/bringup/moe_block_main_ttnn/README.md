# ttnn.bringup.moe_block_main (moe_block_main_ttnn)

Marko Staletovic's reworked MoE block ops (`ttnn.experimental.moe_block` on `mstaletovic/moe-block-main`), carried as a
separate bring-up op set next to the MoE block this branch already runs (models/demos/moe_block on Python
ProgramDescriptor ops and the `ttnn.bringup.flat_routed_expert` / `combine_rf` forks). Python block:
models/demos/moe_block_main (`MoEBlock`). DeepSeek-V4.1 selects it with `DS41_MOE_BLOCK_SRC=main`.

## What it does

The MoE block's prefill ops (`ttnn.bringup.moe_block_main`): token owners, 2D dispatch, flat experts + combine in
one program. A different algorithm from `deepseek_prefill/` (sparse 2D multi-hop unicast dispatch, L1 accumulate
combine, fused head); the Python `MoEBlock` (models/demos/moe_block_main) orchestrates them.

- `common/`: `MoEBlockConfig` / `MoEModelDims` / `MoEDtypes`, the shared host math (routes and the fabric's parity tie
  rule, span tables, manifest layout, mesh plan, port LUTs, Ethernet-nearest port placement) and `MoEBlockResources`
  (the persistent words, placement and plans the ops share), `address_patches` (address-only program-cache
  hits for a per-chip mesh workload: a tensor's runtime args are rewritten only when its address moved).
- `fused_head/`, `sparse_dispatch_2d/`, `flat_routed_expert/`, `routed_experts_combine/`, `fused_tail/`: one device op
  each (one program a chip); each has its own README.
- `block/`: `moe_block_arena_spec`, the block's one L1 arena a call (the fused head runs in it, then the routed
  experts + combine; both take `arena=`).

A forward (`models/demos/moe_block_main/tt/moe_block.py` `MoEBlock`): allocate the arena, `fused_head(..., arena=)`,
`sparse_dispatch_2d`, `routed_experts_combine(..., arena=)`, free the arena, `fused_tail`. `models/demos/moe_block_main/tests/test_moe_block_smoke.py` runs it on the mock
Galaxy or an 8x4 Galaxy.

What changed against the block this branch carries (models/demos/moe_block, P.17 drop of
`dnijemcevic/model_bringup_lowbit` @ `5e86f048ad4`):
- the whole block is C++ device operations (one program a chip each, address-only program-cache hits) instead of
  Python ProgramDescriptor builders (`generic_op` / `prepared_program`);
- **no DRISC**: the routed experts' gate/up weights are read by the gate/up cores' own NCRISC on NoC0
  (`flat_routed_expert/device/kernels/gate_up_weight_read.hpp`, two reads in flight, each slot's cores the gate/up
  cells fewest NoC0 hops from the bank's optimal reader core), where the carried flat op had the DRAM cores' RISC
  (DRISC, `flat_routed_expert_ttnn/device/flat_drisc.cpp`, `se_drisc_post.hpp`) push them. Nothing in this folder
  launches a DRAM-core program;
- the persistent words and semaphores live in L1_SMALL (`MoEBlockConfig.persist_l1_small`, one packed slab of
  `words_slab_bytes` a core), the head and the experts + combine share one L1 arena a call;
- a padded chunk (actual_start / actual_end), token loops and chunk splits are planned by the ops.

## Consider it instead of

`models/demos/moe_block` (the carried Python block on `ttnn.bringup.flat_routed_expert` + `combine_rf`) when the DRAM
cores' RISCs must stay free (the serving framework's prefill runner uses them). Not a drop-in: another weight layout,
no OWN_ROWS head / tail mode (DeepSeek-V4.1's adapter composes it from a TP all-to-all and two stock ops).

## Options

The ops' arguments are the source's (each op's README, `common/moe_block_config.hpp`), plus what this branch adds,
each default off (= the source's program):
- `FlatLayoutOptions(weights_nk=False)`: the routed experts' weights laid out from nn.Linear's [N, K] matrices, every
  weight tile transposed and transposed back by the gate/up and down matmuls (`SE_W_NK`), so a bfp tile shares its
  exponents along K: DeepSeek-V4.1's e2m1 x ue8m0 experts keep the checkpoint's 32-wide K groups (bfp8 exact, bfp4 in
  them) - the `ttnn.bringup.flat_routed_expert` fork's weights_nk (DS41-35). `flat_routed_expert_weights.py`:
  `flat_weight_regions(..., nk=)`, FlatRoutedExpert with `layout.weights_nk` takes [N, K] matrices. Off: the source's
  program and key.
- `set_max_grid(max_x, max_y)` (module function; default (0, 0) = the whole grid): every op's planner (ports, fused
  head, flat experts, combine cells, fused tail) takes its grid from `common/core_grid.hpp` `block_grid()`, the compute
  grid cut to the cap - a Blackhole Galaxy's 110-core rule (DeepSeek-V4.1 FIX.15: 11 x 10 of the 12 x 10). Set once,
  before the first plan. The call's L1 arena and the words' slab still span the whole grid (memory, no program).
- `MoEDtypes.shared_activations = bfloat8_b` (default bfloat16 = the source's program): the fused head's shared expert
  keeps x and h in bf8 tiles (x own, x in, h own, h all, the x gather's packets; `SE_XTB` / `SE_HTB` / `XTB` 1088 B),
  read from the caller's `fused_head(..., xq=)` (x in bf8; the router keeps reading bf16 x, from DRAM: the row's
  router x is no longer taken from x own's L1); the dispatch prep untilizes x own as stored. DeepSeek-V4.1's reference
  fp8-quantizes both (FIX.9, DS41-132). `MoEBlock` makes xq with one capped typecast a call unless the caller gives it.
  Not with a latent (model_checks refuses). With the DRAM fallback x_full is gathered from xq.
- `MoEDtypes.shared_pack_precise = True` (default False): the head's compute packs bfp8 h with `bfp8_pack_precise` -
  one rounding from fp32 DEST at the group's shared exponent (as ttnn.typecast packs), not two (DS41-137); only bfp8
  packs change.

## Measured

bf8 shared-expert activations (P.37, Galaxy 8x4, DeepSeek-V4.1-Flash dims, random weights, LoFi, 1024 tokens):
the head's output vs torch with x / h on the bf8 grid rel 0.0471 (bf16 activations 0.0585), vs full precision 0.0465
(0.0567); `shared_pack_precise`: every one of 1,777,311 decided h values packed by the one-rounding rule (232,605 of
them where one and two roundings differ), off: every one by the two-rounding rule (`tests/test_fused_head_bf8.py`).

DeepSeek-V4.1-Flash, Galaxy 8x4 (FABRIC_2D_TORUS_XY), chunk 5120 at start 51200, P.34 defaults, cap 11 x 10,
weights_nk, bfp4_mse experts (2026-11-06, /scratch/dnijemcevic/rebase/moe_main/check vs fasttests/check, the carried
block): chunk wall 196.1 ms vs 180.6; device 177.3 vs 165.4 ms (experts section 106.3 vs 94.9: routed_experts_combine
63.4, sparse_dispatch_2d 25.0, fused_head 21.9 ms a chunk, slowest chip); host: the MoE layer's call 49.7 vs 68.0 ms
a chunk (cProfile), 8 launches a layer (fused_head ~350 us, routed_experts_combine ~490 us, sparse_dispatch_2d ~110 us
each, profiled run); every quick-ladder metric inside the P.34 gate's limits (mean err_layer 0.164 vs 0.166). Serving
runner smoke (tt-d-gen engine): pass.

## Limits

- Blackhole, a 2D fabric (FABRIC_2D or FABRIC_2D_TORUS_XY); the mesh opened with L1_SMALL for the words' slab and a
  fabric payload of one combine row (`models/demos/moe_block_main/tt/moe_impl.py`).
- The ops take the whole compute grid unless `set_max_grid` caps it; the flat expert's fixed (unshaped) gate/up
  rectangles need 12 columns on a Galaxy chip (a shaped plan, e.g. DeepSeek-V4.1's 48 cores, fits 11).
- Kernels are main's of 2026-11-06 (`f93403fa228`); this branch's kernel APIs may differ (JIT errors show at first
  use).

## How to call

`models/demos/moe_block_main/tt/moe_block.py` (`MoEBlock(mesh, resources, dtypes=, activation=)(x, weights)`);
smoke test: `models/demos/moe_block_main/tests/test_moe_block_smoke.py`; op tests: `tests/` here.

## Used by

deepseek_v41_flash_d_p (`DS41_MOE_BLOCK_SRC=main`, tt/moe_block_main_adapter.py)

## History

CHANGELOG.md
