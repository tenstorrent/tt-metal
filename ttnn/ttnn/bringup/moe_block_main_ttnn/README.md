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
- `MoEBlockConfig.combine_snap_split = True` (default False; needs `combine_snap` q): the combine's sum order-free past
  2^24 q. Each weighted term is split into hi (a multiple of 2^18 q) and lo (the rest, a multiple of q), summed in two
  fp32 sums (exact while |hi sums| < 2^24 x 2^18 q, top_k <= 64) and added once at the end. The single sum is exact only
  below 2^24 q (32 at q = 2^-19); DeepSeek-V4.1's routed sums reach 21376, so without the split its outputs depended
  on the packets' arrival order (FIX.23). Costs a second fp32 sum a token in the accumulators' L1 and the split's SFPU
  work (README Measured). Bit-identical to the single sum wherever that one was exact and its terms below 2^22 q.

- `RouterSpec.candidates = n` (default 0 = the source): replica slots - `num_experts` counts slots (chips x slots a
  chip), the router ranks n candidates, and `prepare_fused_head_weights(slot_map=)` (required then: 1024 integer-valued
  fp32 words, the layer's runtime map, one extra router-bias tile) maps each (token, candidate) to one of the
  candidate's slots by a hash of the token (fused_head/device/kernels/router_rows.hpp `slot_of`); every later op works
  on slots as before. Needs fold 1 and candidates + slots <= 1024.
- `expert_slot_copy(src_gu, src_d, dst_gu, dst_d, runs, resources, jobs, links=None, depth=8, batch=4)`: run-table
  copies of weight bytes between chips (2D fabric unicast, any chip to any chip; jobs [(src, dst, table)], a destination
  once a call; the destination's program brackets the write: ready at its start, done before its end) or within a chip
  (src == dst, split over the port cores). One cached program for every job set (jobs are runtime args). Limits: at
  most 32 jobs a port core, 4 links, 255 chips, 256 tables; tensors DRAM sharded over the banks (offsets are bank-local
  byte offsets of the shard); the run tables are the caller's (DeepSeek-V4.1: models/demos/deepseek_v41_flash_d_p/tt/
  expert_slots.py from the flat plan's facts).

## Measured

combine_snap_split (FIX.23 on the replica branch, Galaxy 8x4, DeepSeek-V4.1, 40 layers, chunk 5120, q = 2^-19, 17
accumulators, D 6 - the plan unchanged, the accumulators' L1 + 10 tokens x 5 tiles x 4 KB): routed_experts_combine
55.84 -> 56.79 ms a chunk (+0.95, ~24 us a layer; the first version with six stock SFPU passes +3.18), device 171.05
-> 172.85 ms, clean chunk wall 183.2 -> 183.5 ms (spreads 15.6 / 6.5: no wall change). Identical prefills: code prompt
max |diff| up to 0.219 -> 0 (tests/test_determinism.py, 4 chunks x 3 runs, novel and code). Ladder (rung last): every
metric identical. Fork test `test_experts_combine_split_determinism` (x 256, |y| to 362): 10 calls bit-identical, PCC
0.98428 = the single sum's on the same inputs. On this base (P.50): the orchestrator's A/B.

expert_slot_copy (P.47, Galaxy 8x4, FABRIC_2D_TORUS_XY, DeepSeek-V4.1 layer 20's bfp4 weights: one slot 18.98 MiB in
272 runs, chunk 10880 B, 2 links, 8 port cores a chip; op + synchronize, median of 20 after 2 warm-up,
models/demos/deepseek_v41_flash_d_p/tests/test_expert_slot_copy.py): one copy 0.717 ms (27.8 GB/s) at 1 hop, 0.75-0.79
ms (25-27 GB/s) at 2 and 4 hops (row, column, diagonal; odd and even source slots); 32 copies at once (random sources,
up to 3 a source) 2.79 ms (228 GB/s total); a layer's 32-copy oracle table (one source sending 6) 4.47 ms; the
load-time 12 -> 13 slot re-layout (local, 272 merged runs) hides behind the next layer's cache load (enqueue < 10 ms).
Bit-exact every time; every other tile unchanged.

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

Replica slots (P.46, 8x4, DeepSeek-V4.1 layer 20, 5120 tokens): 13 slots a chip with an empty map and with 10 copies
both bit-identical to the 12-slot block (tests in models/demos/deepseek_v41_flash_d_p/tests/test_moe_replicas.py).

## Used by

deepseek_v41_flash_d_p (`DS41_MOE_BLOCK_SRC=main`, tt/moe_block_main_adapter.py)

## History

CHANGELOG.md
