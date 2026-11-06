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
no OWN_ROWS head / tail mode, no `max_grid` (see Limits).

## Options

The ops' arguments are the source's (each op's README, `common/moe_block_config.hpp`), plus what this branch adds,
each default off (= the source's program):
- (none yet; the CHANGELOG lists every change)

## Measured

not measured here yet (DeepSeek-V4.1 galaxy_8x4: deepseek_v41_flash_d_p, DS41_MOE_BLOCK_SRC=main).

## Limits

- Blackhole, a 2D fabric (FABRIC_2D or FABRIC_2D_TORUS_XY); the mesh opened with L1_SMALL for the words' slab and a
  fabric payload of one combine row (`models/demos/moe_block_main/tt/moe_impl.py`).
- The ops take the whole compute grid (no core cap).
- Kernels are main's of 2026-11-06 (`f93403fa228`); this branch's kernel APIs may differ (JIT errors show at first
  use).

## How to call

`models/demos/moe_block_main/tt/moe_block.py` (`MoEBlock(mesh, resources, dtypes=, activation=)(x, weights)`);
smoke test: `models/demos/moe_block_main/tests/test_moe_block_smoke.py`; op tests: `tests/` here.

## Used by

deepseek_v41_flash_d_p (`DS41_MOE_BLOCK_SRC=main`, tt/moe_block_main_adapter.py)

## History

CHANGELOG.md
