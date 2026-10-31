# fused_head

The MoE block's head as one program a chip (reference: `models/demos/moe_block/tt/fused_head.py` `HeadSE3`,
`moe_block.py` `FusedHead` on mstaletovic/moe-trim): the x all-gather over TP straight into the shared expert's
matmul cores, the shared expert's down + reduce-scatter, the router (logits, top-K, the joint routing rows), the
routing exchange + the combine planner (into the exchange's plan tensors), the dispatch prep and, with a latent, the
latent down + all-to-all.

## Status

| Piece | Where | State |
|---|---|---|
| Layout planner (`head_plan.fit`) and its sizing closure (`SeEngine3` arena sizing, `se_rs.RsPlan` + `plan_transport`, `RouterStream.layout_of`, `routing_a2a` sizes, the routing exchange's paths, `router_placement`) | `device/head_plan.*`, `device/head_sizing.*` | done; exact parity with the reference on 230 cases |
| Block-level plan (`FusedHead.plan` / `model_checks` / `shape`) cached in `MoEBlockResources` | `fused_head.*` (`plan_fused_head`) | done |
| Kernels (se_engine3, x_gather3, router_stream, routing_stream Y sender, routing_a2a planner via `_plan_fn`, prep_stream route words) | `device/kernels/` | materialized as files, not yet compiled on device |
| Device operation, program factory, `override_runtime_arguments` | `device/` | not written |
| Reference dump (`dump_ref_fused_head.py`, lead runs it in the reference tree) | tests | written, not run |
| Bit-exact device test, op host-timing test | tests | need the device op |

## Host plan

`head_plan_fit(HeadShape, HeadFitOptions)` returns a `HeadFit`: `ok`, the planned layouts (best first, each with its
arena bytes, region sizes, predicted us and the cost's parts), `why` (the requirement that failed, with its numbers
and the largest rows that fit), `free` (the cost model's pick with unlimited L1) and `note` (what L1 forced, or the
DRAM fallback's cost). It reproduces the reference bit for bit, messages included (`fmt_fixed` formats as Python's
`f"{x:.0f}"`; floating-point operations keep the reference's order). `plan_fused_head(resources, dtypes, A)` builds
the shape from the block's dims / config / topology / compute grid / fabric payload and caches the answer in the
resources, so a program-cache hit never re-plans. Python: `ttnn.experimental.moe_block.head_plan_fit`,
`HeadShape`, `plan_fused_head`.

The free L1 figure (`l1_free`) is the reference's host default (`L1_FREE - L1_MARGIN`); the reference measured it on
the device at the head's build. With the arena as a program CB (cpp_op_conversion_plan.md section 2) the factory
should pass the L1 the CB may take.

## Kernels

`device/kernels/head_kernel_contract.hpp` lists every kernel (RISC, NoC, compile-time args, defines and the
reference expression behind each, runtime / common-arg layouts, the kernel_lib multicast families). The files are
the reference's stitched sources with the string surgery turned into headers and defines:

- shared pieces (x_gather3 `_CT` / `_WALK` / `_LA_*` / `_RS_*`, router_stream `_ROWS` / `LSTREAM`, routing_stream
  `RA_SEND`, prep_stream's route-word function, routing_a2a's `_PLAN` as rewritten by `routing_stream._plan_fn`) are
  headers (`ra_plan.hpp`: `ra::plan(a0, B)`);
- placeholders: the planner's `__X__` are `RAP_X` defines, prep_stream's `__X__` are `RW_X` defines; the `GROW()`
  macro text is `se_defs.hpp`;
- addresses the reference baked into defines (`SE_EW`, `RT_ENG`, `RT_MSYNC`, `RT_PWORD`, `RT_PART`) are common
  runtime args, so no define carries an address.

`tests/.../moe_block/fused_head/gen_head_kernels.py --check` (reference environment) rebuilds each stitched text from
the files and compares it with the reference's.

## Tests

- `test_head_plan_parity.py` (host only, any machine): `head_plan_fit` against `head_plan_golden.json.gz`, the
  reference's answers for GLM-5.3, K3, K2.7, GLM-5.3-Flash, DSv4.1-Flash at rows 128 .. 2048 a mesh row on 8x4
  (line and torus), 2x4 and a ring of 4, plus fit()'s options, other accumulator counts, TP 2 / 8, one link and
  smaller free L1. Regenerate with `dump_head_plan_golden.py` in the reference environment.
- `dump_ref_fused_head.py` (reference environment, a Galaxy): saves the head's per-chip inputs and outputs (se, rows,
  xbuf, route, latent, the routing plan tensors) for the device test, under `$MOE_BLOCK_REFS/fused_head/`.
