# moe_tail

`ttnn.experimental.moe_block.moe_tail(y_own, shared_expert, *, resources, w_up=None, math_fidelity=HiFi2, output=None)`

The last op of the MoE block: it brings the routed experts' combined rows back into the residual's layout
`[1, 1, S/SP, H/TP]` and adds the shared expert's output. The TP group is the mesh row.

## What it does

In practice it is one of two fused ops, picked by the model (`resources.dims().latent_dim`):

| case | models | equivalent ops |
|---|---|---|
| all-to-all (most models) | GLM-5.3, K2.7, GLM-Flash, DSv4.1-Flash | `tilize(y_own) + shared_expert`, then a TP all-to-all (each chip keeps its `H/TP` column block of every chip's rows) |
| latent engine | Kimi-K3 | a TP all-gather of the latent rows, `rmsnorm(latent) @ W_up[:, this chip's H/TP columns] + shared_expert` |

Inputs: `y_own` is the combine's output - this chip's owned rows, row-major (`[S/N, H]`, or `[S/N, latent_dim]` for
the latent engine); `shared_expert` is moe_head's shared expert output (TILE). The latent engine also takes `w_up`,
made once a layer by `prepare_moe_tail_weights(latent_up, latent_norm, *, resources, dtype=bfloat8_b)` (the norm
weight folded in, laid out for the engine).

## How

One device operation (`ttnn::prim::moe_block_tail`), two program factories; one program for the whole mesh.

**All-to-all** (`TailAllToAllFactory`, `device/kernels/all_to_all/`):
- worker cores (each `worker_columns` <= 8 tile columns of `H`): `worker_input_reader` reads the rows and the shared
  expert, `worker_tilize_add_compute` tilizes (`compute_kernel_lib::tilize`) and adds, `worker_output_writer` writes
  this chip's own block straight into the output and the other chips' blocks to a DRAM staging buffer;
- port cores (one per direction and fabric link, next to the link's Ethernet core): `port_staging_reader` reads the
  staging buffer once the workers wrote a row tile, `port_fabric_sender` sends it over the fabric (2D multi-hop) into
  the destination chip's output, with an arrival count the destination waits for.

**Latent engine** (`TailEngineFactory`, `device/kernels/latent_engine/`):
- engine cores (every grid column but the last, in pairs - one core per K half of the latent): the latent rows are
  gathered over TP straight into their L1 (`gather_port_*` in the last grid column), multicast along each row
  (`engine_row_exchange_writer`), and `engine_norm_projection_compute` computes the sum of squares and the projection
  in fp32 (`engine_dram_reader` reads W', the own rows and the shared expert).

The plans (geometry, schedules, port cores: `device/moe_tail_plan.hpp`) are built once per `MoEBlockResources`. Each
data-movement kernel's per-core args live in a per-chip DRAM table (`common/args_table.hpp`), so all chips run one
program; a cache hit only rewrites tensor addresses that moved (`common/address_patches.hpp`). Counters only count up
or are re-armed by their waiters, so traced replays need no reset.

A padded forward (moe_head's `actual_start` / `actual_end` or `metadata`): the all-to-all skips the row tiles with no
real row and writes zeros to the output's padding row tiles; the latent engine computes every row.

## Tests

- `tests/ttnn/unit_tests/operations/experimental/moe_block/test_moe_tail_lb.py` (a LoudBox, SP 2 x TP 4; GLM-5.3 and
  Kimi-K3): accuracy vs torch, determinism and traced device time.
