# fused_tail

`ttnn.experimental.moe_block.fused_tail(y_own, shared_expert, *, resources, w_up=None, math_fidelity=HiFi2, output=None)`

The MoE block's tail as one program a chip: the combine's owner rows (+ the shared expert's output) to the residual's
layout `[1, 1, S/SP, H/TP]`. The TP group is the mesh row (a ring when the fabric wraps it). One device operation
(`ttnn::prim::moe_block_fused_tail`) with two program factories, chosen by `resources.dims().latent_dim`:

| factory | models | what it does |
|---|---|---|
| `TailEngineFactory` | latent (Kimi-K3) | the latent all-gather over TP straight into the engine's L1, the chip's own rows tilized there, `out = rmsnorm(z) @ W_up[:, chip's columns] + se` (the norm folded: its weight in `W_up`, `1 / rms` on the fp32 sums) |
| `TailA2AFactory` | no latent (GLM-5.3, K2.7, GLM-Flash, DSv4.1-Flash) | the own rows tilized, `+ se` (owner rows), the TP all-to-all into `[S/SP, H/TP]` |

`prepare_fused_tail_weights(latent_up, latent_norm, *, resources, dtype=bfloat8_b)` uploads a latent model's up
projection once a layer in the engine's layout (per chip its output columns, the norm weight folded in, each engine
column's K ranges as one DRAM burst each).

## Programs

**TailEngine** (`GX x 10` engine cores, `GX` = grid columns - 1, plus four gather ports in the last column):
- engine core `(cx, cy)`: pair `cy % 5`, K half `cy / 5`, output columns `column_split(Ho / 32, GX)`. The call's row
  tiles are steps round-robin over the pairs. W' is resident (each half column's K ranges read once and multicast);
  z arrives share by share into the landing slots (own rows tilized there, the others by the ports) and is multicast
  down each row; the sum of squares in fp32 (HiFi4), the partner halves swap partials; out = (mine + partner's) x
  rsqrt(sum / Hr + eps) + se.
- ports (direction x link): the balanced ring / line all-gather, link l carrying K half l, straight into the next
  chip's landing slots; a fused increment a 3-column group's last share (its watcher counts the step landed for the
  group).
- persistent (resources): `fused_tail/engine_words` (per-step counters that only count up, a call count),
  `fused_tail/engine_ready` (the fence), `fused_tail/engine_port_epoch` (the ports' call counts).
- no per-call L1 arena: the regions are program CBs. The landing CB is created first, on the
  engine and port cores, so it sits at one address on every core of every chip; kernels read the slot bases with
  `get_write_ptr`. The ring depth (`NSLOT`) is chosen from the free L1 when the plan is built (hashed).

**TailA2A** (`H / 32 / W` worker cores, `W` <= 8 tiles each, plus a port per direction and link next to its
Ethernet core):
- worker: its tile columns of every own row tile: read row-major, tilize, + se, its own block's tiles straight into
  the output, the others into the staging buffer `z`, then +1 on every carrying port's z-ready word of that row tile.
- port: once a row tile's workers wrote it, its runs of every destination's block, multi-hop 2D unicast into the
  destination's output; a fused arrival increment every 8th run; ready / arrival fence, re-armed by the waiters.
- persistent (resources): `fused_tail/a2a_zready`, `fused_tail/a2a_arrival`, `fused_tail/a2a_ready` (port cores),
  `fused_tail/a2a_z` (DRAM staging).
- ring ties (a destination half the ring away) follow the fabric (`get_eth_forwarding_direction`).

## Caching and host cost

The plans (`fused_tail/engine_plan`, `fused_tail/a2a_plan`: geometry, schedules, port cores) are built once per
`MoEBlockResources`. The program hash is `(kind, resources.plan_key(), fidelity, NSLOT, output memory config, input
specs)`; no address. Every address a kernel takes is a tensor's base (inputs, output, the resources' persistent
tensors); the mesh workload factory records where each sits (`common/address_patches.hpp`), and a cache hit rewrites a
tensor's arguments only when its address moved (nothing when none did).
Counters only count up or are re-armed by their waiters: traced replays need no reset.

## Tests

`tests/ttnn/unit_tests/operations/experimental/moe_block/fused_tail/`:
- `test_fused_tail.py`: bit-exact against saved reference cases (`$MOE_BLOCK_REFS`).
