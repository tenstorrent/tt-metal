# Port Plan — gated_delta_attn

Port plan for `transformer/gated_delta_attn` (`GatedDeltaAttnSeqDeviceOperation`), ported from the `ProgramDescriptor` API (direct `create_descriptor`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

> **Revision 2026-10-15 (re-port).** The first attempt (2026-10-08) capitulated on the state buffer `S` (`c_8`): the reader seeded it and compute recycled into it. PR #60011 (`7c7fafc9758`, merged 2026-10-13) split the seed into a reader-only `c_7` / `S0`. The branch was rebased onto `origin/main` (2026-10-15) to pick it up. The inventory and plan below describe the **post-#60011** legacy op; the sections that changed are `CBs` (index 7, 8), `Planned Spec Shape`, `Applied Patterns` and `Deferred / Flagged`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, in the **direct-descriptor** shape: `static ProgramDescriptor create_descriptor(...)` is a member of `GatedDeltaAttnSeqDeviceOperation` itself (`device/gated_delta_attn_device_operation.hpp:31-32`), body `device/gated_delta_attn_program_factory.cpp:28-216`. There is no `program_factory_t`, so this triggers [exception 3](ttnn_factory.md#3-give-a-direct-descriptor-op-a-conventional-program-factory).
- Variants: single (one program, one core per head).
- Custom `compute_program_hash`: none. The default reflection-based hash applies (#57409 removed the custom one). No `attribute_values` / `to_hash` backdoor.
- No `override_runtime_arguments`, no `get_dynamic_runtime_args`, no op-owned tensors, no pybound `create_descriptor` (`gated_delta_attn_nanobind.cpp` binds only the user function).

### Kernels
All three share `core_ranges = cores`, the set of single-core ranges `{(h / grid_y, h % grid_y)}` for `h < BH` (column-major).

| unique_id | source | CTAs (positional) | CTAs (named) | RTAs (per core) | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_gated_delta_attn.cpp` | `Ct, Kt, Vt`, then 9 `TensorAccessorArgs` blocks (L_unit, v_beta_sc, k_bd_sc, intra_attn, q_decay, k_decay_t, dl_exp, L_inv, initial_state, the last being `nullptr` when absent) | none | `h, NC, L_unit*, v_beta_sc*, k_bd_sc*, intra_attn*, q_decay*, k_decay_t*, dl_exp*, L_inv*, initial_state* (null → 0)` | none | none | O2 (unset, DM) | `DataMovementConfigDescriptor{RISCV_1, NOC::RISCV_1_default}`, noc_mode default `DM_DEDICATED_NOC` — **custom** (the reader default NOC is NOC_0) |
| writer | `device/kernels/dataflow/writer_gated_delta_attn.cpp` | `Ct, Kt, Vt`, then 2 `TensorAccessorArgs` blocks (outputs[0], outputs[1]) | none | `h, NC, out*, final_state*` | none | none | O2 (unset, DM) | `DataMovementConfigDescriptor{RISCV_0, NOC::RISCV_0_default}`, `DM_DEDICATED_NOC` — **custom** (the writer default NOC is NOC_1) |
| compute | `device/kernels/compute/gated_delta_attn.cpp` | `Ct, Kt, Vt` | none | `NC` | none | none | O3 (unset, compute) | `ComputeConfigDescriptor{HiFi2, fp32_dest_acc_en=true, math_approx_mode=false}`; `dst_full_sync_en`, `bfp8_pack_precise`, `unpack_to_dest_mode` all default |

(`*` = `Buffer*` RTA, emitted through `emplace_runtime_args`.)

### CBs
(Post-#60011.) All CBs are Float32, `page_size = 4096`, `core_ranges = cores`, no `tile`, no buffer backing, no `address_offset`, one format descriptor each (no aliasing). `n` = tiles.

| index | name (kernel) | total_size | role |
|---|---|---|---|
| 0 | L_unit | attn_tiles × 4096 | reader → compute |
| 1 | v_beta_sc | out_tiles × 4096 | reader → compute |
| 2 | k_bd_sc | in_kv_tiles × 4096 | reader → compute |
| 3 | intra_att | attn_tiles × 4096 | reader → compute |
| 4 | q_decay | in_kv_tiles × 4096 | reader → compute |
| 5 | k_dt | kdt_tiles × 4096 | reader → compute |
| 6 | dl_exp | 1 × 4096 | reader → compute |
| 7 | S0 | state_tiles × 4096 | reader seeds once (initial state or zeros) → compute consumes on chunk 0 |
| 8 | S | state_tiles × 4096 | compute only: re-pushes the updated state on every non-last chunk, consumes it on chunks ≥ 1 |
| 9, 10, 11 | nm_P_a, nm_P_b, nm_R_a | max(Vt,Kt) × 4096 | compute only |
| 12, 13 | — | not allocated | — |
| 14–17 | L_inv_0..3 | 1 × 4096 | reader → compute |
| 18, 19 | v_cor, k_cum | out_tiles / in_kv_tiles × 4096 | compute only |
| 20–23 | v_prime, v_new, o_inter, intra_v | out_tiles × 4096 | compute only |
| 24 | out | out_tiles × 2 × 4096 | compute → writer |
| 25, 26 | s_upd, S_tmp | state_tiles × 4096 | compute only |
| 27 | final_state | state_tiles × 4096 | compute → writer |

No GlobalCircularBuffer.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `program_factory.cpp:144,198` | `in.L_unit` | reader 2 |
| `:145,199` | `in.v_beta_sc` | reader 3 |
| `:146,200` | `in.k_bd_sc` | reader 4 |
| `:147,201` | `in.intra_attn` | reader 5 |
| `:148,202` | `in.q_decay` | reader 6 |
| `:149,203` | `in.k_decay_t` | reader 7 |
| `:150,204` | `in.dl_exp` | reader 8 |
| `:151,205` | `in.L_inv` | reader 9 |
| `:152,190,206` | `in.initial_state` (optional) | reader 10 |
| `:155,207` | `outputs[0]` | writer 2 |
| `:156,207` | `outputs[1]` | writer 3 |

Kernel side: each is `TensorAccessor(args, addr, f32_tile)` (3rd arg = 4096, Class 2 per the brief).

### Work split
n/a. One core per head, `BH ≤ grid.x × grid.y` (`TT_FATAL` at `program_factory.cpp:53`), no `split_work_to_cores`.

### Shared kernels
none. `grep -rl` for each kernel filename under `ttnn/cpp/ttnn/operations/` hits only this factory.

### Flags
- The legacy DM configs are **non-default NOC assignments** (reader on NOC_1, writer on NOC_0). The brief doesn't mention this. They are carried verbatim as custom Gen1 configs.
- Pre-port: the legacy op **fails under full Watcher** on this Wormhole (`Program size (73424) too large for kernel config buffer (70656)`, rebased tree). It runs under a reduced Watcher (PAUSE, RING_BUFFER, WAYPOINT, STACK_USAGE disabled; NoC sanitize, asserts and CB sanitize kept), which is what verification used. Not a port issue; see the report.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none (default hash)
- **Implementation notes**: direct-descriptor op, so exception 3 applies. `create_descriptor` moves to a nested `GatedDeltaAttnSeqProgramFactory::create_program_artifacts`, plus `using program_factory_t = std::variant<GatedDeltaAttnSeqProgramFactory>`. The header comment about the cache-hit refresh moves with it.

## Planned Spec Shape

- **KernelSpecs**: `reader`, `writer`, `compute`, same sources, in place, 1:1 with legacy.
- **DataflowBufferSpecs**: 25, one per allocated legacy CB: `entry_size = 4096`, `num_entries = n_tiles × n_bufs`, `data_format_metadata = Float32`, no tile metadata (legacy didn't set one). Spec name = kernel accessor name = legacy kernel variable name without `cb_`. No `advanced_options` on any DFB.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 10 always (`L_unit`, `v_beta_sc`, `k_bd_sc`, `intra_attn`, `q_decay`, `k_decay_t`, `dl_exp`, `L_inv`, `output`, `final_state`), plus `initial_state` only when present.
- **WorkUnitSpecs**: one, `{reader, writer, compute}` on `cores`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTAs 2–9 | `Buffer*` of the 8 inputs | `TensorBinding` `tensor::{L_unit, v_beta_sc, k_bd_sc, intra_attn, q_decay, k_decay_t, dl_exp, L_inv}` |
| reader RTA 10 | `initial_state` `Buffer*` or null | conditional `TensorBinding` `tensor::initial_state` + `HAS_INITIAL_STATE` define |
| writer RTAs 2–3 | `Buffer*` of outputs[0], outputs[1] | `TensorBinding` `tensor::output`, `tensor::final_state` |
| reader CTAs 3.. | 9 `TensorAccessorArgs(...).append_to` blocks (`:144-152`) + kernel `TensorAccessorArgs<3>()` chain (reader `:70-87`) | binding mechanism |
| writer CTAs 3.. | 2 `TensorAccessorArgs` blocks (`:155-156`) + kernel chain (writer `:264-267`) | binding mechanism |
| reader/writer `TensorAccessor(..., f32_tile)` | 3rd page-size arg (11 sites) | dropped (`f32_tile` stays: it sizes transfers and offsets) |
| all kernels, CTAs 0–2 | positional `Ct, Kt, Vt` | named CTAs `Ct`, `Kt`, `Vt` |
| reader/writer RTAs 0–1, compute RTA 0 | positional `h, NC` / `NC` | named RTAs `head_idx`, `num_chunks` |
| kernel-side `tt::CBIndex::c_*` constants | magic CB indices | `dfb::<name>` tokens (no CTA carried CB ids) |
| compute `cb_state = (c == 0) ? cb_S0 : cb_S` | runtime CB-id select | `dfb_state = (c == 0) ? dfb_S0 : dfb_S` (`uint32_t` via the `dfb::` conversion) |
| reader `s0_addr != 0` runtime branch | address-as-presence flag | `#ifdef HAS_INITIAL_STATE` (presence is hashed via `tensor_args_t`) |

## Applied Patterns

- [Conditional / optional resource bindings](port_patterns.md#pattern-conditional--optional-resource-bindings): `initial_state` `TensorParameter`/`TensorBinding` and `HAS_INITIAL_STATE` define, emitted together on the reader. The kernel `#ifdef`-gates the `s0` accessor and the read branch; the zero-fill stays as the `#else`.
- [Self-loop DFB binding](port_patterns.md#pattern-self-loop-dfb-binding): 12 compute-only DFBs (`S`, `nm_P_a`, `nm_P_b`, `nm_R_a`, `v_cor`, `k_cum`, `v_prime`, `v_new`, `o_inter`, `intra_v`, `s_upd`, `S_tmp`). Each is real FIFO produce + consume inside compute, one toucher.
- [Two-toucher → 1P+1C](port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split): 12 reader→compute (incl. `S0`) and 2 compute→writer DFBs, plain 1:1.
- **`S0` / `S` census (re-derived after #60011).** `S0`: the reader `reserve_back`/`push_back`s it once (reader `:79,90`); compute `wait_front`s it before the loop and `pop_front`s it at the end of chunk 0 through `dfb_state` (compute `:233,418`). One producer, one consumer, so 1P+1C. `S`: only compute touches it. It `push_back`s the updated state via `dst_dfb` on non-last chunks (`:426-437`) and `pop_front`s it on chunks ≥ 1 through `dfb_state`, so it is a compute self-loop. No DFB needs `allow_instance_multi_binding`, and the brief's multi-binding item for `c_8` is superseded (see Deferred / Flagged).
- [Pass DFB handles directly to LLKs and kernel-lib helpers](port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers): compute passes `dfb_*` token aliases to LLKs and to its own `uint32_t`-taking helpers (`matmul_reconfig_and_init`, `fwd_sub_row`, `fwd_sub_4rows`). Those helpers keep constructing `DataflowBuffer(<uint32_t id>)` internally, as legacy did with `CircularBuffer(x)`.
- `constexpr` metadata (whitelist §A): `constexpr uint32_t f32_tile = get_tile_size(dfb_S0)` / `get_tile_size(dfb_out)` keeps the free-function token form.
- Hardware config: custom Gen1 DM configs copied verbatim. Compute is a Style B literal: `HiFi2`, `Precise`, `enable_32_bit_dest = true`, `double_buffer_dest` default `true` (= `!dst_full_sync_en`), plus explicit `UnpackToSrc` on every consumed Float32 DFB (legacy `Default`), as the validator requires. Compute `opt_level = O3`.

## Deferred / Flagged

- New findings during planning: the custom DM NOC assignment above; the legacy op's Watcher failure; the model test's weak coverage (NC=1, no `initial_state`, self-comparison). Verification adds a scratch bit-exact harness against the legacy outputs (see report).
- **First attempt (2026-10-08) capitulated** on `S`: the planned reader PRODUCER + compute PRODUCER/CONSUMER + `allow_instance_multi_binding` binding was rejected by `ValidateProgramSpec`'s self-loop set-equality rule (`program_spec.cpp:2026-2045`).
- **Re-port (2026-10-15):** PR #60011 split the seed into `S0`, so the brief's `c_8` multi-binding item no longer applies. The audit predates #60011. #60011 changes `device/` and the kernels only in the `S0`/`S` split, and the user asked for the port to proceed on that code, so no re-audit was run. Its only other effect is +64 KiB L1 per core (tracked upstream in #60025).
