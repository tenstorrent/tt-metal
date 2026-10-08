# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/transformer/gated_delta_attn`

- **`GatedDeltaAttnSeqDeviceOperation`** (`device/gated_delta_attn_device_operation.hpp`)
  - Single factory: `GatedDeltaAttnSeqDeviceOperation::create_descriptor`. It is a static on the device-op itself (declared `device_operation.hpp:31`, defined `device/gated_delta_attn_program_factory.cpp:28`). There is no separate factory struct. PD batch #57409 (`f5093e705ae`) deleted `gated_delta_attn_program_factory.hpp` (35 lines) and its `GatedDeltaAttnSeqProgramFactory` type.
- Kernels bound. All three are op-owned, and no other factory binds them:
  - reader: `device/kernels/dataflow/reader_gated_delta_attn.cpp` (RISCV_1)
  - writer: `device/kernels/dataflow/writer_gated_delta_attn.cpp` (RISCV_0)
  - compute: `device/kernels/compute/gated_delta_attn.cpp`
- The op directory has no unreferenced kernel files.
- This is a first audit; there was no earlier `METAL2_PREPORT_AUDIT.md`. Since the PD migration, the op's only commit is `f5093e705ae [Cleanup] Port More Ops to PD (#57409)`, which did not touch the kernels.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/transformer/gated_delta_attn` |
| **Overall** | **GREEN (user waiver)**. The audit found RED on the stale readiness-sheet row from PD batch #57409 only, and the user waived it on 2026-10-07. Every other gate is clear |
| **DOps / Factories** | `GatedDeltaAttnSeqDeviceOperation` → `create_descriptor` (single program, one core per head) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (all includes are `tt_metal/*`; no borrowed kernel files) |
| *Feature Support* — overall | GREEN (no Appendix A entry fires) |
| *Feature Support* — Variadic-CTA | Ok (none) |
| *TTNN Readiness* — `Is able to port?` (the gate) | The sheet says `yes (with PD step)`, but the row is **spreadsheet-broken**: `Concept` and `Custom hash` conflict with the code, and the factory row is phantom. → GATE, routed to the readiness-sheet owner |
| *TTNN Readiness* — Concept (current) | Code: `descriptor`. Sheet: `legacy device-op` (stale) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | **No** in code (removed by `f5093e705ae`). Sheet: `yes` (stale) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; code agrees) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `n/a`; code has none) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`; `gated_delta_attn_nanobind.cpp:42` binds only the function via `bind_function`) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (the sheet's `Porting Target` reads `TBD (study op structure)`, which is stale) |
| *Port work* — Offset base pointer | none (all 11 addresses are clean bases; per-head and per-chunk offsets are `.page_id` tile indices) |
| *Port work* — Tensor bindings (per binding) | All 11 are Case 1 (`TensorAccessor`). `initial_state` is optional |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) at all 11 accessor sites |
| *Port work* — CB endpoints | 1:1 ×12, self-loop ×11, **multi-binding flag ×1** (`c_8` / `cb_S`) |

## Result

**GREEN (user waiver) → brief issued.** On 2026-10-07 the user waived the stale-sheet RED below ("Waive"). The original finding is kept unchanged. The sheet row still needs reconciling by its owner.

*Original finding:* **RED: blocked on the TTNN factory concept gate, because the readiness sheet is broken for this op.** Routed to the **readiness-sheet owner** (Diego, `dgomez@tenstorrent.com`) to reconcile.

The sheet row predates PD-migration batch #57409 (`f5093e705ae`, 2026-09-25). That batch moved this op to `ProgramDescriptor`, deleted the `GatedDeltaAttnSeqProgramFactory` type the row still names, and removed the custom hash the row still reports. **Every other gate is clear:** Device 2.0, Feature compatibility, Offset base pointers, and TensorAccessor 3rd argument (all Class 2). Once the row is reconciled, or the user waives it, the op is a clean port to `ProgramSpecFactoryConcept`. RED at op level; the op has a single program, so there is no subset to offer.

The blocker clears in the sheet, not in the op's code. So per the Red-outcome scoping exception, all seven informational subjects were run in full below, and they will hold on re-audit.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED (spreadsheet-broken).** There is exactly one sheet row for this op (sheet fetched 2026-10-07 in this session):
  - `Device operation` = `GatedDeltaAttnSeqDeviceOperation`, `Factory (variant)` = `GatedDeltaAttnSeqProgramFactory`, `Concept` = `legacy device-op`, `Op Classification` = `Legacy Op`, `Execution Model` = `SPMD`, `Porting Target` = `TBD (study op structure)`, `Custom hash` = `yes`, `Backdoor custom hash` = `no`, `Runtime-args update (get_dynamic_runtime_args)` = `no`, `Override runtime args method?` = `n/a`, `Pybind descriptor` = `no`, `Smuggled pointer` = `no`, `Known op issues` = *(empty)*, `Diego validation` = `no`, `Is able to port?` = `yes (with PD step)`, `TensorParameter relaxation` = `none`, `Op-owned tensors?` = `no`, `Factory definition path` = `…/device/gated_delta_attn_program_factory.hpp`.

  Triggers, each grounded in code:
  1. **Primary-column conflict on `Concept`.** The sheet says `legacy device-op`. The code is `descriptor`: `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` at `device/gated_delta_attn_device_operation.hpp:31`, defined at `device/gated_delta_attn_program_factory.cpp:28`. There is no `create()` and no `override_runtime_arguments`.
  2. **Primary-column conflict on `Custom hash`.** The sheet says `yes`. The code has no `compute_program_hash`; `f5093e705ae` deleted it from `device_operation.cpp` and `.hpp`. The op is not pybound, so the rename caveat does not apply.
  3. **Phantom factory row.** `GatedDeltaAttnSeqProgramFactory` no longer exists. `f5093e705ae` deleted `gated_delta_attn_program_factory.hpp`, which is also the row's `Factory definition path`. Conversely, the code's only program source, `GatedDeltaAttnSeqDeviceOperation::create_descriptor`, has no row.

  Every other primary column cross-checks clean (`get_dynamic_runtime_args`, override, pybind, op-owned tensors). No cross-column invariant is violated. The `yes (with PD step)` verdict reads as "portable once the PD migration lands", and that migration has landed, so this is staleness, not a hidden blocker. This is the same pattern as the other ops from batch #57409.
  **Path forward:** the readiness-sheet owner refreshes the row (Concept → `descriptor`, Custom hash → `no`, factory → `create_descriptor` on the device op, definition path → `gated_delta_attn_program_factory.cpp`), then a cheap re-audit. Alternatively, the user can waive the gate.
- **Device 2.0 (every kernel used):** **GREEN.**
  - **Reader:** uses `Noc noc` (`:89`), twelve `CircularBuffer` wrappers (`:90-101`), `noc.async_read(accessor, cb_obj, size, {.page_id}, {.offset_bytes})` throughout, `noc.async_write_zeros(cb_S_o, …)` + `noc.write_zeros_l1_barrier()` (`:112-113`, both `Noc` methods in `tt_metal/hw/inc/api/dataflow/noc.h:768`), and `noc.async_read_barrier()`.
  - **Writer:** uses `Noc` (`:272`), `CircularBuffer` (`:273-274`), and `noc.async_write(cb_obj, accessor, …)`.
  - **Compute:** all CB sync goes through `CircularBuffer(<idx>).wait_front/pop_front/reserve_back/push_back`. The remaining CB-index arguments are compute LLK calls (`copy_tile`, `matmul_tiles`, `pack_tile`, `reconfig_data_format`, …), not data-movement idioms.
  - The only CB-index free functions are `get_tile_size(cb_S)` (reader `:64`) and `get_tile_size(cb_out)` (writer `:258`), and both are sanctioned.
  - There is no raw `noc_async_*`, no addr-gen, and no raw semaphore address.
- **Feature compatibility:** **GREEN** (no gate fired).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | The `make_cb` `CBDescriptor`s (`program_factory.cpp:77-88`) set no `global_circular_buffer` field; there is no remote-CB idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` is not set (default 0); no CB is buffer-backed |
  | GlobalSemaphore | N/A | No semaphores at all |

- **CB endpoints (GATE-free):** see the census under Port-work. One CB needs the multi-binding flag (`c_8`), and nothing gates.
- **Offset base pointers:** **GREEN.**
  - All 11 address RTAs are delivered as `Buffer*` (`emplace_runtime_args`, `program_factory.cpp:194-207`), so they are clean bases with no host-side fold.
  - Every per-head and per-chunk offset is a tile index computed in-kernel (`h_off_* + c * tiles`, reader `:118-136`; writer `:269-270`) and passed as `.page_id` to the accessor.
  - The op is not listed in `2026-07-19_offset_base_pointers.md`, so this is the "no fold, not in tables" outcome.
- **TensorAccessor 3rd argument:** **GREEN (all Class 2, drop).**
  - There are 11 sites: reader `:71,73,75,77,79,81,83,85,87` and writer `:265,267`. Each is `TensorAccessor(args, addr, f32_tile)`, where `f32_tile = get_tile_size(cb)` on a Float32 CB, so 4096 B.
  - The inputs are validated as FLOAT32 TILE interleaved DRAM (`device_operation.cpp:15-21`). So 4096 is the true page size and already 64-aligned, which makes it `== aligned_page_size`.
  - The outputs are FLOAT32 TILE under the user's `output_mem_config` (default interleaved DRAM). Even if a caller passed a sharded config, the verbatim 4096 still equals the fp32 tile page.
  - This confirms the triage doc's `gated_delta_attn | 2 — Redundant`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (all **Case 1**: each `Buffer*` RTA is fed to a `TensorAccessor` and accessed only through it):
  - reader: `L_unit`, `v_beta_sc`, `k_bd_sc`, `intra_attn`, `q_decay`, `k_decay_t`, `dl_exp`, `L_inv`, `initial_state` (RTAs 2-10, CTA blocks chained from offset 3)
  - writer: `output` (`outputs[0]`), `final_state` (`outputs[1]`) (RTAs 2-3, CTA blocks from offset 3)
  - The `Buffer*` RTAs are patched on cache hit today, so they are not a live hazard. The port moves each to a `TensorParameter` / `TensorBinding`. The address RTAs and the `TensorAccessorArgs<…>` chains disappear.
  - **`initial_state` is optional.**
    - Host side today: it passes a null `Buffer*` (RTA 10 = 0, `program_factory.cpp:190`) and a placeholder `TensorAccessorArgs(nullptr)` (`:152`).
    - Kernel side today: it branches at runtime on `s0_addr != 0` (reader `:106`), and the else branch zero-fills `cb_S`.
    - Port: use the conditional / optional resource binding pattern (`port_patterns.md`). Bind `initial_state` only when present, emit a matching define, and `#ifdef`-gate the accessor and the read branch. Presence is already part of the default program hash (the optional tensor is in `tensor_args_t`), so moving the branch to compile time changes no behavior across cache hits.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** drop `f32_tile` at the 11 sites above (Class 2).
- **CB endpoints:** 24 CBs are allocated. Indices `c_7`, `c_12`, and `c_13` are deliberately not allocated (`program_factory.cpp:98,107`). Census per node; there is one config apart from `NC`:
  - **plain 1:1 (reader PRODUCER → compute CONSUMER):** `c_0` L_unit, `c_1` v_beta_sc, `c_2` k_bd_sc, `c_3` intra_attn, `c_4` q_decay, `c_5` k_decay_t, `c_6` dl_exp, `c_14`–`c_17` L_inv0–3.
  - **plain 1:1 (compute PRODUCER → writer CONSUMER):** `c_24` out, `c_27` final_state.
  - **self-loop (compute only, FIFO produce + consume):** `c_9`, `c_10`, `c_11` (fwd-sub scratch), `c_18` v_cor, `c_19` k_cum, `c_20` v_prime, `c_21` v_new, `c_22` o_inter, `c_23` intra_v, `c_25` s_upd, `c_26` S_tmp.
  - **multi-binding flag: `c_8` (`cb_S`, persistent state).** Two kernels are locked producers:
    - The reader `reserve_back`/`push_back`s the initial state once (reader `:104,115`, including the `async_write_zeros` path).
    - The compute kernel `push_back`s the updated state at step 7b of every non-last chunk (`compute:423-434`, `dst_cb = is_last_chunk ? cb_final_state : cb_S`).
    - The compute kernel is also the locked consumer (`wait_front` `:231`, `pop_front` `:415`).
    - That is ≥2 locked producers, so set `advanced_options.allow_instance_multi_binding = true` on `cb_S`. Compute binds PRODUCER + CONSUMER, and the reader binds PRODUCER. This is FIFO-doubling, not a two-toucher 1P+1C: both producer roles are real `push_back`s.
  - No dead CBs: every allocated index is referenced by at least one bound kernel.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):**
  - `c_8` / `cb_S` (see above). It is visible FIFO-doubling, not a hidden raw co-fill; no semaphores are involved.
  - When `NC == 1`, compute never takes the `cb_S` push branch at runtime. The binding is static, though, so the flag is needed regardless.
  - Quasar uplift will need to refactor this reader-seeds / compute-recycles state loop.
- **Runtime-selected DFB in compute:** `compute:423-424,431,434` picks `dst_cb` between `cb_final_state` (c_27) and `cb_S` (c_8) at runtime, then calls `CircularBuffer(dst_cb)` and `pack_tile(0, dst_cb, t)`. After the swap to `dfb::` tokens, this must stay a runtime selection between two bound DFBs. The `dfb::name` → `uint32_t` cast makes the ternary itself fine. Confirm that the `DataflowBuffer` construction from a runtime index compiles. Do not restructure the loop.
- **Helper functions take CB indices as `uint32_t`:** `fwd_sub_row` / `fwd_sub_4rows` (`compute:87-209`) take `rhs_cb`, `out_cb`, and `inv0..3` as runtime `uint32_t` and construct `CircularBuffer(x)` inside. Passing `dfb::name` tokens through the `uint32_t` cast keeps their signatures unchanged.
- **Cross-op / shared kernels:** none. All three kernels are op-owned and bound by no other factory, so they convert in place.
- **RTA varargs:** none. The reader has 11 fixed RTAs, the writer 4, and compute 1, all read at constant indices. CTAs are `{Ct, Kt, Vt}` plus the accessor-args chains, which go away.
- **Other:**
  - The reader and writer include `api/dataflow/circular_buffer.h`, and the compute kernel includes it too (`compute:30`). The `CircularBuffer` → `DataflowBuffer` swap via `dfb::` is whitelisted.
  - `get_tile_size(cb_S)` / `get_tile_size(cb_out)` feed `constexpr uint32_t f32_tile` (reader `:64`, writer `:258`). After the 3rd-arg drop, `f32_tile` still sizes every `noc.async_read/write` and `offset_bytes`. Moving it onto the DFB object (whitelist rule 7) must stay usable in a constant expression, or the porter keeps the free form on the token. Confirm which works rather than swapping blind.
  - The header comment at `device_operation.hpp:26-30` says the eleven buffer bindings are the whole cache-hit refresh. That matches plain `ProgramSpecFactoryConcept`.
  - Runtime verification: device runs go through `models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py`. `tests/ttnn/unit_tests/operations/transformers/test_chunk_gated_delta_rule.py` covers only the golden function. The op uses one core per head (`BH` ≤ grid), so check the test's `BH` against the 8×8 Wormhole here.
  - Do not look at `experimental/quasar/` for precedent.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.**
  - Function-call escapes: none. Every include is `tt_metal/*`.

    | Op kernel | Donor file | Class | Status |
    |---|---|---|---|
    | reader (own) | `api/dataflow/{dataflow_api,noc,circular_buffer}.h`, `api/tensor/noc_traits.h` | `tt_metal/*` | ✓ |
    | writer (own) | same set | `tt_metal/*` | ✓ |
    | compute (own) | `api/compute/*`, `api/dataflow/circular_buffer.h` | `tt_metal/*` | ✓ |

  - Borrowed kernel files: none.
- **Relaxation candidates:** none. The custom hash that `f5093e705ae` removed hashed every attribute and every tensor in full (`L_unit` … `initial_state`), which is equivalent to the default hash, so it reveals no narrower dependency.
- **TTNN factory analysis:**
  - Concept: `descriptor` (`create_descriptor` on the device op).
  - No op-owned tensors (single `ProgramDescriptor`).
  - No custom or backdoor hash (removed in #57409).
  - No `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybound `create_descriptor` (`gated_delta_attn_nanobind.cpp:42` uses `bind_function`).
  - Target concept: `ProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **`compute_kernel_config` is ignored but hashed.** The factory hardcodes `math_fidelity = HiFi2`, `fp32_dest_acc_en = true`, and `math_approx_mode = false` (`program_factory.cpp:179-183`). It never reads `attrs.compute_kernel_config`, which the host resolves at `gated_delta_attn.cpp:24-30` and the default hash keys on. A user-supplied config is silently dropped, and differing configs cause needless cache misses.
- **`output_mem_config` is unvalidated.** `validate_on_program_cache_miss` checks every input but never checks the output memory config, so an L1 or sharded output config is accepted. The writer's accessor would handle it, but no test covers that path.
- **`num_chunks == 0` would hang.** With `NC = 0`, compute never enters the chunk loop, so it never pushes `cb_final_state`. The writer then blocks forever in `cb_final_state_o.wait_front(state_tiles)` (writer `:288`). Validation only checks shapes against `NC`; it does not require `NC ≥ 1` (`device_operation.cpp:36-73`).
- **Stale kernel header comments.** The reader and writer file headers list "Compile-time args: Ct, Kt, Vt" only (reader `:19`, writer `:234`), but both also consume the appended `TensorAccessorArgs` chains.

## Questions for the user

1. **Waiver?** The only RED was the stale sheet row from PD batch #57409 (Concept / Custom hash / phantom factory). **Answered 2026-10-07: "Waive".** `METAL2_PORT_BRIEF.md` was issued.

## Recipe notes

- As in other #57409 ops, `create_descriptor` is a static on the DeviceOperation with no factory type, so the "factory-set match" check pairs the sheet's factory row against a device-op method. I treated that as one factory (phantom name, real program).
- The CB endpoints table covers a compute kernel that is both the locked consumer *and* a locked producer of a CB another kernel seeds (`cb_S`: the reader seeds it, compute recycles it). It does so only through the "≥2 locked producers" clause, and that clause reads as if it describes two *different* producers co-filling. A one-line example of the "seed then recycle" state-loop shape under Multi-binding would make this call less of a judgment.
