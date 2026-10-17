# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/transformer/gated_delta_attn`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

> **Superseded item (2026-10-15, port step).** The `c_8` / `cb_S` multi-binding item under *CB endpoints* and *Watch for* describes the op before PR #60011. That PR moved the reader's seed into a new `c_7` / `cb_S0` (reader PRODUCER → compute CONSUMER), leaving `c_8` a compute-only self-loop. The audit was not re-run; the port re-derived the census (see `METAL2_PORT_PLAN.md`). The rest of the brief is unchanged.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section; hash from the `Port_Recipe` checkout)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-07) still describes this op as it was before PR #57409:

- `Concept` = `legacy device-op`
- `Custom hash` = `yes`
- factory `GatedDeltaAttnSeqProgramFactory`, at the now-deleted `gated_delta_attn_program_factory.hpp`
- `Is able to port?` = `yes (with PD step)`

The code is already on a direct `create_descriptor` with no custom hash, so the audit flagged the sheet as broken. On 2026-10-07 the user waived that gate ("Waive"), since the problem is only the out-of-date sheet. Every other primary column cross-checked clean. The sheet refresh is still owed to the readiness-sheet owner.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `GatedDeltaAttnSeqDeviceOperation` itself (`device/gated_delta_attn_device_operation.hpp:31-32`, body `device/gated_delta_attn_program_factory.cpp:28-216`), with no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`; the framework's binding refresh covers the cache hit. The header comment at `device_operation.hpp:26-30` says the same: every non-address arg (head index, `num_chunks`) derives from hashed attributes and specs.
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` (cell is `none`) and `get_dynamic_runtime_args` (absent). The op also has no custom hash (#57409 removed it) and no pybound `create_descriptor`. The only nanobind binding is the user function (`gated_delta_attn_nanobind.cpp:42`, `bind_function`), so no pybind line needs deleting.

## Construct — to do

**Tensor bindings** (all **Case 1**, via `TensorAccessor`). Today each tensor reaches its kernel two ways:

- as a `Buffer*` RTA (`program_factory.cpp:194-207`)
- as a `TensorAccessorArgs(buf).append_to(...)` CTA block chained from CTA offset 3 (`:143-156`)

Express each as a `TensorParameter` / `TensorBinding`. The kernel then builds `TensorAccessor(tensor::<name>)`, and the address RTAs and `TensorAccessorArgs<…>` chains both go away.

- reader: `L_unit` (RTA 2), `v_beta_sc` (3), `k_bd_sc` (4), `intra_attn` (5), `q_decay` (6), `k_decay_t` (7), `dl_exp` (8), `L_inv` (9)
- reader: **`initial_state` (RTA 10) is optional.** Use the conditional / optional resource binding pattern (`port_patterns.md`).
  - Today the host passes a null `Buffer*` (`:190`) and a placeholder `TensorAccessorArgs(nullptr)` (`:152`). The reader branches at runtime on `s0_addr != 0` (reader `:106`), and the else branch zero-fills `cb_S` via `noc.async_write_zeros` (`:112-113`).
  - Port: bind `initial_state` only when present, emit a matching define through `compiler_options.defines`, and `#ifdef`-gate the `s0` accessor and the read branch. Keep the zero-fill branch for the absent case.
  - Presence is already part of the default program hash, so the compile-time split changes no behavior.
- writer: `output` (`outputs[0]`, RTA 2), `final_state` (`outputs[1]`, RTA 3)

The remaining RTAs are named, not varargs. Reader: `head_idx` (0), `num_chunks` (1). Writer: `head_idx`, `num_chunks`. Compute: `num_chunks` (0). CTAs `Ct`, `Kt`, and `Vt` are named on all three kernels.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** drop `f32_tile` (Class 2; 4096 B == fp32 tile `aligned_page_size`) at reader `:71,73,75,77,79,81,83,85,87` and writer `:265,267`. `f32_tile` itself stays: it still sizes every `noc.async_read/write` and `offset_bytes`.

**CB endpoints** (24 DFBs; indices 7, 12, 13 are not allocated):

- **1P+1C, reader PRODUCER → compute CONSUMER:** `c_0`–`c_6`, `c_14`–`c_17`.
- **1P+1C, compute PRODUCER → writer CONSUMER:** `c_24` (out, 2-deep), `c_27` (final_state).
- **self-loop (compute PRODUCER + CONSUMER):** `c_9`, `c_10`, `c_11`, `c_18`, `c_19`, `c_20`, `c_21`, `c_22`, `c_23`, `c_25`, `c_26`.
- **multi-binding advanced option (`allow_instance_multi_binding = true`): `c_8` (`cb_S`).**
  - The reader is a locked producer: it seeds the initial state (`reserve_back`/`push_back`, reader `:104,115`).
  - Compute is a locked consumer (`compute:231,415`) and also a locked producer: it recycles the updated state into `cb_S` on every non-last chunk (`compute:423-434`).
  - Bind the reader PRODUCER and compute PRODUCER + CONSUMER. That is two real `push_back`ers, so the census can't fit 1P+1C. This is not a stacking error: `c_8` gets the flag only and is not self-looped.
- No dead CBs.

## Watch for

- **CB endpoints (multi-binding):**
  - `c_8` / `cb_S`: visible FIFO-doubling (reader seed + compute recycle). There is no hidden raw co-fill and no semaphores. Re-derive the census yourself before setting the flag.
  - The flag is needed even when `NC == 1`, because the binding is static.
- **Runtime-selected DFB in compute:** step 7b picks `dst_cb = is_last_chunk ? cb_final_state : cb_S` at runtime (`compute:422-434`), then calls `CircularBuffer(dst_cb)` and `pack_tile(0, dst_cb, t)`. Keep it a runtime choice between the two bound DFBs; `dfb::name`'s `uint32_t` cast makes the ternary fine. Confirm the `DataflowBuffer` construction from the runtime index compiles. Do not restructure the loop.
- **CB-index helpers in compute:** `matmul_reconfig_and_init`, `fwd_sub_row`, and `fwd_sub_4rows` (`compute:68-209`) take CB indices as runtime `uint32_t` and construct `CircularBuffer(x)` internally. Pass `dfb::` tokens through the `uint32_t` cast and leave the signatures alone.
- **Cross-op / shared kernels:** none. All three kernels are op-owned and bound by no other factory, so they convert in place.
- **RTA varargs:** none.
- **`get_tile_size(cb)` feeding a `constexpr`:** reader `:64` (`cb_S`) and writer `:258` (`cb_out`) compute `constexpr uint32_t f32_tile`. Moving the lookup onto the DFB object (whitelist rule 7) must stay usable in a constant expression, or keep the free form on the token. Confirm which works rather than swapping blind.
- **Includes:** all three kernels include `api/dataflow/circular_buffer.h` (reader `:24`, writer `:239`, compute `:30`). Swap them to `DataflowBuffer` with the `CircularBuffer` → `dfb::` conversion.
- **Runtime verification:**
  - The device test is `models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py`. `tests/ttnn/unit_tests/operations/transformers/test_chunk_gated_delta_rule.py` covers only the golden function.
  - The op places one core per head (column-major, `program_factory.cpp:61-64`), so check the test's head count against the 8×8 Wormhole here before porting.
  - Inputs must be FLOAT32 TILE DRAM with `chunk_size = key_dim = val_dim = 128`.
- **Leave alone (team-only anomalies, not port work):** the hardcoded compute config that ignores `attrs.compute_kernel_config` (`program_factory.cpp:179-183`), and the `NC == 0` hang. Carry them over verbatim.
- Do not look at `experimental/quasar/` for precedent.
