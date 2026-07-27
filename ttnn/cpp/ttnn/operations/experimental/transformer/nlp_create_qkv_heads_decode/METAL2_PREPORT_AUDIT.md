# Metal 2.0 Audit Findings — `experimental/transformer/nlp_create_qkv_heads_decode`

Single device operation, three program factories selected by input layout:

- **`NLPCreateQKVHeadsDecodeDeviceOperation`**
  - `NLPCreateQKVHeadsDecodeInterleavedProgramFactory` (`nlp_create_qkv_heads_decode_interleaved_program_factory.cpp`) — interleaved (non-sharded) input
  - `NLPCreateQKVHeadsDecodeShardedProgramFactory` (`nlp_create_qkv_heads_decode_sharded_program_factory.cpp`) — width-sharded input, whole-grid
  - `NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory` (`nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp`) — width-sharded input, sub-core-grids

Kernels (all owned by this op, under `device/kernels/`):
- `reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` — used by the interleaved factory (reader + writer instances)
- `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` — used by the sharded factory
- `reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp` — used by the subcoregrid factory

`select_program_factory` (`device/nlp_create_qkv_heads_decode_device_operation.cpp:12`): sharded input → subcoregrid if `input_on_subcoregrids` else sharded; non-sharded → interleaved. All three share the one `DeviceOperation` and the `q/k/v` output-tensor structure, so they are audited together as one porting unit; per-factory attribution is retained where findings differ.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/` |
| **Overall** | **GREEN** — all gates cleared; brief issued |
| **DOps / Factories** | `NLPCreateQKVHeadsDecodeDeviceOperation` → Interleaved · Sharded · ShardedSubcoregrid |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all three op kernels + donor `tt_memmove` are Device 2.0 native |
| *Prereqs* — Cross-op escapes | Ok — one shared-lib function-call escape (`tt_memmove`, Device 2.0 native); no file-path escapes |
| *Feature Support* — overall | **GREEN** — all Appendix A entries N/A |
| *Feature Support* — Variadic-CTA | Ok — fixed `tensor_args_t`, CTAs read at constexpr offsets only |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (all 3 factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 3 factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not a WorkloadDescriptor op) |
| *TTNN Readiness* — Is safe to port? | Yes (all 3 rows; `Smuggled pointer = no`) |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash` override) |
| *TTNN Readiness* — Runtime-args update | No (no `get_dynamic_runtime_args` / `override_runtime_arguments`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind uses `bind_function`, no descriptor binding) |
| *TTNN Readiness* — Op-owned tensors | No (blank on sheet; `descriptor` concept can't carry them — invariant holds) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (no host-folded offset; base delivered clean, offsets separate/on-device) |
| *Port work* — Tensor bindings (per binding) | q/k/v out: clean (borrowed-DFB) · input: **Case 1** (interleaved) / **Case 2** (sharded, subcoregrid) · batch_offset: **Case 1** |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (all `TensorAccessor` sites are 2-arg) |
| *Port work* — CB endpoints | self-loop · 1P+1C · (sharded batch-offset CBs need reconciliation — see below) |

**CB endpoints** are dispositions, not gates. Every out-of-window CB here has a port-time resolution. Recorded per `(CB, config)` below.

## Result

**GREEN → brief issued.** All five gate-bearing subjects cleared for all three factories: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? = yes` ×3), Offset base pointers ✓, TensorAccessor 3rd argument ✓. The op is a `descriptor`-concept, three-factory data-movement op targeting `MetalV2FactoryConcept`. Port work is ordinary tensor-binding translation (one Case-2 raw-pointer input on the two sharded factories, Case-1 elsewhere) plus CB-endpoint dispositions. One pre-existing factory inconsistency in the sharded batch-offset CB wiring is surfaced as a question/anomaly — it does not block the port.

`METAL2_PORT_BRIEF.md` is emitted alongside this file.

> **PORT ADDENDUM (2026-07-23):** the audit's "q/k/v out: clean (borrowed-DFB)" disposition was found
> during the port to be blocked by a Metal 2.0 **framework bug** — borrowed DFBs get a corrupted
> device-side base in multi-work-unit programs, which the sharded/subcoregrid `!overlap` layout requires.
> Worked around by binding the outputs as Case-2 `TensorParameter`s (`get_bank_base_address`) instead.
> The interleaved factory (single work unit) keeps borrowed DFBs. See `METAL2_PORT_REPORT.md`. This is a
> gap the audit's gate set did not catch: a "clean borrowed-DFB output" is **not** unconditionally clean —
> it depends on the factory being single-work-unit until the framework bug is fixed.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** All three factory rows on the readiness sheet report `Is able to port? = yes`, with `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no` (both the `get_dynamic_runtime_args` and `PD override_runtime_args` columns), `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `TensorParameter relaxation = none`, `Op-owned tensors? = ` (blank). Cross-check clean:
  - `Concept = descriptor` — confirmed: each factory defines `create_descriptor(...)` returning a `tt::tt_metal::ProgramDescriptor`.
  - `Custom hash = no` — confirmed: no `compute_program_hash` anywhere in the op.
  - `Runtime-args update = no` — confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments`.
  - `Pybind descriptor = no` — confirmed: `nlp_create_qkv_heads_decode_nanobind.cpp:22` binds via `ttnn::bind_function<...>`, no `create_descriptor` binding.
  - `Op-owned tensors? = no` — consistent with `descriptor` concept.
- **Device 2.0 (every kernel used):** **GREEN.** Every kernel this op instantiates is structurally Device 2.0 (`Noc`/`async_read`, `CircularBuffer` wrappers, `CoreLocalMem`, `TensorAccessor`, and (sharded/subcoregrid) `UnicastEndpoint`). Donor `tt_memmove(Noc, ...)` is Device 2.0 native.
- **Feature compatibility:** all Appendix A entries scanned; none fire (no GlobalCircularBuffer, no `address_offset`, no GlobalSemaphore, no variable-count CTAs).
- **CB endpoints (GATE-free):** output CBs → 1P+1C (dual-instance reader+writer); scratch CBs → self-loop. Sharded batch-offset CBs are a probable pre-existing wiring bug — surfaced under Questions.
- **Offset base pointers:** **GREEN.** No `->address()` fold anywhere.
- **TensorAccessor 3rd argument:** **GREEN.** Every construction is the 2-arg form.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding; classification varies per factory):
  - **q/k/v output** — audit disposition **clean (borrowed-memory DFB)**. *(See PORT ADDENDUM: reclassified to Case-2 `TensorParameter` on the sharded/subcoregrid factories to work around the framework bug.)*
  - **input_tensor** — **Interleaved: Case 1** (via `TensorAccessor`); **Sharded / Subcoregrid: Case 2** (raw pointer via `get_bank_base_address`, keep the hand-rolled `UnicastEndpoint` reads).
  - **batch_offset** (sharded + subcoregrid only) — **Case 1** (via `TensorAccessor`), gated by `use_batch_offset`.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** output CBs 1P+1C; interleaved scratch c_0/c_1 self-loop (aligned path); subcoregrid batch-offset c_15/c_14 self-loop; sharded batch-offset c_14/c_15 needs reconciliation (Questions #1).

## Questions for the user

1. **Sharded batch-offset CB wiring (`c_14` vs `c_15`):** the writer kernel's CTA is never switched to `batch_offset_cb_index_writer` (`c_14`), so both reader and writer use `c_15` and `c_14` is dead (vs the subcoregrid factory which switches correctly). Is the intended design two separate scratch CBs (a bug in the sharded factory)? *(PORT: confirmed a bug; the one-line switch was applied.)*

## Misc anomalies  *(team-only, non-gating)*

- **Sharded factory: writer batch-offset CB index never switched → `c_14` dead, `c_15` double-produced.** *(PORT: fixed with the one-line switch.)*
- **Subcoregrid factory: `v_shard_spec` derived from `output[0]` (q), not `output[2]` (v).** Cosmetic over-allocation. *(PORT: moot — the v output is now a `TensorParameter` sized from `output[2]`.)*

## TTNN factory analysis

`Concept = descriptor` ×3; `Op-owned tensors? = no`; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`; not a WorkloadDescriptor. `Model = llama`, `Porting Classification = Simple Port`. Target concept `MetalV2FactoryConcept`.
