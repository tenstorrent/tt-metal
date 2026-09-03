# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/transpose`

Single device-operation directory. One `DeviceOperation` with eight program factories:

- **`TransposeDeviceOperation`** (`device/transpose_device_operation.{hpp,cpp}`)
  - `TransposeCNProgramFactory` (`transpose_cn_program_factory.cpp`)
  - `TransposeHCRMProgramFactory` (`transpose_hc_rm_program_factory.cpp`)
  - `TransposeHCTiledInterleavedProgramFactory` (`transpose_hc_tiled_interleaved_program_factory.cpp`)
  - `TransposeHCTiledProgramFactory` (`transpose_hc_tiled_program_factory.cpp`)
  - `TransposeWHProgramFactory` (`transpose_wh_program_factory.cpp`) — handles both tiled and row-major
  - `TransposeWHShardedProgramFactory` (`transpose_wh_sharded_program_factory.cpp`)
  - `TransposeHCShardedProgramFactory` (`transpose_hc_sharded_program_factory.cpp`) — **gated**
  - `TransposeWHShardedRMProgramFactory` (`transpose_wh_sharded_rm_program_factory.cpp`) — **gated**

**Unreferenced kernel file (out of scope):** `device/kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp` is instantiated by no factory (dead code in the directory; the WH factory uses the `_start_id` variants). Its contents were not audited.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/transpose` |
| **Overall** | **RED at op level; subset {CN, HC-RM, HC-Tiled-Interleaved, HC-Tiled, WH, WH-Sharded} is clear** |
| **DOps / Factories** | `TransposeDeviceOperation` → 8 factories (6 clear, 2 gated) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes (GREEN)** — all referenced own + donor kernels are Device 2.0 |
| *Prereqs* — Cross-op escapes | Ok (workable) — 3 broadly-shared donor kernels + 1 in-family helper, all Device 2.0 |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (N/A) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No for 2 factories** (`Runtime-args update` + `Is safe to port?`); **Yes for 6** |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 8) |
| *TTNN Readiness* — Secretly SPMD | N/A (no `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | `yes` (6 clean) / **`no` (HC-Sharded, WH-Sharded-RM → readiness-sheet owner)** |
| *TTNN Readiness* — Custom hash | No (all) |
| *TTNN Readiness* — Runtime-args update | No (6 clean) / **Yes (HC-Sharded, WH-Sharded-RM → `get_dynamic_runtime_args`)** |
| *TTNN Readiness* — Pybind `create_descriptor` | No (function-only nanobind) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (GREEN) — no host-folded offsets |
| *Port work* — Tensor bindings | Case 1 (interleaved factories, in+out) / clean borrowed-DFB (WH-Sharded) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — every accessor is 2-arg |
| *Port work* — CB endpoints | 1:1 (most) / self-loop (`c_1` scratch, `c_24` tilize) / **dead-CB drop (`c_25` in WH-RM)** |

## Result

**RED at op level; subset {CN, HC-RM, HC-Tiled-Interleaved, HC-Tiled, WH, WH-Sharded} is clear.**

Two of the eight factories are blocked by the **TTNN factory-concept gate** (`Is able to port? == no`):

- **`TransposeHCShardedProgramFactory`** and **`TransposeWHShardedRMProgramFactory`** each fail on **two** conjuncts:
  - `Runtime-args update == yes` — both opt into the device-op-level `get_dynamic_runtime_args` fast-path hook (#48928). → **TTNN / ProgramDescriptor-migration team**; the gate lifts once the runtime-args-update infra ships in the Metal 2.0 path.
  - `Is safe to port? == no` — the readiness-sheet owner's correctness call. → **readiness-sheet owner** (Diego); the buggy/at-risk PD-migration state must be reconciled before these two port.

This is a **config-scoped gate**: the six non-`get_dynamic_runtime_args` factories clear every gate, so a **subset port of those six is available now**, and a brief is issued for them (`METAL2_PORT_BRIEF.md`). The two sharded RM factories re-audit once their gate conjuncts clear.

**Coupling note for a subset port:** `get_dynamic_runtime_args` is declared once on the shared `TransposeDeviceOperation` (in `transpose_hc_sharded_program_factory.cpp:432`) and returns non-empty only for the two gated factories (returns `{}` for the other six — confirmed in code). Porting the six clean factories leaves the shared device op and its hook in place for the two that remain unported; this is expected for a subset port but should be carried as a heads-up.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** RED at op level (config-scoped). Readiness-sheet `Concept = descriptor` for all 8, `Custom hash = no`, `Pybind descriptor = no`, `Smuggled pointer = no`. `Is able to port? = yes` for CN / HC-RM / HC-Tiled-Interleaved / HC-Tiled / WH / WH-Sharded; `= no` for HC-Sharded and WH-Sharded-RM (`Runtime-args update = yes` **and** `Is safe to port? = no`). **Cross-check (clean):** all factories define `create_descriptor()` returning `ProgramDescriptor` (`descriptor` ✓); no `compute_program_hash` override anywhere (`Custom hash = no` ✓); nanobind binds only the `transpose` free function, no `create_descriptor` (`Pybind = no` ✓); `get_dynamic_runtime_args` is defined and dispatches to exactly `TransposeHCShardedProgramFactory` + `TransposeWHShardedRMProgramFactory` (`transpose_hc_sharded_program_factory.cpp:432-456`), matching the sheet's two `Runtime-args update = yes` rows exactly. No `WorkloadDescriptor`/`create_mesh_workload`. Cross-column invariants hold (no op-owned tensors on a `descriptor` op). **Sheet is trustworthy.**
- **Device 2.0 (every kernel used):** **GREEN.** Every referenced kernel — own and donor — uses Device 2.0 idioms: `Noc noc;`, `DataflowBuffer dfb(...)`, `TensorAccessor(args, addr)`, and object-method `dfb.get_read_ptr()` / `dfb.get_write_ptr()`. No Device 1.0 addr-gen (`InterleavedAddrGen`/`ShardedAddrGen`/`InterleavedPow2AddrGen*`), no raw `noc_async_read`/`noc_async_write`/`get_noc_addr_from_bank_id`, no raw sem addresses. The only CB free-function is `get_local_cb_interface(cb_id_out).fifo_page_size` in the eltwise/unary donor writer (`writer_unary_interleaved_start_id.cpp:19`) — **sanctioned** per the Green rule (not a holdover). Donor kernels checked: `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`, `eltwise/unary/.../reader_unary_sharded.cpp`, `data_movement/sharded/.../writer_unary_sharded.cpp` — all Device 2.0. (Kernels for the two gated factories — `reader/writer_unary_transpose_hc_sharded_rm.cpp`, `reader/writer_unary_transpose_wh_sharded_rm.cpp` — are also Device 2.0, so Device 2.0 is not an additional blocker on them.)

- **Feature compatibility:** every Appendix A entry is absent (all N/A — a clean scan).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`/`CreateGlobalCircularBuffer` anywhere |
  | CBDescriptor `address_offset` (non-zero) | N/A | The four `UpdateDynamicCircularBufferAddress` references are **comments** describing the standard borrowed-memory (`.buffer =`) re-application on cache hit, **not** a non-zero `address_offset` / `set_address_offset`. No Type-3 offset. |
  | GlobalSemaphore | N/A | no semaphores of any kind in these kernels |
  | Variable-count compile-time arguments (CTA varargs) | N/A | no runtime-varying CTA loop in any *referenced* kernel. (The one `get_compile_time_arg_val(src_args.next_compile_time_args_offset())` is a fixed accessor-args boundary in the **unreferenced** dead kernel `reader_unary_transpose_wh_interleaved.cpp`.) |

- **CB endpoints (GATE-free):** classified per `(CB, config)` for the clean subset; all resolve without a flag. See Port-work summary. No multi-binding anywhere.
- **Offset base pointers:** **GREEN.** No factory contains a `->address()` expression at all — bases reach kernels either via the **`Buffer*`-binding form** (`emplace_runtime_args(core, {input_tensor.buffer(), ...})`, framework auto-registers a `BufferBinding`; the kernel gets a clean `uint32_t` base it feeds to a `TensorAccessor`) or via **borrowed-memory CBs** (`.buffer = ...buffer()`). No host-folded `base + offset`. Kernel-side `dfb.get_write_ptr() + offset` arithmetic is L1-scratch addressing inside a CB, not a device-tensor base+offset fold; `TensorAccessor` `page_id`/`offset_bytes` are ordinary accessor addressing. Not present in the offset-base triage tables — consistent.
- **TensorAccessor 3rd argument:** **GREEN.** Every `TensorAccessor(...)` construction across all referenced kernels is **2-arg** (`TensorAccessor(args, base_addr)`); no page-size 3rd argument is passed anywhere. Nothing to drop or gate. Not present in the 3rd-arg triage table — consistent.

## Port-work summary  *(mirrors the brief; scoped to the clean 6-factory subset)*

- **Tensor bindings** (per binding):
  - **Interleaved factories** — CN / HC-RM / HC-Tiled-Interleaved / HC-Tiled / WH (tiled + RM): **input** and **output** are both **Case 1** (delivered today via the `Buffer*`-binding form, consumed through a `TensorAccessor`). Express each as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`. Mechanical.
  - **WH-Sharded**: input CB (`c_0`) and output CB (`c_16`) are **borrowed-memory DFBs** (`.buffer = input/output ...buffer()`) → **clean** (causal-link gate); port via `DataflowBufferSpec::borrowed_from`. No Case-1/2 work.
- **TensorParameter relaxation:** none (`TensorParameter relaxation = none` on every row).
- **TensorAccessor 3rd arg:** none — no sites.
- **CB endpoints** (per `(CB, config)`):
  - **CN:** `c_0` — 1:1 (reader P → writer C).
  - **HC-RM:** `c_0` — 1:1 (reader P → writer C).
  - **HC-Tiled-Interleaved:** `c_0` — 1:1 (reader P → writer C); `c_1` (padding) — 1:1 (reader P → writer C), **only when `needs_padding`** (`C % tile_height != 0`).
  - **HC-Tiled:** `c_0` — 1:1 (reader P → donor unary writer C); `c_1` (scratch) — **self-loop** (touched only by the reader via `dfb_scratch.get_write_ptr()`), **only when `misaligned`** (`dst alignment > sub_tile_line_bytes`, e.g. Blackhole DRAM 64B).
  - **WH (tiled):** `c_0` — 1:1 (reader P → compute C); `c_16` — 1:1 (compute P → donor unary writer C).
  - **WH (row-major):** `c_0` — 1:1 (reader P → compute C); `c_16` — 1:1 (compute P → writer C); `c_24` (im / tilize) — **self-loop** (produced and consumed within the compute kernel); **`c_25` (im2) — DEAD-CB drop** (see below).
  - **WH-Sharded:** `c_0` — 1:1 (donor reader P → compute C, borrowed); `c_16` — 1:1 (compute P → donor writer C, borrowed).
  - **Dead-CB drop:** `c_25` (`im2`) allocated in `transpose_wh_program_factory.cpp:203-213` (row-major branch) is referenced by **no** kernel. `transpose_wh_rm.cpp` uses `c_25` as `cb_tilize` **only under `#ifdef SHARDED`** — a define set solely by the *gated* `TransposeWHShardedRMProgramFactory`, never by `TransposeWHProgramFactory`. In the WH factory's RM path `cb_tilize = c_24`, so `c_25` is unused. Corroborated by the factory's own `// TODO REMOVE` comment (line 202). **Confirmed dead → porter drops the allocation.**

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader, no ≥3-toucher CB in the clean subset.
- **Cross-op / shared kernels (borrowed kernel files, port-together sets):**
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — donor writer for **HC-Tiled** (consumes `c_0`) and **WH tiled** (consumes `c_16`). **Broadly shared** (~42 host `.cpp` files reference it). Its Metal 2.0 rewrite (CB→DFB, named tokens) is a single change all co-borrowers adopt together.
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` — donor reader for **WH-Sharded** (produces borrowed `c_0`). Broadly shared (~17 host files).
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — donor writer for **WH-Sharded** (consumes borrowed `c_16`). Broadly shared (~15 host files).
- **RTA varargs:** none in the clean subset — every referenced kernel reads a fixed set of distinctly-named args at constant indices (or a fixed `arg_idx++` run); no loop-indexed or data-selected reads.
- **Shared device-op `get_dynamic_runtime_args` coupling:** the hook lives on the shared `TransposeDeviceOperation` and returns `{}` for the six clean factories; a subset port leaves it in place for the two gated factories.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ⚠ workable** — three broadly-shared donor kernel files (file-path instantiation) + one in-family function-call escape; all donors are Device 2.0, so nothing sequence-blocks beyond the standard port-together coupling.

**Function-call escape:** the transpose dataflow kernels `#include "ttnn/operations/data_movement/common/kernels/common.hpp"` and call `tt::data_movement::common::noc_async_read_sharded` / `noc_async_write_sharded` / `fill_with_val` / `round_up`. This is an **in-family** (`data_movement/common`) shared helper pool. Signature shapes are Device 2.0 native — `(Noc& noc, uint32_t l1_addr, const TensorAccessor& s, ...)` (TensorAccessor Shape 1 ✓, `Noc&` ✓). No pre-Device-2.0 donor shapes → no Device 2.0 donor gate. Includes appear in: `reader_unary_transpose_cn...`, `reader/writer_unary_transpose_wh_interleaved_start_id_rm`, `reader_unary_transpose_hc_interleaved_tiled_padding_aware`, `reader_unary_transpose_hc_interleaved_partitioned_rm`, `writer_unary_transpose_hc_interleaved_start_id_rm`, `writer_unary_transpose_hc_interleaved_tiled_padding_aware`.

**Borrowed kernel files (file-path instantiation):**

| Kernel file | Owning family | Instantiated by transpose factory | Broadly shared? |
|---|---|---|---|
| `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | eltwise/unary (cross-family) | HC-Tiled, WH tiled | Yes — ~42 host `.cpp` files |
| `eltwise/unary/.../reader_unary_sharded.cpp` | eltwise/unary (cross-family) | WH-Sharded | Yes — ~17 host `.cpp` files |
| `data_movement/sharded/.../writer_unary_sharded.cpp` | data_movement/sharded (cross-op, same family) | WH-Sharded | Yes — ~15 host `.cpp` files |

Each forms a Metal 2.0 port-together set with its co-borrowers; sequence the shared rewrite as one unit.

### TTNN factory analysis (sheet-derived, with cross-check)

- **Concept:** `descriptor` (all 8 factories) — verified in code (`create_descriptor` → `ProgramDescriptor`).
- **Custom hash:** No — no `compute_program_hash` override in the op.
- **Runtime-args update:** Yes for HC-Sharded + WH-Sharded-RM only (device-op `get_dynamic_runtime_args`, #48928); No for the other six.
- **Pybind `create_descriptor`:** No — nanobind exposes only the `transpose` function.
- **Op-owned tensors:** No.
- **Target concept (cleared factories):** `MetalV2FactoryConcept` (no op-owned tensors).

## Misc anomalies  *(team-only, non-gating)*

- **Dead CB `c_25` (im2) in the WH factory RM path** — `transpose_wh_program_factory.cpp:203-213`, carries a `// TODO REMOVE` (line 202); no kernel references it. (Handled as a CB-endpoints dead-CB drop in the subset port; noted here for the ops team as a latent allocation waste.)
- **Unreferenced kernel file** `device/kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp` — present in the directory but instantiated by no factory (dead code). Candidate for removal by the ops team.

## Per-DeviceOperation attribution

Single `TransposeDeviceOperation`; per-factory verdicts:

| Factory | `Is able to port?` | Blocker (if any) |
|---|---|---|
| `TransposeCNProgramFactory` | yes | — (in clean subset) |
| `TransposeHCRMProgramFactory` | yes | — (in clean subset) |
| `TransposeHCTiledInterleavedProgramFactory` | yes | — (in clean subset) |
| `TransposeHCTiledProgramFactory` | yes | — (in clean subset) |
| `TransposeWHProgramFactory` | yes | — (in clean subset) |
| `TransposeWHShardedProgramFactory` | yes | — (in clean subset) |
| `TransposeHCShardedProgramFactory` | **no** | `Runtime-args update` (→ TTNN/PD-migration) + `Is safe to port? = no` (→ readiness-sheet owner) |
| `TransposeWHShardedRMProgramFactory` | **no** | `Runtime-args update` (→ TTNN/PD-migration) + `Is safe to port? = no` (→ readiness-sheet owner) |

## Recipe notes

- **Device-op-level `get_dynamic_runtime_args` vs. per-factory gate attribution.** The `Runtime-args update` gate conjunct is described as a per-factory property, but the hook it keys on (`get_dynamic_runtime_args`) is necessarily declared once on the shared `DeviceOperation`. Here it dispatches on `select_program_factory` and returns non-empty for only 2 of 8 factories. The readiness sheet correctly scopes `Runtime-args update = yes` to those 2, and the cross-check ("grep the factory for `get_dynamic_runtime_args`") resolved cleanly once I read the hook *body* rather than just its presence. Worth an explicit line in the TTNN-factory-concept cross-check guidance: for a shared hook, verify *which factories it actually services*, not merely that the symbol exists — otherwise a naive grep would over-attribute the gate to all 8.
