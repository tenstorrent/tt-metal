# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads`

- **`NLPConcatHeadsDeviceOperation`** (`device/nlp_concat_heads_device_operation.{hpp,cpp}`)
  - `NLPConcatHeadsProgramFactory` (`device/nlp_concat_heads_program_factory.cpp`) — a **single** `create_descriptor` factory with an internal `if (in_sharded)` branch that selects one of two code paths (**interleaved** vs. **sharded**). Both paths are the same factory / same readiness-sheet row; findings that differ between them are labelled per-config below.

**Kernels referenced** (all in scope; followed by kernel reference, not directory):
- `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` — interleaved reader.
- `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` — sharded path; instantiated **twice** (reader-config + writer-config) over the same core range.
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — **cross-family donor** (eltwise/unary), interleaved writer.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `NLPConcatHeadsDeviceOperation` → `NLPConcatHeadsProgramFactory` (interleaved + sharded branches) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all three kernels are structurally Device 2.0 |
| *Prereqs* — Cross-op escapes | Ok — no function-call escapes; one broadly-shared donor writer (file-path instantiation) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — single fixed `Tensor` input; CTAs read at constexpr offsets |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean bases) |
| *Port work* — Tensor bindings (per binding) | interleaved: input **Case 1**, output **Case 1**; sharded: input **clean** (borrowed-DFB), output **clean** (borrowed-DFB) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no accessor passes a 3rd arg) |
| *Port work* — CB endpoints | interleaved `cb_src0`: legal 1:1; sharded `cb_src0` / `cb_out0`: **1P+1C** (dual-instance work-split) |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below.

## Result

**GREEN → brief issued.** Every gate clears:

- **Device 2.0** — all three kernels use Device-2.0 idioms (`Noc`, `CircularBuffer`/`DataflowBuffer` wrappers, `TensorAccessor`, `UnicastEndpoint`); only sanctioned CB-index free functions (`get_tile_size`, `get_local_cb_interface`) appear.
- **Feature compatibility** — no GlobalCircularBuffer, no non-zero `address_offset`, no GlobalSemaphore, no CTA varargs.
- **TTNN factory concept** — readiness sheet `Is able to port? == yes` on the sole `descriptor`-concept factory; code cross-check agrees on every checkable column.
- **Offset base pointers** — no address RTA folds a host-side offset into a base.
- **TensorAccessor 3rd argument** — no accessor passes a 3rd argument.

No subset scoping needed: both the interleaved and sharded code paths are GREEN.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness-sheet row (line 324): `Concept = descriptor`, `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Runtime-args update (PD override) = ` (blank), `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, **`Is able to port? = yes`**, `TensorParameter relaxation = none`, `Op-owned tensors? = ` (blank — consistent with `descriptor`). Cross-check against code:
  - `Concept = descriptor` ✓ — `NLPConcatHeadsProgramFactory::create_descriptor()` returns `ProgramDescriptor` (`nlp_concat_heads_program_factory.hpp:15`, `.cpp:19`).
  - `Custom hash = no` ✓ — no `compute_program_hash` override anywhere in the op.
  - `Runtime-args update = no` ✓ — no `get_dynamic_runtime_args` / `override_runtime_arguments`.
  - `Pybind descriptor = no` ✓ — `nlp_concat_heads_nanobind.cpp:17` binds a plain function, no `create_descriptor` / `nb::class_` of the device op.
  - Cross-column invariants hold (`Op-owned tensors` blank on a `descriptor` row; no `Runtime-args update` on a legacy concept). No conflict → the sheet is trusted for this op.

- **Device 2.0 (every kernel used):** **GREEN.** No violations. Per kernel:
  - `reader_tm_tile_layout_nlp_concat_heads.cpp` — `Noc`, `CircularBuffer cb_in0`, `TensorAccessor`, `noc.async_read`/`async_read_barrier`, `cb_in0.get_write_ptr()`, `reserve_back`/`push_back`. Uses `get_tile_size(cb_id_in0)` (line 30) — **sanctioned** free function.
  - `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` — `Noc`, `CircularBuffer` ×2, `UnicastEndpoint`, `noc.async_read` (endpoint form), `noc.get_noc_id()`, `my_x[]`/`my_y[]` self-coordinate arrays (standard, not a data-movement idiom), `get_tile_size` (line 30, sanctioned), `get_read_ptr()`/`get_write_ptr()` **method** form.
  - `writer_unary_interleaved_start_id.cpp` (donor) — `Noc`, `DataflowBuffer dfb`, `TensorAccessor`, `noc.async_write`, `dfb.wait_front`/`pop_front`, `get_local_cb_interface(cb_id_out).fifo_page_size` (line 19) — **sanctioned** free function.

  No `InterleavedAddrGen`/`ShardedAddrGen`/raw `noc_async_read`/raw sem addresses; no CB-index free-function holdovers (`get_read_ptr(cb_id)` etc. — all method-form). Nothing to route to the Device 2.0 team.

- **Feature compatibility:** every Appendix A entry, in order. All **N/A** — a clean scan.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `global_circular_buffer` field, no `remote_index`/remote-CB idiom, no `CreateGlobalCircularBuffer`. Sharded CBs use plain `.buffer = <borrowed buffer>` (borrowed-memory pattern — a mechanical porting-recipe translation, not GCB). |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any `CBDescriptor` (`nlp_concat_heads_program_factory.cpp:142`, `155`). Kernel-side offsets (`start_read_offset_bytes` etc.) are added to CB `get_read_ptr()`/`get_write_ptr()`, not to a `CBDescriptor.address_offset`. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type / `CreateGlobalSemaphore`. The op declares no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single, fixed). All kernels read CTAs at **constexpr** offsets (`get_compile_time_arg_val(0..5)`), no runtime-varying CTA index. |

- **CB endpoints (GATE-free):** classified per `(CB, config)`, per node — see [Port-work summary](#port-work-summary-mirrors-the-brief).
- **Offset base pointers:** **GREEN.** Not in the offset-base-pointer triage tables (`2026-07-19_offset_base_pointers.md`; note: the sibling op `nlp_create_qkv_heads` **is** a Type-1 entry there, but this op is not). Every address RTA resolved to a clean base:
  - *Interleaved:* reader RTA slot 0 = `in0_buffer` (a `Buffer*`; `nlp_concat_heads_program_factory.cpp:201`) → clean base; writer RTA slot 0 = `out_buffer` (a `Buffer*`; `:210`) → clean base. No `->address() + <offset>` fold.
  - *Sharded:* the only offset-valued RTAs (`start_read_offset_bytes`, `start_write_offset_bytes`; `:185–186`) are **scalar byte offsets already split out from the base** — added on-device to the borrowed-CB `get_read_ptr()`/`get_write_ptr()` (`sharded.cpp:43–44`). This is the *clean* base+separate-offset shape, not a folded device pointer. No Type 1/2/3/4.
- **TensorAccessor 3rd argument:** **GREEN.** Not in the 3rd-arg triage table (`2026-07-06_tensor_accessor_3rd_arg_triage.md`), and no accessor passes a 3rd arg: interleaved reader `TensorAccessor(in0_args, in0_tensor_addr)` (`reader...nlp_concat_heads.cpp:31`) and donor writer `TensorAccessor(dst_args, dst_addr)` (`writer_unary_interleaved_start_id.cpp:31`) are both 2-arg; the sharded path constructs no `TensorAccessor`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per config):
  - **input tensor** — *interleaved:* **Case 1** (`Buffer*` RTA → `in0_tensor_addr` → `TensorAccessor(in0_args, in0_tensor_addr)`, `reader...nlp_concat_heads.cpp:31`). *sharded:* **clean** — borrowed-memory DFB (`CBDescriptor.buffer = in0_buffer`, `program_factory.cpp:150`; read via `cb_in0.get_read_ptr()`, `sharded.cpp:43`) → port via `DataflowBufferSpec::borrowed_from`.
  - **output tensor** — *interleaved:* **Case 1** (`Buffer*` RTA → `dst_addr` → `TensorAccessor(dst_args, dst_addr)`, `writer_unary_interleaved_start_id.cpp:31`). *sharded:* **clean** — borrowed-memory DFB (`CBDescriptor.buffer = out_buffer`, `program_factory.cpp:163`; written via `cb_out0.get_write_ptr()`, `sharded.cpp:44`) → `borrowed_from`.
  - Both bindings are the per-config split the recipe anticipates (same `TensorParameter`, clean in the sharded config, Case 1 in the interleaved config). Both `Buffer*`-form RTAs are the framework's cache-hit-safe interim binding — routine port work, **not** a stale-pointer hazard.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (per `(CB, config)`, per node):
  - *Interleaved* `cb_src0` (index 0): reader FIFO-produces (`reserve_back`/`push_back`), donor writer FIFO-consumes (its `cb_id_out` CTA = `src0_cb_index = 0`; `wait_front`/`pop_front`). **Plain 1:1 — legal, no action.** (Output is interleaved here, so CB 16 is not allocated.)
  - *Sharded* `cb_src0` (index 0, borrowed from `in0_buffer`): 2 touchers — the reader-config and writer-config instances of the *same* sharded kernel, each raw-reading a disjoint head range via `cb_in0.get_read_ptr() + start_read_offset_bytes`. **Dual-instance work-split → assign 1P+1C** (bind one instance PRODUCER, the other CONSUMER; cosmetic on Gen1).
  - *Sharded* `cb_out0` (index 16, borrowed from `out_buffer`): 2 touchers — same two instances, each raw-writing a disjoint range via `cb_out0.get_write_ptr() + start_write_offset_bytes`. **Assign 1P+1C.**
  - See Recipe notes re: the redundant `reserve_back` in the sharded kernel and why the disposition is 1P+1C rather than the multi-binding flag.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none forced. The sharded path is a **dual-instance work-split** (face (c)) — the dominant two-toucher shape that resolves to 1P+1C. No hidden second writer (the co-fills are visible and sync-free; no coordinating semaphore), no ≥3-toucher CB, no FIFO-role doubling.
- **Cross-op / shared kernels:** the interleaved writer is file-path-instantiated from `eltwise/unary` (`writer_unary_interleaved_start_id.cpp`) — a **broadly-shared** donor (≈44 ops instantiate it). Its Metal 2.0 rewrite is a single shared change; this op is one member of a large port-together set. (It is already Device 2.0, so no Device-2.0 gate from it.)
- **RTA varargs:** none. Every kernel reads its RTAs as a fixed set of distinct fields at constant indices (`get_arg_val<uint32_t>(0..3)`) — the preferred nameable case, ordinary port work.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** `✓ clean` — no function-call escapes. Every `#include` in the op's own kernels resolves to `api/*` (tt_metal / HAL / firmware; donor class 1 — no concern). The only out-of-directory coupling is one file-path kernel instantiation.
  - **Summary table:**

    | Op kernel (instantiation) | Donor file | Class | Status |
    |---|---|---|---|
    | interleaved writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | cross-family donor (file-path) | ✓ (Device 2.0; broadly shared) |

  - **Per-call detail:** none — no function-call escapes to classify by signature shape.
  - **Borrowed kernel files (file-path instantiation):**
    - Path: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
    - Owning pool: `eltwise/unary` (cross-family).
    - Broadly shared: **yes** — ≈44 operation `.cpp` files under `ttnn/cpp/ttnn/operations` reference this kernel. Its Metal 2.0 rewrite must land as one change adopted by all co-borrowers → treat as a **port-together set** for planning; do not migrate this kernel in isolation.
- **Relaxation candidates (mined from a custom hash):** none — the op has no custom hash.
- **TTNN factory analysis:** sheet-derived facts, cross-checked against code — `Concept = descriptor`; no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`. Target concept: `MetalV2FactoryConcept` (no op-owned tensors). All gate-conjuncts confirmed absent.

## Misc anomalies  *(team-only, non-gating)*

- **Redundant / dead FIFO ops in the sharded kernel.** `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp:35` calls `cb_in0.reserve_back(block_size)` with an inline `// Redundant` comment, and line 36 `cb_out0.reserve_back(block_size)` is never paired with a `push_back` (`:62` `// cb_out0.push_back(block_size);` is commented out). The actual data movement is pure raw-pointer addressing; these `reserve_back` calls do no useful work on the borrowed CBs. Candidate cleanup for the ops team (not porter work).
- **Conditional `cb_out0` allocation vs. unconditional kernel use (sharded path).** The sharded kernel unconditionally binds and writes `cb_out0` (index 16; `sharded.cpp:33,44`), but the factory allocates CB 16 only under `if (out_sharded)` (`program_factory.cpp:153`). `validate_on_program_cache_miss` permits a sharded input with a non-HEIGHT_SHARDED output, which *could* be INTERLEAVED (→ `out_sharded == false`). In practice a borrowed-memory output CB requires L1-sharded output, so sharded-in effectively implies sharded-out here — but the coupling is implicit, not asserted. Worth an ops-team confirmation (see Questions).
- **Stale comments in the factory.** `program_factory.cpp:36` carries a hardcoded `// 142` example (Falcon-specific); `:73` reads `Grayskull Device Setup` (Grayskull is deprecated / not a Gen1 target). Cosmetic.

## Questions for the user  *(for the ops team)*

1. **Sharded-in / interleaved-out reachability:** Is the `in_sharded == true && out_sharded == false` configuration actually reachable? The sharded kernel references `cb_out0` (index 16) unconditionally, but the factory allocates it only when the output is sharded (`nlp_concat_heads_program_factory.cpp:153`). If reachable, this is a pre-existing latent issue independent of the port; if not, an assertion would make the coupling explicit. (Does not block the port — CB endpoints is gate-free — but the porter needs the borrowed-output-CB assumption confirmed.)

## Recipe notes

- **`reserve_back`-without-`push_back` vs. the CB-endpoints locked-role rule.** The CB endpoints classification table in `metal2_audit.md` states "a kernel that FIFO-produces (`reserve_back`/`push_back`) is a **locked producer**." Read strictly, the sharded path's two same-source instances each call `cb_in0.reserve_back` / `cb_out0.reserve_back`, which would make **two locked producers** per node → the multi-binding flag. But the "last resort" guidance clarifies the FIFO-doubling trigger as "*two real `push_back`ers, or two real `pop_front`ers*," and face (c) explicitly names the concat-family sharded dual-instance work-split as an **expected 1P+1C** case. Here there are **zero** `push_back`/`pop_front` commits — the `reserve_back` calls are redundant no-ops (one is literally commented `// Redundant`) and the real movement is sync-free raw addressing. I resolved this to **1P+1C**, treating a `reserve_back` that never advances the cursor (no `push_back`) as effectively role-free, consistent with the `evil_set_write_ptr` distinction (only a *cursor-mutating* raw call locks a role). Flagging because the strict table wording and the clarifying guidance point in opposite directions for `reserve_back`-only kernels; a one-line note in the table ("a `reserve_back` with no matching `push_back` does not lock the producer role") would remove the ambiguity for the next auditor.
