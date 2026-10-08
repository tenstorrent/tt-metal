# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_boltz`

- **`NLPConcatHeadsBoltzDeviceOperation`** (`device/nlp_concat_heads_boltz_device_operation.hpp:17`) is the only DeviceOperation in the directory.
  - **Direct-descriptor factory.** `create_descriptor` is a static member of the device-op itself, and there is no `program_factory_t` (`device/nlp_concat_heads_boltz_device_operation.hpp:27-28`; body at `device/nlp_concat_heads_boltz_program_factory.cpp:18-219`). The framework wraps it in the `MeshDeviceOperationAdapter::DirectDescriptorFactory` shim (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170`), selected by `HasDirectDescriptor` (`ttnn/api/ttnn/operation_concepts.hpp:158`).
  - One factory body, two code paths chosen by `a.is_sharded()` (`nlp_concat_heads_boltz_program_factory.cpp:29,83`). There is no compute kernel.
    - **Interleaved path:**
      - reader `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_boltz.cpp` (`ReaderConfigDescriptor`, `:117-123`)
      - writer `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**borrowed**, `WriterConfigDescriptor`, `:125-130`)
    - **Sharded path:** one kernel, `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_boltz_sharded.cpp`, instantiated **twice** over `all_cores` (reader config + writer config, same CTAs, different RTAs; `:93-105`). This is a dual-instance work-split.

The op shuffles `[num_heads, S, S, head_dim]` (TILE) into `[1, S, S, num_heads · head_dim]` (`device/nlp_concat_heads_boltz_device_operation.cpp:76-82`). Its Python entry point is `ttnn.experimental.nlp_concat_heads_boltz` (`nlp_concat_heads_boltz_nanobind.cpp:18`). It is a Boltz variant of `experimental/transformer/nlp_concat_heads`, which is **already ported** to Metal 2.0 (#54782) and has the same sharded-kernel shape.

**Scope:** TTNN op, Gen1 (WH/BH) target. This is within the scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
*(The recipe tree isn't in this checkout (`Metal_Ports`, branch `edwinlee/PD_Metal_Ports`). The hash comes from the `Port_Recipe` checkout (branch `akertesz/op-porting-recipe`). This audit followed `/localdev/edwinlee/metal2_audit.md`, which is a symlink to that checkout's `ai/audit/metal2_audit.md`.)*

**Readiness sheet:** fetched live on 2026-10-06 via the Google Drive connector (`download_file_content`, CSV), in the main session. One row for this op.

**First audit** (no earlier `METAL2_PREPORT_AUDIT.md`). Recent history on the op: #57409 (`f5093e705ae`, 2026-09-25, PD migration) and #58858 (`8e04962ad72`, 2026-10-01, removed a non-compliant unused `get_dataformat` local from the interleaved reader).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_boltz` |
| **Overall** | **GREEN (user waiver).** Every code-side gate is clean. The only RED, the stale readiness-sheet row from PD batch #57409, was waived by the user on 2026-10-06 (see Result). |
| **DOps / Factories** | `NLPConcatHeadsBoltzDeviceOperation` → direct `create_descriptor` (no factory struct). The sheet still lists `NLPConcatHeadsBoltzProgramFactory`, which #57409 deleted. |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** All three kernels (two own, one borrowed) are on `Noc` / `CircularBuffer`-or-`DataflowBuffer` / `TensorAccessor` / `UnicastEndpoint`. The CB-index free functions used are only the sanctioned `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`. |
| *Prereqs* — Cross-op escapes | Ok. The includes are `tt_metal/hw/inc/*` only. One borrowed kernel file (the eltwise/unary writer), which already has a `_metal2` fork. |
| *Feature Support* — overall | GREEN (all N/A) |
| *Feature Support* — Variadic-CTA | Ok. All CTAs are at fixed indices. |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet says `yes (with PD step)`, but the row is **stale**: `Concept` conflicts with the code, and the factory row is a phantom. Treated as **spreadsheet-broken**, which makes this a GATE routed to the readiness-sheet owner. |
| *TTNN Readiness* — Concept (current) | Code: **`descriptor`** (direct-descriptor shape, since #57409 on 2026-09-25). Sheet: `legacy device-op`. |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No. The default hash is used. Sheet agrees (`no` / backdoor `no`). |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No. Sheet agrees. |
| *TTNN Readiness* — `override_runtime_arguments` | No: the code has none. The sheet says `n/a`, which fits its stale `legacy` concept. |
| *TTNN Readiness* — Pybind `create_descriptor` | No. The only binding is the user entry point (`nlp_concat_heads_boltz_nanobind.cpp:18`). Sheet agrees. |
| *TTNN Readiness* — Op-owned tensors | No. Sheet agrees. |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept`. The sheet's `Porting Target` column agrees. The port must first introduce a factory struct (see Heads-ups). |
| *Port work* — Offset base pointer | none. Both interleaved address args are bare `Buffer*`. The sharded byte offsets are L1 offsets into borrowed CBs, added kernel-side. |
| *Port work* — Tensor bindings (per binding) | `input`: **Case 1** (interleaved) / **clean** borrowed-DFB (sharded). `output`: **Case 1** (interleaved) / **clean** borrowed-DFB (sharded, when the output is sharded). |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none. No accessor passes a 3rd argument. |
| *Port work* — CB endpoints | interleaved CB 0: legal 1:1 · sharded CB 0 / CB 16: **1P+1C** across the two instances (recipe gap, see below) · CB 16: **conditional DFB** (dead under interleaved-in + sharded preallocated output) |

**CB endpoints** are dispositions, not gates. See Port-work summary for the per-`(CB, config)` table.

## Result

**GREEN by user waiver → brief issued** (`METAL2_PORT_BRIEF.md`). As audited, this was RED on the TTNN factory concept gate (spreadsheet-broken). On 2026-10-06 the user waived that gate ("Waive"): the blocker is only the sheet being out of date, not anything in the op. The sheet refresh is still owed to the readiness-sheet owner as housekeeping; it is no longer a port blocker. The original finding is kept below.

**Original finding (pre-waiver):** RED → blocked on the TTNN factory concept gate (spreadsheet-broken)**, routed to the **readiness-sheet owner** (Diego, `dgomez@tenstorrent.com`). **This is the only RED, and it is the known stale-row problem from PD-migration batch #57409.** Nothing in the op's code blocks the port.

The live sheet (fetched 2026-10-06) still describes the op as it was before PR #57409, *[Cleanup] Port More Ops to PD* (`f5093e705ae`, 2026-09-25). That PR deleted the `NLPConcatHeadsBoltzProgramFactory` struct and its header `device/nlp_concat_heads_boltz_program_factory.hpp` (diffstat: `nlp_concat_heads_boltz_program_factory.hpp | 37 ----`), and moved `create_descriptor` onto the device-op. The sheet still shows:

- `Concept` = `legacy device-op`
- `Factory (variant)` = `NLPConcatHeadsBoltzProgramFactory`
- `Factory definition path` = that now-deleted `.hpp`
- `Is able to port?` = `yes (with PD step)`. That PD step has since landed.

**No other gate fires.** Device 2.0, Feature compatibility, Offset base pointers and TensorAccessor 3rd argument are all clean on the code.

**This RED is cleared outside the op's code.** The sheet owner updates the row; nothing in the op changes. Per the recipe's exception, I ran all the informational subjects. The detail below should survive re-audit unchanged.

**Brief:** none was issued at first. It was issued after the user's waiver (2026-10-06), from this audit unchanged.

The op has two code paths (interleaved, sharded), but the blocker is the sheet row, not a path. Whole-op RED; there is no subset distinction to offer.

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED (spreadsheet-broken)**, routed to the readiness-sheet owner to reconcile. The sheet row is `Op` = `experimental/transformer/nlp_concat_heads_boltz`, `Device operation` = `NLPConcatHeadsBoltzDeviceOperation`, `Factory (variant)` = `NLPConcatHeadsBoltzProgramFactory`.
  - **Primary-column conflict, `Concept`.** The sheet says `legacy device-op`. The code is `descriptor`: `static ProgramDescriptor create_descriptor(...)` on the device-op (`device/nlp_concat_heads_boltz_device_operation.hpp:27-28`), with no `create()` and no `override_runtime_arguments`. Before #57409 the device-op declared `using program_factory_t = std::variant<NLPConcatHeadsBoltzProgramFactory>;` (`git show f5093e705ae^:…/nlp_concat_heads_boltz_device_operation.hpp`, line 21). That line is gone now.
  - **Phantom factory row.** `NLPConcatHeadsBoltzProgramFactory` no longer exists, and `Factory definition path` points at the deleted `.hpp`. There is also a **missing row**: the code's only factory is the device-op's direct `create_descriptor` (wrapped by `DirectDescriptorFactory`), and no row matches it.
  - `Is able to port?` = `yes (with PD step)`. This is a derived cell, so it is read, not vetted. It is moot here, because the conflicts above already make the row spreadsheet-broken.
  - **The rest of the primary cross-check is clean:**
    - `Custom hash` = `no` and backdoor = `no`. No `compute_program_hash`, `attribute_values` or `to_hash` exists in the op.
    - `Runtime-args update (get_dynamic_runtime_args)` = `no`. There is no hook.
    - `Override runtime args method?` = `n/a`. The code has none.
    - `Pybind descriptor` = `no`. There is no `create_descriptor` binding.
    - `Op-owned tensors?` = `no`.
    - `Smuggled pointer` = `no`. Both interleaved addresses use the `Buffer*`-binding form, and the sharded path uses `.buffer`-backed CBs.

    No cross-column invariant is violated. `Known op issues` is empty.
  - **Path forward:** the sheet owner refreshes the row (`Concept` = `descriptor`, factory row renamed or removed, `Is able to port?` re-derived). Re-audit after that is expected to go GREEN. Or the user waives this gate now.

- **Device 2.0 (every kernel used): GREEN.**
  - **Interleaved reader** (`reader_tm_tile_layout_nlp_concat_heads_boltz.cpp`, own):
    - `Noc noc` (`:14`)
    - `CircularBuffer cb_in0(cb_id_in0)` with `reserve_back` / `push_back` / `get_write_ptr` (`:32,40,46,58`)
    - `TensorAccessor(in0_args, in0_tensor_addr)` (`:30`)
    - `noc.async_read(s0, CoreLocalMem<uint32_t>(…), size, {.page_id=…}, {})` and `noc.async_read_barrier()` (`:48-57`)
    - CB-index free function: `get_tile_size(cb_id_in0)` (`:29`), **sanctioned**.
    - #58858 removed this kernel's one non-compliant line (an unused `get_dataformat(cb_id_in0)` local).
  - **Sharded kernel** (`reader_tm_tile_layout_nlp_concat_heads_boltz_sharded.cpp`, own; both instances):
    - `Noc noc`, `CircularBuffer cb_in0` / `cb_out0` with `reserve_back`, `get_read_ptr`, `get_write_ptr` (`:31-35,42-43`)
    - local loopback via `UnicastEndpoint src_ep` + `noc.async_read(src_ep, CoreLocalMem<uint32_t>(…), …, {.noc_x, .noc_y, .addr}, {})` (`:40,47-52`), the migration guide's own `UnicastEndpoint` read form
    - The own-core coordinates come from `my_x[noc_id]` / `my_y[noc_id]` with `noc_id = noc.get_noc_id()` (`:37-39`). That is a coordinate lookup, not a legacy addressing idiom. The same pattern is used by migrated kernels (`data_movement/concat/.../reader_s2s_tensor_concat.cpp:50`) and was kept verbatim by the Metal 2.0 port of the sibling `nlp_concat_heads` sharded kernel (`:32-34`). Not a violation.
    - CB-index free function: `get_tile_size(cb_id_in0)` (`:29`), **sanctioned**.
  - **Borrowed writer** (`eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`):
    - Already on `DataflowBuffer dfb(cb_id_out)`, `Noc`, `TensorAccessor` (`:29-30,39,48-53`).
    - CB-index free function: `get_local_cb_interface(cb_id_out).fifo_page_size` (`:27`), **sanctioned**.
  - None of the three uses raw `noc_async_*`, `*AddrGen*`, `get_noc_addr`, or raw semaphores.

- **Feature compatibility: GREEN** (no gate fired).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | There are no `GlobalCircularBuffer`, `global_circular_buffer` or `remote_*` signals. Both `CBDescriptor`s (`nlp_concat_heads_boltz_program_factory.cpp:139-148,152-161`) leave `.global_circular_buffer` unset. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` is never set (default 0). The `.buffer`-backed CBs are the ordinary borrowed-memory pattern. |
  | GlobalSemaphore | N/A | The op has no semaphores at all. |

- **CB endpoints (GATE-free):** see Port-work summary.

- **Offset base pointers: GREEN.**
  - Interleaved reader RTA 0 `in0_tensor_addr` is `in0_buffer` (`nlp_concat_heads_boltz_program_factory.cpp:197`). Interleaved writer RTA 0 `dst_addr` is `out_buffer` (`:206`). Both are bare `Buffer*` with no host arithmetic. The per-core offsets travel as separate scalars (reader `in0_tensor_tile_id`, `:192,200`; writer `start_id`, `:208`) and feed `{.page_id = …}` on the accessor.
  - Sharded RTAs 1/2 (`start_read_offset_bytes`, `start_write_offset_bytes`, `:181-182`) are byte offsets into L1. The kernel adds them to `cb_in0.get_read_ptr()` / `cb_out0.get_write_ptr()` (sharded kernel `:42-43`), so the base is the borrowed CB, not a host-folded address.
  - The op is not in the `2026-07-19_offset_base_pointers.md` tables (only the different op `nlp_create_qkv_heads_boltz` is), so this is the "no fold, not in tables" outcome: clean.

- **TensorAccessor 3rd argument: N/A.** The subject never fires: the reader accessor is 2-arg (`reader_tm_tile_layout_nlp_concat_heads_boltz.cpp:30`), and so is the writer's (`writer_unary_interleaved_start_id.cpp:39`). The sharded kernel builds no accessor. The op is not in the `2026-07-06_tensor_accessor_3rd_arg_triage.md` table either.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per path):
  - `input`, interleaved path: **Case 1**. The host passes `in0_buffer` as a `Buffer*` RTA (`nlp_concat_heads_boltz_program_factory.cpp:197`) and its accessor args as CTAs from index 4 (`:113`; kernel `TensorAccessorArgs<4>()` at reader `:26`). The reader builds `TensorAccessor(in0_args, in0_tensor_addr)` (`:30`) and reads only through it (`:48-53`).
  - `output`, interleaved path: **Case 1**. The host passes `out_buffer` as a `Buffer*` RTA (`:206`) and its accessor args as CTAs from index 1 (`:115`). The borrowed writer builds `TensorAccessor(dst_args, dst_addr)` (`writer_unary_interleaved_start_id.cpp:39`). In the port this binding goes through the fork's `tensor::dst`.
  - `input`, sharded path: **clean** (borrowed-memory DFB). CB 0 `.buffer = in0_buffer` (`:147`); the kernel reads via `cb_in0.get_read_ptr()` (sharded `:42`). Port via `DataflowBufferSpec::borrowed_from`.
  - `output`, sharded path with a sharded output: **clean** (borrowed-memory DFB). CB 16 `.buffer = out_buffer` (`:160`); the kernel writes via `cb_out0.get_write_ptr()` (sharded `:43`).
  - The `Buffer*`-binding form is correct on cache hits today (the framework patches it; the device-op comment at `nlp_concat_heads_boltz_device_operation.hpp:23-26` says the same). So this is routine work, not a hazard.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**

  | CB (index) | Config | Touchers (per node) | Disposition |
  |---|---|---|---|
  | CB 0 `src0_cb_index` (`nlp_concat_heads_boltz_program_factory.cpp:139-148`), `2 × per_tensor_tiles` tiles | interleaved input | reader: **locked producer** (`reserve_back` `:46`, `push_back` `:58`; `get_write_ptr` `:40` is a peek on its own binding). Writer: **locked consumer** (`wait_front` / `pop_front`, `writer_unary_interleaved_start_id.cpp:48,51`; `fifo_page_size` read `:27`). | **plain 1:1.** Reader PRODUCER, writer CONSUMER. No flag. |
  | CB 0, borrowed from `input` (`:147`) | sharded input | both instances of the sharded kernel: `cb_in0.reserve_back(block_size)` (`:34`, self-commented `// Redundant`; never pushed) + `get_read_ptr` peek (`:42`) | **1P+1C** across the two instances (dual-instance work-split). See the note below. |
  | CB 16 `out_cb_index`, borrowed from `output` (`:150-162`) | sharded input + sharded output | both instances: `cb_out0.reserve_back(block_size)` (`:35`; the matching `push_back` is commented out, `:61`) + raw writes via `get_write_ptr` (`:43,47-52`), disjoint per RISC by `start_write_offset_bytes` | **1P+1C** across the two instances. See the note below. |
  | CB 16 | interleaved input + **sharded preallocated output** | allocated (`out_sharded` is true, `:150`), but the interleaved kernels never touch index 16 | **dead in this config, live in the one above → conditional DFB.** Legacy allocates CB 16 iff `out_sharded`. The port must create the DFB iff `in_sharded && out_sharded`. Do **not** drop it. (Reachable because validation checks a preallocated output's logical shape only, `nlp_concat_heads_boltz_device_operation.cpp:58-64`.) |
  | CB 16 | sharded input + **interleaved output** | the kernel touches index 16 (`:32,35,43`) but no CB 16 is allocated (`:150`) | **no legal disposition: a legacy bug.** See Misc anomalies and Questions. |

  **The sharded-CB note (recipe gap).** By the recipe's census rule, each instance's `reserve_back` is a FIFO-produce op, so both instances are *locked producers* (≥2 locked producers → multi-binding row). But the multi-binding flag can't express this DFB: under `allow_instance_multi_binding` the validator still requires at least one CONSUMER instance on every node (`tt_metal/impl/metal2_host_api/program_spec.cpp`, the per-node census in the "Validate local DFB endpoint placement" block, ~`:1953-1960`). Two producers with no consumer is rejected, and a self-loop is limited to the plain SPSC case. Nothing in the census can fit. My call is **1P+1C** (one instance PRODUCER, the other CONSUMER), with the kernel left verbatim. The reasons:
  - The `reserve_back` calls are functionally dead: they reserve the whole borrowed buffer once at entry, on an empty CB, and nothing is ever pushed.
  - On Gen1 the DFB kernel API does not role-check `reserve_back` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:175` forwards straight to `reserve_back_impl`, with no producer template or assertion), so a CONSUMER-bound instance still compiles and runs identically.
  - Both instances run the **same** kernel source, so stripping the calls changes both at once. That is a kernel edit outside the port's kernel-side whitelist.

  The sibling `nlp_concat_heads` port (#54782) took the other route: it **stripped** both `reserve_back` calls and bound 1P+1C. That is raised as a question below; it is not the recipe's default.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none need the flag. I checked for a hidden second writer: the only raw writes are the two sharded instances' disjoint co-writes into CB 16 (no semaphores, no third toucher). Interleaved CB 0 has no raw write by the consumer.
- **Cross-op / shared kernels:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` is borrowed (eltwise/unary pool) and broadly shared. A **`_metal2` fork already exists** beside it: `writer_unary_interleaved_start_id_metal2.cpp`. Reuse it (rung 1); don't re-fork.
  - Fork interface: `dfb::out` (CONSUMER), `tensor::dst`, named RTAs `args::num_pages` / `args::start_id`, page size via `dfb.get_entry_size()`, `#ifdef OUT_SHARDED` / `BACKWARDS` (this op sets neither define).
  - Map the writer's RTAs 1/2 (`:207-208`) onto `num_pages` / `start_id`, bind CB 0's DFB as `dfb::out`, and bind `output` as `tensor::dst`.
  - The sibling `nlp_concat_heads` already binds the fork the same way (`nlp_concat_heads_program_factory.cpp:177-189`).
  - **Sunset list (not authorization to convert the legacy file in place):** legacy binders still on the original include `data_movement/reshape_on_device`, `data_movement/slice` (tile), `eltwise/unary_backward` (+ `gelu_bw`), `examples/example`, `experimental/matmul/attn_matmul`, `matmul` (multicore). The full list is tracked in issue #52228.
  - The two op-owned kernels are bound only by this op's factory (outside `experimental/quasar/`), so convert them in place.
- **RTA varargs:** none. Every RTA is read at a fixed index and names directly:
  - interleaved reader: `num_blocks`, `in0_h_dim`, `in0_tensor_tile_id` (`:18-20`; RTA 0 becomes the binding)
  - sharded kernel: `nheads`, `start_read_offset_bytes`, `start_write_offset_bytes` (`:16-18`)

  CTAs are also fixed-index (interleaved reader `0..3` then accessor args; sharded `0..5`, where `0/1` are CB indices that become `dfb::` tokens), so there are no CTA varargs.
- **Direct-descriptor shape: the port must introduce a factory struct.** The op declares `create_descriptor` on the device-op with no `program_factory_t` (`nlp_concat_heads_boltz_device_operation.hpp:27-28`). The `DirectDescriptorFactory` shim is keyed on the name `create_descriptor` and has no `create_program_artifacts` counterpart. Follow `ttnn_factory.md` §"3. Give a direct-descriptor op a conventional program factory":
  - nest `NLPConcatHeadsBoltzProgramFactory` (the pre-#57409 name) with `create_program_artifacts`
  - add `using program_factory_t = std::variant<...>;`
  - remove the device-op-level `create_descriptor` and its comment (`:23-28`)

  Keep the body in `device/nlp_concat_heads_boltz_program_factory.cpp` (already in `experimental/transformer/sources.cmake:25`). Record it under Handoff points. *(Check again at port time in case TTNN has added a `program_factory_t` since this audit.)*
- **Sharded `reserve_back` count narrows.** `CircularBuffer::reserve_back` takes `int32_t` (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:31`); `DataflowBuffer::reserve_back` takes `uint16_t` (`dataflow_buffer.h:175`). The sharded CTA `block_size` = `num_blocks_per_core · in0_HtWt` (`nlp_concat_heads_boltz_program_factory.cpp:90`) can exceed 65535 for large S. If the calls are kept, the CB→DFB swap silently truncates the argument there. It's dead code either way, but it's worth a line in the port report.
- **Sharded path isn't exercised by any test.** The only test, `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_boltz.py`, covers interleaved only. Its docstring says sharded inputs are rejected at `TensorSpec` construction "for every realistic S". The porter can verify the interleaved path at runtime (that test's 3-iteration loop is also the cache-hit check for the new tensor bindings: it moves the input address each iteration and asserts one cache entry). The sharded path can only be verified by compiling it.
- **Kernels are already Device 2.0.** The kernel change is a binding-layer swap, not an idiom rewrite:
  - `CircularBuffer` → `DataflowBuffer` from the `dfb::` token (interleaved reader `:32`; sharded `:31-32`)
  - drop the `api/dataflow/circular_buffer.h` include (interleaved reader `:9`; sharded `:8`)
  - `get_tile_size(cb_id)` → `dfb.get_tile_size()` (interleaved reader `:29`; sharded `:29`; whitelist rule 7)
  - accessor from the `tensor::` token; named RTAs/CTAs
  - keep `my_x[noc_id]` / `my_y[noc_id]` and the `UnicastEndpoint` loopback verbatim
- **Preserve the interleaved reader's pointer walk exactly.** It takes `get_write_ptr()` once per block (`:40`), *before* the first `reserve_back(1)`, and advances it linearly across `in0_c · in0_w_tiles` single-tile `reserve_back`/`push_back` rounds (`:43-61`). This works because CB 0 holds exactly two blocks (`2 × per_tensor_tiles`, `:135-138`), so block boundaries line up with the wrap. On Gen1 a DFB lowers to the same circular buffer; keep it as is.
- **`scripts/check_kernel_cb_ptrs.py:41`** lists the sharded kernel as a known violation by path. The lint matches only the free-function `cb_reserve_back` / `get_read_ptr(cb)` spellings, so it is inert for the method form either way. Converting in place keeps the path, so no lint edit is needed.

## Team-only

- **Out-of-directory coupling & donor shape: ✓ clean.**
  - Function-call escapes: every include resolves under `tt_metal/hw/inc/`:
    - `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/endpoints.h`, `api/dataflow/dataflow_buffer.h`
    - `api/core_local_mem.h`, `api/tensor/noc_traits.h`
    - plus `<stdint.h>` / `<array>`

    That makes them all donor class 1 (LLK/HAL), with no concern. No donor functions are called, so there is no per-call table.
  - Borrowed kernel files:

    | Kernel file | Owner | Sharing | `_metal2` fork |
    |---|---|---|---|
    | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary shared pool | broadly shared: about 10 legacy binders, about 24 already on the fork (tracked in #52228). The sheet's `Uses llama kernels?` = `yes` presumably refers to this. | **exists**, `writer_unary_interleaved_start_id_metal2.cpp` |
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:**
  - op-owned tensors: none
  - MeshWorkload need: none (plain `ProgramDescriptor`)
  - pybind `create_descriptor`: none
  - other risky pybind: none (only `bind_function` of the user API)
  - custom hash: none. The default hash covers `NLPConcatHeadsBoltzParams{output_mem_config}` plus the tensor args, and the interleaved/sharded choice derives from the memory configs.
  - `get_dynamic_runtime_args`: none
  - `override_runtime_arguments`: none
  - target concept: **`ProgramSpecFactoryConcept`**

## Misc anomalies  *(team-only, non-gating)*

- **Sharded input with an interleaved output writes to an unallocated CB.** Validation forbids only a HEIGHT_SHARDED output when the input is sharded (`nlp_concat_heads_boltz_device_operation.cpp:47-50`), so an interleaved output is accepted. The factory then allocates CB 16 only when `out_sharded` (`nlp_concat_heads_boltz_program_factory.cpp:150`), but the sharded kernel unconditionally does `cb_out0.reserve_back` and raw writes through `cb_out0.get_write_ptr()` (sharded `:32,35,43`). The CB 16 interface is unconfigured: expect a hang or stray L1 writes. The sibling `nlp_concat_heads` closed this with a `TT_FATAL` requiring a sharded output (`nlp_concat_heads_device_operation.cpp:52-56`). Route to the ops team.
- **The sharded path appears to read far past its shard.** In the sharded path a "block" is `padded_shape[-2]` = S rows (`num_blocks_per_core_group_1 = shard_h / S`, `:55`; validation `:36-40`). But the kernel reads `in0_h_tiles` tile-rows per block, and `in0_h_tiles = ashape[1] · ashape[2] / 32` = S·S/32 (`:38`, CTA `:87`; kernel loop `:45-57`). A shard holds only S/32 tile-rows per block, so this over-reads by a factor of S. It looks inherited from `nlp_concat_heads`, where `ashape[1]` is 1. It is masked because a sharded output can't be constructed for realistic S (test docstring). Route to the ops team. I did not fully trace it, so treat it as a lead.
- **Dead sharded-kernel leftovers.** The sharded kernel computes `single_tile_size_bytes` (`:29`) and never uses it, and carries a self-annotated redundant `reserve_back` (`:34`), a second unpaired one (`:35`), and a commented-out `push_back` (`:61`). CTA 5 `block_size` exists only to feed those `reserve_back`s.
- **Stale comments in the factory.** `// 142` (`:34`), `Output shape is: [B, 1, s, 4544]` (`:37`) and `Grayskull Device Setup` (`:71`) are Falcon/Grayskull leftovers that don't describe this op.
- **Preallocated output is checked by logical shape only** (`nlp_concat_heads_boltz_device_operation.cpp:58-64`). Its memory layout bypasses the `output_mem_config` checks above it, which is what makes the dead-CB-16 config reachable.

## Questions for the user

1. ~~**Waive the stale-sheet gate?**~~ **Resolved 2026-10-06: waived by the user.** The audit is now "GREEN (user waiver)" and the brief is issued.
2. **Sharded `reserve_back` calls: keep (1P+1C, kernel verbatim) or strip (sibling precedent)?** This audit recommends keeping them and binding 1P+1C, since the Gen1 DFB API doesn't role-check `reserve_back` (`dataflow_buffer.h:175`). The alternative is #54782's route: strip both functionally dead calls (sharded `:34-35`), which takes a kernel edit outside the whitelist.
3. **Sharded-in + interleaved-out config (sharded `:32,35,43` vs factory `:150`).** It is broken in legacy (unallocated CB 16). After the port, the kernel's `dfb::out0` has no binding in that config, so it fails at JIT instead of hanging. Should the ops team land the `nlp_concat_heads`-style `TT_FATAL` first, or should the port carry the config as is and record the behavior change in the port report?

## Recipe notes

- **CB endpoints: two locked producers, zero consumers, has no listed resolution.** The recipe says the subject is GATE-free because "every out-of-window case has a port-time resolution". But for a dual-instance work-split where *both* instances carry a FIFO-producer op (here a dead full-capacity `reserve_back`) and nothing consumes, the census lands on the multi-binding row, and the multi-binding flag still needs ≥1 consumer per node (`program_spec.cpp` per-node census). The "role-free" escape covers only raw peeks, not dead FIFO calls. A rule on whether a dead, never-pushed `reserve_back` locks a role (or whether, on Gen1, any FIFO call locks a role, given the kernel API doesn't enforce it) would settle this. `nlp_concat_heads` (#54782) hit the same shape and resolved it by stripping the calls.
- **Stale-sheet RED after the #57409 PD batch (repeat).** Same shape as `hang_device`, `concatenate_heads`, `create_qkv_heads_from_separate_tensors` and `topk_router_gpt`: a `Concept` conflict, a phantom factory row, and `Is able to port?` = `yes (with PD step)`. A batch sheet refresh would save a re-audit round-trip per op.
- **Provenance in a separate checkout** (repeat): the provenance `git log` prints nothing in `Metal_Ports`. The hash above comes from the sibling `Port_Recipe` checkout.
