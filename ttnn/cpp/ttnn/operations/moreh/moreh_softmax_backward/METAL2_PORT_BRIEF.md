# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `bace43c8fb5 2026-08-12 docs(metal_2.0): stop the port from deleting the op's custom program hash` *(carry this line into the port report's Provenance section)*

**Scope:** one `DeviceOperation`, five factories, 13 referenced kernels:

| Factory | Factory file | Reader | Writer | Compute |
|---|---|---|---|---|
| `MorehSoftmaxBackwardWSmallFactory` | `device/softmax_backward_w_small/softmax_backward_w_small.cpp` | `reader_moreh_softmax_backward_w.cpp` | `writer_moreh_softmax_w.cpp` | `moreh_softmax_backward_w.cpp` |
| `MorehSoftmaxBackwardWLargeFactory` | `device/softmax_backward_w_large/softmax_backward_w_large.cpp` | `reader_moreh_softmax_backward_w_large.cpp` | `writer_moreh_softmax_w.cpp` | `moreh_softmax_backward_w_large.cpp` |
| `MorehSoftmaxBackwardHSmallFactory` | `device/softmax_backward_h_small/softmax_backward_h_small.cpp` | `reader_moreh_softmax_backward_h.cpp` | `writer_moreh_softmax_h.cpp` | `moreh_softmax_backward_h.cpp` |
| `MorehSoftmaxBackwardHLargeFactory` | `device/softmax_backward_h_large/softmax_backward_h_large.cpp` | `reader_moreh_softmax_backward_h_large.cpp` | `writer_moreh_softmax_h.cpp` | `moreh_softmax_backward_h_large.cpp` |
| `MorehSoftmaxBackwardCLargeFactory` | `device/softmax_backward_c_large/softmax_backward_c_large.cpp` | `reader_moreh_softmax_backward_c.cpp` | `writer_moreh_softmax_backward_c.cpp` | `moreh_softmax_backward_c_large.cpp` |

Note the two writers each serve **two** factories (`writer_moreh_softmax_w.cpp` → W pair, `writer_moreh_softmax_h.cpp` → H pair) — see *Watch for*.

**Two kernel files in `device/kernels/` are bound by nothing** and were not audited: `writer_moreh_softmax_backward_h.cpp`, `writer_moreh_softmax_backward_w.cpp`. They are near-name-twins of the writers you *do* bind — don't let one in by mistake, and don't port them.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — five `static ProgramDescriptor create_descriptor(...)` factories, declared via the `DEFINE_SOFTMAX_BACKWARD_FACTORY` macro at `device/moreh_softmax_backward_device_operation.hpp:50-63`.
- **Op-owned tensors:** none. Output allocation is ordinary TTNN (`create_output_tensors` → `create_device_tensor`, or the caller's preallocated `input_grad_tensor`, `device/moreh_softmax_backward_device_operation.cpp:108-117`).
- **Target concept:** `ProgramSpecFactoryConcept` (plain, no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus **other migration-risky pybind**, which surfaces as a `safe` warning that also fails the gate. All `no` on this op. The nanobind file exposes only the three public entry points and two value enums (`moreh_softmax_backward_nanobind.cpp:18-63`); no factory or descriptor internals.

## Construct — to do

**Tensor bindings** (per binding) — three, all Case 1, and **identical across all five factories**:

- `output` (`y`) — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::…)`. Today: base delivered as a `Buffer*` at reader RTA index 0, consumed only through `TensorAccessor(y_args, y_addr)`; the accessor args ride a CTA block built by `TensorAccessorArgs(*output.buffer()).append_to(reader_ct_args)`. Both the RTA slot and that CTA block disappear.
- `output_grad` (`dy`) — **Case 1**, same shape: reader RTA index 1, second `TensorAccessorArgs` block, consumed via `TensorAccessor(dy_args, dy_addr)`.
- `input_grad` (`dx`, the output) — **Case 1**: writer RTA index 0, sole `TensorAccessorArgs` block, consumed via `TensorAccessor(out_args, dst_addr)`.

No Case 2 anywhere — no kernel does raw address arithmetic on a base, so no `get_bank_base_address` bridge is needed. The bases arrive today as `Buffer*` entries in `emplace_runtime_args` (not `->address()` values), so the framework already patches them on cache hits: this is routine conversion, not a stale-pointer repair.

**TensorParameter relaxation:** none. The op has no custom hash, so there is nothing to relax and nothing to reconcile.

**TensorAccessor 3rd arg:** none — all 15 `TensorAccessor(` sites are 2-arg. Nothing to drop, and no `dynamic_tensor_shape` follows.

**CB endpoints:** every reader/writer↔compute CB is already a plain 1P+1C and needs no action. Self-loop the compute-local intermediates (bind the compute kernel PRODUCER **and** CONSUMER — one toucher, legal on Gen1):

| Factory | Self-loop | Plain 1:1 (no action) |
|---|---|---|
| `WSmall` | `c_24`, `c_25`, `c_26` | `c_0`, `c_1`, `c_2`, `c_3`, `c_16` |
| `HSmall` | `c_24`, `c_25`, `c_26` | `c_0`, `c_1`, `c_2`, `c_3`, `c_16` |
| `WLarge` | `c_24`, `c_25`, `c_26`, `c_27` | `c_0`, `c_1`, `c_2`, `c_3`, `c_16` |
| `HLarge` | `c_24`, `c_25`, `c_26`, `c_27` | `c_0`, `c_1`, `c_2`, `c_3`, `c_16` |
| `CLarge` | `c_24`, `c_25`, `c_26` | `c_0`, `c_1`, `c_16` *(no scaler/mask CB in this factory)* |

No 1P+1C assignment decisions, no multi-binding advanced option, no dead CB to drop. **The dispositions do not flip with config** — verified across `LOG` vs `SOFTMAX`/`SOFTMIN`, `fp32_dest_acc_en`, and `has_core_group_2`; under `LOG` the intermediates are re-aliased under second names (`cb_exp`, `cb_inter0/1/2` for the same `c_24`/`c_25`/`c_26`), same single toucher.

Two consumers hold their tile and never `pop_front` — `c_2` (scaler, waited once inside `compute_kernel_lib::reduce`) and `c_3` (mask, every call site passes `popm=0`). That is a legal 1:1 FIFO; don't read the missing pop as a missing endpoint.

**Per-core-group compute pair — keep both specs.** Each factory emits the compute kernel **twice** from one source (`compute_desc_1` over `core_group_1`, `compute_desc_2` over `core_group_2`, the second guarded by `has_core_group_2`), differing only in the leading CTA (`{num_tiles_per_core_group_1, Wt}` vs `{num_tiles_per_core_group_2, Wt}`). Port it as **two `KernelSpec`s of the same source in two `WorkUnitSpec`s** over disjoint `target_nodes`, both binding the same DFBs. Collapsing them into one spec by demoting that CTA to an RTA is the *Demoting per-group CTA to RTA* anti-pattern (`port_patterns.md`) — it costs real kernel perf and its premise is false. The node sets are disjoint, so each node still sees exactly one compute instance: ordinary single-role bindings, **not** the multi-binding flag.

**Donor call sites need no bridging work.** The kernel-lib and shared-pool helpers this op calls take either a `DataflowBuffer` **by value** (`generate_bcast_scaler`, `generate_mask_h/w`, and the whole `*_to_cb` compute family) or a `uint32_t` CB id as an **NTTP** (`dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<…>`, `compute_kernel_lib::reduce<…>`). A `dfb::name` token satisfies both directly — the implicit `DataflowBuffer(DFBBindingToken)` constructor and the `constexpr operator uint32_t()` respectively (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:106`, `:89`). Pass the handles straight through; no `.id` extraction, no temp DFB wrappers at the call sites.

**Sanctioned free function you may retire on the object:** the kernels look up tile sizes with `get_tile_size(cb_id)` (e.g. `reader_moreh_softmax_backward_c.cpp:33-34`, `writer_moreh_softmax_w.cpp:26`). `DataflowBuffer::get_tile_size()` exists as a member (`dataflow_buffer.h:201`), so moving these onto the object is ordinary port work under kernel-side whitelist rule 7. This was *not* a Device 2.0 finding — the free function is sanctioned at that stage — so treat it as recipe-normal cleanup, not a gate you inherited.

## Watch for

- **CB endpoints (multi-binding):** none. Both faces were hunted and came back empty — no hidden second writer (every raw `get_write_ptr()` in play sits inside a donor helper bracketed by `reserve_back`/`push_back` on the *same* kernel, and the op has no semaphores at all to coordinate a co-fill), and no multi-reader CB (no Buffer-backed / borrowed-memory CB exists anywhere in the op). You do not need to re-run the hunt.
- **Cross-op / shared kernels:** **no cross-op sharing — but two intra-op multi-binders.** `writer_moreh_softmax_w.cpp` is bound by `WSmall` (`softmax_backward_w_small.cpp:156`) **and** `WLarge` (`softmax_backward_w_large.cpp:187`); `writer_moreh_softmax_h.cpp` by `HSmall` (`softmax_backward_h_small.cpp:156`) **and** `HLarge` (`softmax_backward_h_large.cpp:187`). This is the **intra-op** shape of *Caution: Porting a shared kernel*. Which rung applies depends on your assignment, so settle it before touching either file:
  - Porting **all five factories in one change** → every binder converts together, so **rung 3 (convert in place)** is legitimately available; confirm the assigned set covers both binders first.
  - Porting **one factory at a time** → **rung 2: fork.** `<stem>_metal2.cpp` beside the original in this op's own `device/kernels/`, original untouched apart from the pointer comment, and the sibling factory keeps binding the legacy copy until it ports.

  No `_metal2` fork exists beside any of this op's kernels — if you fork, yours is the first. Name its bindings for the kernel's own role vocabulary, not for whichever factory you port first.

  **Filename trap — don't let a grep mislead you.** `writer_moreh_softmax_h.cpp` and `writer_moreh_softmax_w.cpp` also exist as **separate private copies** under `moreh_softmax/device/kernels/` (the *forward* op), and those copies are what `moreh_softmax`'s factories and `normalization/softmax`'s general factories bind — the latter via `SOFTMAX_KERNEL_PATH_GENERAL`, which resolves to the forward op's directory (`softmax_operation_types.hpp:39-40`). A filename grep shows four hits and reads like a cross-op sunset list; checking the bound **path** shows those ops never touch this directory. The consumer set for both files is exactly the two same-op factories named above — **a sunset and coordination list, not authorization to convert in place.**
- **RTA varargs:** none — prefer named RTAs throughout. Every arg is read at a fixed literal index (`get_arg_val<uint32_t>(0)` … `(7)`); no counted loop, no running `arg_index++`, no data-selected read. Names are legible from the kernel locals: readers `y_addr`, `dy_addr`, then `N`/`num_tiles`, `tile_offset`, `Ht`/`Wt` (or `outer_stride`, `inner_size`, `dim_size` for `CLarge`), `scaler`, `mask_h`/`mask_w`; writers `dst_addr`, `N`/`num_tiles`, `tile_offset`, `Ht`/`Wt` (or the C-dim triple).
- **One dead RTA — drop it, don't name it.** The W factories push `std::bit_cast<uint32_t>(scaler)` at reader RTA **index 5** (`softmax_backward_w_small.cpp:240`, `softmax_backward_w_large.cpp:256`), but `reader_moreh_softmax_backward_w.cpp:19` and `reader_moreh_softmax_backward_w_large.cpp:19` read `mask_w` from **index 6** and never read index 5 — the W readers get their scaler on-device from `calculate_and_prepare_reduce_scaler`. Don't invent a name for an arg no kernel reads. (The H readers' `scaler` at index 6 **is** live — `reader_moreh_softmax_backward_h.cpp:19` feeds it to `generate_bcast_scaler` — so keep that one and name it.)
- **`experimental/quasar/` has no copy of this op or its kernels.** Nothing there to mistake for prior art or for a fork to reuse — the tree is out of bounds either way.
