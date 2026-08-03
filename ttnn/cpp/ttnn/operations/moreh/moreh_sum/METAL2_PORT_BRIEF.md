# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_sum`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `20c1692eb08 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

**Scope:** one DeviceOperation, `MorehSumOperation`, with **six** factories — `MorehSumHFactory`, `MorehSumWFactory`, `MorehSumNCFactory`, `MorehSumHIntFactory`, `MorehSumWIntFactory`, `MorehSumNCIntFactory` — and **16** kernel files under `device/moreh_sum_{h,w,nc}_impl_kernels/`. All 16 are live; none is borrowed from another op, and no other op instantiates them. Factory choice is dtype × reduced-dim (`moreh_sum_device_operation.cpp:17-39`).

Note `reader_moreh_sum_nc.cpp` and `writer_moreh_sum_nc.cpp` are each shared by **two** factories (`MorehSumNCFactory` and `MorehSumNCIntFactory`) — one kernel port serves both, but check both factories' DFB sets when you bind them (they differ: the Int factory allocates no `c_1`).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — six `static ProgramDescriptor create_descriptor(...)` at `device/moreh_sum_device_operation.hpp:34,41,48,55,62,69`
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept` (plain). Identical for all six factories — one wiring pattern covers the op.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no`, confirmed both on the readiness sheet and independently in code.

## Construct — to do

### Tensor bindings

Two per factory, **identical across all six**, both **Case 1**:

- **`input`** — Case 1 (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::input)`.
- **`output`** — Case 1 → `TensorAccessor(tensor::output)`.

Two things disappear together at each site: the host-side `TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args)` CTA plumbing, and the address argument in `emplace_runtime_args`. The bases arrive today as **`Buffer*` entries** in the RTA list (`auto* const input_buf = input.buffer();` → `emplace_runtime_args(core, {input_buf, …})`), not as `->address()` — the framework's `BufferBinding` form, superseded by the typed binding. RTA sites: `moreh_sum_h_program_factory.cpp:236-250` · `moreh_sum_w_program_factory.cpp:246-259` · `moreh_sum_nc_program_factory.cpp:203-213` · `moreh_int_sum_h_program_factory.cpp:224-238` · `moreh_int_sum_w_program_factory.cpp:229-242` · `moreh_int_sum_nc_program_factory.cpp:189-199`.

**No Case 2 anywhere — do not reach for the `get_bank_base_address` bridge.** The two int32 writers *do* perform raw pointer arithmetic (`writer_moreh_int_sum_h.cpp:32-40`, `writer_moreh_int_sum_w.cpp:31-39`), but on **CB memory** via `dfb_out_obj.get_read_ptr()`, never on tensor memory. Those loops are untouched by the port; the tensor write beside them still goes through the accessor.

**TensorParameter relaxation:** none — the op has no custom hash, so none can be pending.

**TensorAccessor 3rd arg:** none — all 10 accessor sites are already 2-arg. Nothing to drop.

### CB endpoints

No multi-binding anywhere: no CB on any node has ≥3 touchers or two kernels locked to the same FIFO role. Compute is instantiated twice per factory over **disjoint** core groups, so each node sees one compute instance — this is *not* the dual-instance work-split shape, and no 1P+1C assignment question arises from it.

**Self-loop** (single toucher — the compute kernel both fills and drains; bind it PRODUCER *and* CONSUMER):

| Factory | CB | Config |
|---|---|---|
| `MorehSumHFactory` | `c_24` accum | all |
| | `c_25` masked_input | `do_mask_h` only |
| | `c_3` mask_h | `!do_mask_h` |
| `MorehSumWFactory` | `c_24` accum, `c_25` masked_input | all |
| | `c_3` mask_w | `!do_mask_w` |
| `MorehSumHIntFactory` | `c_24` intermed0 | all |
| | `c_1` mask_h | `!do_mask_h` |
| `MorehSumWIntFactory` | `c_24` intermed0 | all |
| | `c_1` mask_w | `!do_mask_w` |
| `MorehSumNCIntFactory` | `c_24` intermed0 | all |

Everything else is a plain 1P+1C FIFO (reader→compute on `c_0`/`c_1`/`c_2`/`c_3`, compute→writer on `c_16`) — no action.

**Dead-CB drop — confirmed, `MorehSumNCFactory` only:**

> `c_24` @ `moreh_sum_nc_program_factory.cpp:95-103` is allocated but **no kernel in that factory references it in any config.** Drop the allocation, and with it the now-unused locals `intermed0_t` (line 60), `intermed_data_format` (line 70), `intermed_tile_size` (line 72). A dead CB has no behavior, so removing it changes none; and a bindingless DFB cannot be expressed in Metal 2.0 at all, so this is not optional. Record the drop with `file:line` in the port report.
>
> Verified across all three of that factory's kernels (`reader_moreh_sum_nc.cpp`, `writer_moreh_sum_nc.cpp`, `moreh_sum_nc.cpp`), their CTAs, their RTAs, and their shared headers. Cause: `moreh_sum_nc.cpp` accumulates in DST (`add_tiles(..., acc_to_dest = true)`), so it never needed an L1 intermediate — the Int sibling does, and the float factory carried the allocation along.
>
> **Do not generalize this to `c_24` elsewhere** — it is live (self-loop) in all five other factories.

**Decision needed before you bind it — `c_25` in `MorehSumHFactory` under `!do_mask_h`:**

> `moreh_sum_h_program_factory.cpp:120-128` allocates `c_25` unconditionally, but `moreh_sum_h.cpp:54` guards every *access* behind `if constexpr (do_mask_h)` — so when `origin_H % 32 == 0` the CB has **zero endpoints** and cannot be expressed as a DFB. This is raised as Question 1 in the audit; do not resolve it yourself.
>
> Note the kernel also constructs the object **unconditionally** at `moreh_sum_h.cpp:23`, outside the guard — so dropping the DFB from the spec without also guarding line 23 will fail to compile (the `dfb::` token would not exist).
>
> **Its W-float twin behaves differently:** `moreh_sum_w.cpp:71` uses a plain `if (do_mask_w)`, so its `c_25` references survive compilation and it is a live self-loop in every config. Same logical structure, opposite disposition — do not assume the two mirror each other.

## Watch for

- **Runtime-selected DFB handles — the sharpest thing in this op.** Three kernels choose *which* CB to act on at runtime rather than binding a fixed handle. `dfb::name` tokens are static, so these do not translate one-for-one; you will need a branch on two bound tokens rather than a mutable index. Flag it if the shape resists a clean rewrite rather than inventing one:
  - `moreh_sum_w.cpp:15,46,94` — `cb_input` is a **mutable** variable reassigned from `c_0` to `cb_masked_input` (`c_25`) mid-loop, then used through temporaries `DataflowBuffer(cb_input)` at lines 51, 58, 73, 93, 98, 124. The same expression denotes two different DFBs at different points.
  - `moreh_int_sum_nc.cpp:39-40` — `uint32_t cb_out = last_out ? cb_out0 : cb_intermed0;` then `pack_tile_from_dst(DataflowBuffer(cb_out), dst0)`; one call site produces into either `c_16` or `c_24`.
  - `moreh_int_sum_h.cpp:14` — `auto cb_in0 = tt::CBIndex::c_0;` declared non-`constexpr` unlike its siblings. Never reassigned, so cosmetic — but it reads as mutable.
- **`if constexpr` vs plain `if` guards diverge between the float H and W compute kernels** — `moreh_sum_h.cpp:54` uses `if constexpr (do_mask_h)`, `moreh_sum_w.cpp:71` a plain `if (do_mask_w)`. This is what makes their `c_25` dispositions differ. Read each kernel's guards rather than porting one from the other's shape.
- **Cross-op / shared kernels:** no borrowed kernel `.cpp` files — the op owns all 16, and no other op instantiates them. **No `_metal2` fork question and no sunset list.** Coupling is header-only (function-call escape) into `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp`, `kernel/dataflow/generate_mm_scaler.hpp`, and `kernel_lib/{l1_helpers,reduce_helpers_dataflow,reduce_helpers_compute}.hpp`.

  **Every donor call shape bridges natively — this port should require zero donor-side edits.** The donors take either `DataflowBuffer` by value (which a `dfb::name` token converts to implicitly, via the non-explicit `DataflowBuffer(DFBBindingToken)` constructor at `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:72`) or a `uint32_t` CB id as an NTTP (handled by `dfb::name`'s constexpr cast). These headers have 45–64 consumers each; **do not "modernize" one in passing** — a signature change there ripples across dozens of ops.
- **RTA varargs:** none. Every RTA is nameable — name them all. `ArgFetcher` (`kernel/dataflow/moreh_common.hpp:44-53`) is a running `arg_idx++` counter over a **fixed** run of reads at the top of `reader_moreh_sum_nc.cpp` (7 args) and `writer_moreh_sum_nc.cpp` (3 args) — the recipe's explicit non-signal, not a vararg block. All other kernels use literal indices `get_arg_val<uint32_t>(0..4)`.
- **`experimental/quasar/` holds no copy of this op** — checked. If you find a `*_metal2.cpp` that looks like a solved version of a problem here, it is not from this op; do not use it as a source.
