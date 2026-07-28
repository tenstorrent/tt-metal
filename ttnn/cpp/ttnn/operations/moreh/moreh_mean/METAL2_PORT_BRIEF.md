# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_mean`

> ## ⚠ SCOPED BRIEF — the op is RED at op level
>
> This is **not** an all-GREEN audit. It is the config-scoped-subset case: the Device 2.0 gate fails on **one** factory, and this brief covers the two that are clear.
>
> **In scope — port these two factories:**
> - `MorehMeanOperation::MorehMeanWFactory` (`device/moreh_mean_w_program_factory.cpp`)
> - `MorehMeanOperation::MorehMeanHFactory` (`device/moreh_mean_h_program_factory.cpp`)
>
> **Out of scope — do not touch:**
> - `MorehMeanOperation::MorehMeanNCFactory` — **blocked on Device 2.0.** Its reader calls `fill_cb_with_value` (`reader_moreh_mean_nc.cpp:32, 36`), whose body still uses the CB-index free function `get_dataformat(cb.get_id())` at `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:100`. Fixing that is the Device 2.0 team's one-line change and is **off your whitelist**. Leave that factory on the `ProgramDescriptor` path; it ports after a re-audit.
>
> **Two open items before you start** — both explained in `METAL2_PREPORT_AUDIT.md`:
> 1. The readiness sheet could not be fetched during the audit (Google Drive connector unauthorized in a non-interactive session). Every code-checkable conjunct of the TTNN factory-concept gate was verified clean, but the sheet-owned **`Is safe to port?`** call is unread. Confirm that cell reads `yes` for these two factory rows before committing.
> 2. A second, weaker `get_dataformat(id)` site sits on the **H path** — `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl:140`, reached from `reader_moreh_mean_h.cpp:33`. The audit did **not** gate it (constexpr template id, no wrapper object in scope at the call site, kernel_lib donor class). If the Device 2.0 team rules `get_dataformat(id)` out categorically, **H drops out of scope too** and only W remains. Worth confirming before you start on H.

**Gates cleared (for the two in-scope factories):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(code cross-check only — see above)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ *(no sites)*

**Recipe docs:** `66ac84052d4 2026-07-27 docs(metal_2.0): split the runtime-args porting gate into its two sheet columns` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all three factories expose `static tt::tt_metal::ProgramDescriptor create_descriptor(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`. Note the factories are **nested structs inside `MorehMeanOperation`** (`device/moreh_mean_device_operation.hpp:34-53`), not free-standing types — the `program_factory_t` variant at `:55` names them as `MorehMeanWFactory` / `MorehMeanHFactory` / `MorehMeanNCFactory` within that scope.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All verified directly in the code; `moreh_mean_nanobind.cpp:19-30` binds only the public `ttnn::moreh_mean` free function.
- **Mixed-concept transition state to expect:** you are converting two of three factories in a `std::variant` `program_factory_t`. The third stays on `ProgramDescriptor`. `select_program_factory` (`device/moreh_mean_device_operation.cpp:34-47`) is untouched by the port, as are `validate_tensors` and `compute_output_specs`.
- **`tensor_args_t` carries an optional output** (`{const Tensor& input; const std::optional<Tensor>& output;}`). When the caller supplies one, `create_output_tensors` returns it verbatim (`device/moreh_mean_device_operation.cpp:96-99`); otherwise a fresh tensor is allocated. Either way the factory sees a concrete `tensor_return_value_t& output` and binds `output.buffer()`, so the optionality does not reach the binding layer — one `TensorParameter` for the output in both cases.

## Construct — to do

**Tensor bindings** (per binding) — **both bindings are Case 1** in both in-scope factories; there is no Case 2 and no borrowed-memory DFB anywhere in this op (no `CBDescriptor` sets `.buffer`).

Today each tensor reaches its kernel as a `Buffer*` pushed into `KernelDescriptor::RTArgList` (the framework's interim `BufferBinding` hack), and every kernel feeds that base straight into a `TensorAccessor`. For each: express it as a `TensorParameter` / `TensorBinding`, have the kernel build `TensorAccessor(tensor::name)`, and delete both the `Buffer*` RTA and the `TensorAccessorArgs(...).append_to(...)` CTA plumbing.

| Factory | Binding | Host sites to remove | Kernel site to convert |
|---|---|---|---|
| W | `input` | `moreh_mean_w_program_factory.cpp:219` (RTA arg 0), `:123` (`TensorAccessorArgs`) | `reader_moreh_mean_w.cpp:12` (`src_addr`), `:16` (`src_args`), `:34` (construction) |
| W | `output` | `moreh_mean_w_program_factory.cpp:222` (RTA arg 0), `:141` | `writer_moreh_mean_unary_interleaved_start_id.cpp:11, 19, 20` |
| H | `input` | `moreh_mean_h_program_factory.cpp:212-214` (RTA arg 0), `:119` | `reader_moreh_mean_h.cpp:12, 28, 46` |
| H | `output` | `moreh_mean_h_program_factory.cpp:216` (RTA arg 0), `:137` | `writer_moreh_mean_unary_interleaved_start_id.cpp:11, 19, 20` |

**Watch the CTA ordering when you remove `TensorAccessorArgs`.** Both readers append accessor args *in the middle* of their CTA list and then read a trailing scalar off the accessor's own offset:

- `moreh_mean_w_program_factory.cpp:122-124` appends the accessor args **first**, then pushes `packed_scaler_value`; `reader_moreh_mean_w.cpp:17` reads it back as `get_compile_time_arg_val(src_args.next_compile_time_args_offset())`.
- `moreh_mean_h_program_factory.cpp:118-120` pushes `{Ht, Wt, HtWt}`, appends accessor args, **then** pushes `origin_H`; `reader_moreh_mean_h.cpp:32` reads it as `get_compile_time_arg_val(src_args.next_compile_time_args_offset())`.

Once the accessor args disappear, those trailing scalars become ordinary named compile-time args and the `next_compile_time_args_offset()` indirection goes with them. Do not try to preserve the offset arithmetic.

Note on urgency: the `Buffer*` form is the framework's interim binding hack, patched correctly on cache hits today. This is **routine port work, not a correctness hazard** — there are no `->address()`-in-RTA smuggled pointers in this op.

**TensorParameter relaxation:** none. The op has no custom hash, so no relaxation can be active. *(The sheet's column is unread — see the banner. If it names one, stop and re-check against the hash before applying it.)*

**TensorAccessor 3rd arg:** none — every `TensorAccessor` in this op is the 2-arg form. Nothing to drop.

**CB endpoints:** no multi-binding flag, no dead-CB drop. Three things to do:

1. **Self-loop `c_24`** in both factories — the accumulator is touched only by the compute kernel (W: reserve/push at `moreh_mean_w.cpp:67, 71`, wait/pop at `:101, 122`. H: produced by `reduce<…, cb_accum_dst>` at `moreh_mean_h.cpp:54`, consumed via `Accumulate::at(cb_accum_dst, …)` at `:84, 92`). Bind the compute kernel PRODUCER **and** CONSUMER.
2. **Self-loop `c_25`** in both factories, in the mask configuration — also compute-only (W: reserve/push at `moreh_mean_w.cpp:88, 92`, then wait/pop at `:99, 120` through the reassigned `cb_input`. H: reserve/push at `:72, 76`, consumed by `reduce<…, cb_masked_input, …>` at `:81`).
3. **Decide the mask-CB binding policy — this is the one real judgement call in the port.** See below.

Everything else is a plain 1P + 1C and needs no special handling: `c_0` (reader → compute), `c_2` (reader → compute), `c_3` in the mask configuration (reader → compute), `c_16` (compute → writer). The two compute `KernelDescriptor`s per factory sit on `core_group_1` / `core_group_2`, which `split_work_to_cores_wt_core_range` derives from `tt_metal::split_work_to_cores` — **disjoint, union = `all_cores`** — so every node carries exactly one reader, one writer and one compute. Preserve that partitioning verbatim; do not merge the two groups.

### The mask CBs — `c_3` and `c_25` have zero touchers in the no-mask configuration

Both are allocated **unconditionally** by the host (`moreh_mean_w_program_factory.cpp:80-88` and `:98-106`; `moreh_mean_h_program_factory.cpp:80-88` and `:98-106`), while their *use* is gated by `do_mask_w` / `do_mask_h`. When the reduced dimension is a tile multiple, no kernel touches either CB — and a DFB with no producer and no consumer binding is rejected by the spec validator.

They are **not** dead CBs: they are genuinely live whenever `origin_W % 32 != 0` (W) or `origin_H % 32 != 0` (H). Do not drop them.

**Recommended: keep both bindings unconditional.** Declare the `DataflowBufferSpec` and both bindings in every configuration, exactly as the legacy op allocates the CBs today. This reproduces current behaviour byte-for-byte, keeps the kernel diff minimal, and leaves the same small unused L1 allocation in the no-mask config that exists on `main` right now.

**Alternative (cleaner end-state, more churn):** conditional binding per `migration_guide.md` → *Optional resources*. If you take this route, note that the doc is explicit that `if constexpr` is **not** sufficient — `dfb::<name>` only exists when the host actually binds it, and `if constexpr` in a non-template `kernel_main` still performs name lookup on the discarded branch. So you would need **all** of:

- convert the W kernel's plain `if (do_mask_w)` guards at `moreh_mean_w.cpp:42, 74, 127` to `#ifdef DO_MASK_W`;
- convert the H kernel's `if constexpr (do_mask_h)` guards at `moreh_mean_h.cpp:41, 59, 97` to `#ifdef DO_MASK_H` — the `if constexpr` does *not* protect you here;
- move the unconditional aliases and `DataflowBuffer` constructions at `moreh_mean_w.cpp:24-25, 28-29` and `moreh_mean_h.cpp:25-26, 28-29` inside those `#ifdef`s;
- move `cb_input = cb_masked_input` (`moreh_mean_w.cpp:95`) inside the `#ifdef` too;
- emit the `DO_MASK_W` / `DO_MASK_H` define to the **compute** kernel as well as the reader — the host computes the flag already (`moreh_mean_w_program_factory.cpp:43`, `moreh_mean_h_program_factory.cpp:43`) but currently only passes it to the reader (`:127-129` and `:123-125` respectively).

Pick one deliberately. Translating the CB list mechanically without choosing will produce a spec the validator rejects in the no-mask configuration.

## Watch for

- **CB endpoints (multi-binding):** none. The op has **no semaphores at all**, which structurally rules out the hidden-second-writer face, and it never instantiates one kernel source twice over the same core range, which rules out the dual-instance work-split face. Skip that hunt.

- **Runtime-selected CB index in `moreh_mean_w.cpp` — the one place a mechanical `dfb::name` substitution will not drop straight in.** Line 21 declares `auto cb_input = tt::CBIndex::c_0;` as a **mutable** variable, reassigns it to `cb_masked_input` at `:95`, and constructs `DataflowBuffer(cb_input)` inline at `:57, 63, 76, 94, 99, 120`. It is still expressible — `dfb::name` accessors implicitly convert to `uint32_t` (`migration_guide.md`), so a `uint32_t` variable can hold either identity and `DataflowBuffer(that_variable)` still constructs — but if you take the conditional-binding route above, the reassignment must sit inside the same `#ifdef` as the binding, or `dfb::cb_masked_input` will not exist at that point.

  *(The same shape appears out-of-scope in `moreh_mean_nc.cpp:48`, selected on a genuine runtime bool. Noted only so it is not a surprise when NC unblocks.)*

- **Cross-op / shared kernels: nothing to coordinate — unusually clean.** The op **owns all eight** of its kernel `.cpp` files and file-path-instantiates none from a shared pool, so there is **no port-together set**. Your rewrite of `reader_moreh_mean_*.cpp`, `writer_moreh_mean_*.cpp` and `moreh_mean_*.cpp` affects no other op.

  The function-call escapes are all workable and need no special handling:
  - `generate_mm_scaler(DataflowBuffer, uint32_t)` (`ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp:12`) — pass `DataflowBuffer(dfb::name)`.
  - `generate_mask_w<T>(DataflowBuffer, uint32_t)` / `generate_mask_h<T>(DataflowBuffer, uint32_t)` (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:223, 183`) — same.
  - `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb_id, pool, dim, reduce_factor>()` (H reader) and `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, in, scaler, out>(…)` (H compute) — DFB identity rides as a `constexpr uint32_t` NTTP, which `dfb::name`'s constexpr cast covers in template-parameter position.
  - the `*_with_dt` helpers in `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` — all take `DataflowBuffer` by value.

- **RTA varargs:** **none.** Every runtime arg in every in-scope kernel is a distinct field at a fixed index — name them all. One shape not to mis-flag: `reader_moreh_mean_nc.cpp:12-19` (out of scope anyway) pulls seven args through a running `i++`, but it is a fixed run at the top of the kernel over a fixed set — the recipe's explicit non-signal, not a loop.
