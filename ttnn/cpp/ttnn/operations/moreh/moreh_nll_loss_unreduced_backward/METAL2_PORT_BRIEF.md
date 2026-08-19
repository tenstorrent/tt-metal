# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · TensorParameter relaxation ✓ (`none`) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A — no site passes one)

**Recipe docs:** `f6033c9ec2d 2026-08-19 docs(metal_2.0): a direct-descriptor op converts to a real program factory` *(carry this line into the port report's Provenance section)*

**This op was RED on a previous audit and was fixed for you.** `047fecfec7f (#53534)` added the two scratch-CB allocations the readers were already using — `c_7` under `weight_has_value` in all three rank paths, `c_8` in 2d only. Consequence for you: `c_7`/`c_8` are **live, conditionally-allocated, sync-free CBs**, not leftovers. Do not read them as dead and drop them.

## Scope — one factory, three rank paths, no compute kernel

`MorehNllLossUnreducedBackwardDeviceOperation::Factory::create_descriptor` (`device/moreh_nll_loss_unreduced_backward_program_factory.cpp:443`) branches on `input_grad.logical_shape().rank()` into three free functions in the same file. They are **configs of one factory** — one port converts all three.

| Rank path | Reader | Writer |
|---|---|---|
| 2d (`:46`) | `reader_moreh_nll_loss_unreduced_backward_2d.cpp` | `writer_moreh_nll_loss_unreduced_backward.cpp` |
| 3d (`:182`) | `reader_moreh_nll_loss_unreduced_backward_3d.cpp` | *(same)* |
| 4d (`:318`) | `reader_moreh_nll_loss_unreduced_backward_4d.cpp` | *(same)* |

**There is no compute kernel.** The readers compute `input_grad` themselves with `CoreLocalMem` scalar writes into the output CB. So every program is reader + writer only, and most CBs are reader-local.

**Config axes you must carry through:** rank ∈ {2d, 3d, 4d} × **`WEIGHT`** (optional `weight_tensor`). Dtypes are pinned by validation — `target` INT32, `output_grad` / `weight` / `input_grad` BFLOAT16.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`.

- **Current concept:** `descriptor` — `static ProgramDescriptor create_descriptor(...)` on the nested `Factory` struct (`..._device_operation.hpp:35-40`).
- **Op-owned tensors:** none. Output is the caller's `input_grad_tensor` or an ordinary `create_device_tensor`.
- **Target concept:** `ProgramSpecFactoryConcept` — the **base** concept. `Override runtime args method?` is `no`, so there is nothing to translate and this is *not* a `CustomProgramSpecFactoryConcept` port.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none` `TensorParameter relaxation` (the cell reads `none`) · `get_dynamic_runtime_args`. A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list — none of them gate — and on this op all three are absent anyway: nothing to preserve, translate, or delete, so this port carries no user-visible API change.

**The direct-descriptor exception does *not* apply here — check this before you touch the device-op.** The recipe's newest sanctioned exception (*"Give a direct-descriptor op a conventional program factory"*) fires only when the device-operation declares `create_descriptor` as its own static member **with no `program_factory_t`**. This op already has both the nested struct and `using program_factory_t = std::variant<Factory>` (`..._device_operation.hpp:35-42`), so the exception is closed: your port is a **method swap inside the existing `Factory` struct**, and the device-operation class stays untouched. Flagged because a moreh op whose factory struct is named bare `Factory` looks like the shape that exception targets. The `<OpName>ProgramFactory` naming convention belongs to the conversion case — renaming an existing struct is not port work.

## Construct — to do

### Tensor bindings — four, all Case 1, one conditional

| Binding | Legacy delivery | Kernel use | Case |
|---|---|---|---|
| `target` | `Buffer*`, reader RTA 0 | `TensorAccessor(target_args, target_addr)` → donor `read_tile` | **1** |
| `output_grad` | `Buffer*`, reader RTA 1 | `TensorAccessor(output_grad_args, output_grad_addr)` → `read_line` (2d) / `read_tile` (3d/4d) | **1** |
| `weight` | `Buffer*` **or `nullptr`**, reader RTA 2 | `TensorAccessor(weight_args, weight_addr)` → `read_line`, inside `#if defined(WEIGHT)` | **1**, conditional |
| `input_grad` | `Buffer*`, writer RTA 0 | `TensorAccessor(input_grad_args, input_grad_addr)` | **1** |

Express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::…)`. The legacy `Buffer*` RTA slot and its `TensorAccessorArgs` CTA block both disappear.

**No Case 2 — and this op will try to convince you otherwise.** The readers are dense with raw typed pointers:

```cpp
CoreLocalMem<volatile uint16_t> input_grad_l1_ptr(dfb_input_grad_obj.get_write_ptr());
CoreLocalMem<volatile int32_t>  target_l1_ptr(dfb_target_obj.get_read_ptr());
CoreLocalMem<volatile uint16_t> output_grad_l1_ptr(dfb_output_grad_obj.get_read_ptr());
```

Every one is a pointer into **CB/L1 memory obtained from a DFB method** — not a tensor base from an RTA. No kernel does address arithmetic on a tensor base, so **no binding needs the `get_bank_base_address` bridge**. Leave the raw L1 arithmetic exactly as it is; it is CB access, which the DFB binding already covers.

Delivery note: the bases arrive as `Buffer*` entries (not `->address()`), so the framework already patches them on cache hits. Routine conversion, not a stale-pointer repair.

### The placeholder CTA chain goes away

The factory appends **three** `TensorAccessorArgs` blocks to the reader CTAs *unconditionally*, passing `nullptr` for an absent `weight`:

```cpp
TensorAccessorArgs(*target.buffer()).append_to(reader_compile_time_args);
TensorAccessorArgs(*output_grad.buffer()).append_to(reader_compile_time_args);
TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(reader_compile_time_args);
```

A null block still emits two words, which is what keeps the kernel's offset chain aligned across configs:

```cpp
constexpr auto target_args      = TensorAccessorArgs<0>();
constexpr auto output_grad_args = TensorAccessorArgs<target_args.next_compile_time_args_offset()>();
constexpr auto weight_args      = TensorAccessorArgs<output_grad_args.next_compile_time_args_offset()>();
```

Under Metal 2.0 the framework builds accessor args from the bindings, so **the placeholder block and the whole offset chain both disappear**. Express `weight` as a conditional binding — do not carry a null placeholder forward to preserve a chain that no longer exists.

### CB endpoints — five self-loops and one plain pair

| CB | Role | Disposition | Configs |
|---|---|---|---|
| `c_0` | `target` | **self-loop** — reader produces (`read_tile`), consumes (`wait_front`/`pop_front`) *and* peeks (`get_read_ptr`); no other toucher | all |
| `c_1` | `output_grad` | **self-loop** — reader-only. 2d: `read_line(Nt)` then `wait_front(Nt)` + peek, never popped. 3d/4d: `read_tile` per iteration, popped | all |
| `c_2` | `weight` | **self-loop** — reader produces via `read_line(Ct)`, then `wait_front(Ct)` + peek, never popped | `WEIGHT` only |
| `c_7` | `weight_scratch` | **self-loop** — **sync-free**: NoC-written and `get_write_ptr()`-read inside `read_line`, **no FIFO ops at all** | `WEIGHT` only |
| `c_8` | `output_grad_scratch` | **self-loop** — sync-free, same shape | **2d only** |
| `c_16` | `input_grad` | plain **1P+1C** — reader produces (`reserve_back` + raw write + `push_back`), writer consumes | all |

Self-loop means bind the one toucher **PRODUCER and CONSUMER** — legal on Gen1 for DM. Five of the six CBs are reader-local, which is what having no compute kernel buys you.

**Declare `c_2`, `c_7` and `c_8` conditionally**, mirroring the guards the factory already has: `c_2` and `c_7` under `weight_has_value` (`:85`, `:90` and the 3d/4d equivalents), `c_8` only on the 2d path (`:93`). This is *not* the dead-CB-derived conditional — none of these is ever dead where it is allocated; the legacy factory already gates allocation to match kernel usage exactly, and you carry that conditionality into the DFB specs. Note the kernel side is already guarded to match (`#if defined(WEIGHT)` wraps the `c_2`/`c_7` object constructions), so no construction needs relocating.

Two consumers legitimately never pop — `c_1` on the 2d path (the whole `Nt`-tile row is held for the loop) and `c_2` (the whole weight line is held). A held single-toucher CB is still a self-loop; don't read the missing pop as a missing endpoint.

**No multi-binding anywhere, and you need not re-run the hunt.** Both faces were checked: the reader's raw `get_write_ptr()` write into `c_16` is bracketed by its own `reserve_back`/`push_back` (the producer's own peek, not a hidden second writer), the op has **no semaphores at all** to coordinate a co-fill, and no CB is touched by two co-resident kernels except `c_16`'s legal pair. No dead CB either — that was the previous audit's RED, and it is fixed.

### Donor call sites need no bridging work

The readers call `read_tile(DataflowBuffer, AddrGen, uint32_t, …)` and `read_line(DataflowBuffer, DataflowBuffer, AddrGen, uint32_t, …)` from `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (`:666`, `:739`), plus `get_tilized_idx` (`:618`) and the scalar `bf16_to_fp32` / `fp32_to_bf16_truncate` from `tt_metal/hw/inc/api/numeric/bfloat16.h`.

`DataflowBuffer` **by value** is the recipe's ✓ *excellent* donor row — **no donor-side change, no fork.** Build the named object from the token and pass it, which is what these readers already do:

```cpp
DataflowBuffer dfb_weight_obj(dfb::weight);
DataflowBuffer dfb_weight_scratch_obj(dfb::weight_scratch);
read_line(dfb_weight_obj, dfb_weight_scratch_obj, addrg_weight, Ct);
```

This is the ✓ row, **not** the adjacent ⭐-flagged `CircularBuffer&` row — the donor already migrated to the DFB type, so only the handle's source changes. The accessor parameter is Shape 1 (`TensorAccessor` as a template argument), also ✓: pass `TensorAccessor(tensor::name)` straight through.

### `get_dataformat` metadata lookups — delete, do not modernise

Each reader opens with two calls whose results are **never read** (`_2d.cpp:34,36`; `_3d.cpp:33,35`; `_4d.cpp:33,35`):

```cpp
const DataFormat weight_data_format      = get_dataformat(cb_weight);       // c_2 — absent unless WEIGHT
const DataFormat output_grad_data_format = get_dataformat(cb_output_grad);
```

Whitelist rule 7 would normally have you move a metadata lookup onto the object or the token — **not here.** The `weight` one queries a CB that is **not allocated** in the non-`WEIGHT` config, so `dfb::weight` will not exist to read a format from, and both sit outside the `#if defined(WEIGHT)` guard. They are provably dead, so deleting them is behaviour-preserving. The audit has asked the ops team to confirm the deletion (Question 1) — if that has not landed when you start, raise it rather than guessing, because the alternative has no expressible form.

*(If you do keep any live metadata lookup elsewhere: rule 7's test is the legacy declaration. These are `const`, not `constexpr`, so they would take the member-getter form — but they are being deleted, so the question is moot.)*

## Watch for

- **CB endpoints (multi-binding):** none — see above.
- **Cross-op / shared kernels:** **none.** All four kernel sources live in this op's own `device/kernels/` and each has exactly one binder repo-wide; the op borrows no kernel *file* and lends none. No `_metal2` fork exists beside any of them, and there is no `experimental/quasar/` copy of this op — **so no fork question arises at all.** The only sharing is header-level, and both donors are ✓ (above).
- **RTA varargs:** none — name every arg. Each reader pulls 9 args and the writer 3 through a fixed `i++` run at the top of `kernel_main`, which is ordinary positional plumbing that dissolves into named args, **not** a loop. No counted loop over arg indices, no data-selected read. Names from the kernel locals: `target_addr`, `output_grad_addr`, `weight_addr`, `ignore_index`, `num_tiles_per_core`, `start_id`, then per rank `Nt`/`C`/`Ct` (2d), `C`/`Ct`/`Wt` (3d), `num_inner_tile`/`C`/`Ct` (4d); writer `input_grad_addr`, `num_tiles_per_core`, `start_id`.
- **No CTAs exist in this op.** Not one `get_compile_time_arg_val` call in any kernel — the only compile-time args are the `TensorAccessorArgs` blocks. Per-core work division rides an RTA (`units_per_core`), so the *demoting-per-group-CTA* anti-pattern has no purchase here and there is no per-core-group kernel pair to preserve.
- **Anything else you need:** the **whole compute-kernel-config path is vestigial** — the op has no compute kernel, `ComputeConfigDescriptor` appears nowhere, four of the five values from `get_compute_kernel_config_args` are unused, and the fifth drives an `FP32_DEST_ACC_EN` define that **no kernel reads** (zero hits under `device/kernels/`). Don't hunt for a compute kernel to configure, and don't carry that define into the port. The attribute itself stays (removing it is an ops-team API change, flagged in the audit).
- **`experimental/quasar/` has no copy of this op or its kernels.** Nothing there to mistake for prior art — the tree is out of bounds either way.
