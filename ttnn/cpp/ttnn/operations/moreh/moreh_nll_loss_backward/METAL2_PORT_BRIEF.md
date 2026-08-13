# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · TensorParameter relaxation ✓ (`none`) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `38da2cdbd29 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port` *(carry this line into the port report's Provenance section)*

**Scope: one DeviceOperation, one factory, three rank-dispatched code paths, five kernels.**

`MorehNllLossBackwardDeviceOperation::Factory::create_descriptor` (`device/moreh_nll_loss_backward_program_factory.cpp:691`) branches on `input_grad.logical_shape().rank()` into three free functions in the same file. These are **configs of one factory**, not three factories — one port converts all three.

| Rank path | Reader | Writer | Compute |
|---|---|---|---|
| 2d — `moreh_nll_loss_backward_impl_2d` (`:46`) | `reader_moreh_nll_loss_backward_2d.cpp` | `writer_moreh_nll_loss_backward.cpp` | `moreh_nll_loss_backward_kernel.cpp` |
| 3d — `_impl_3d` (`:259`) | `reader_moreh_nll_loss_backward_3d.cpp` | *(same)* | *(same)* |
| 4d — `_impl_4d` (`:474`) | `reader_moreh_nll_loss_backward_4d.cpp` | *(same)* | *(same)* |

**Config axes you must carry through the whole port:** rank ∈ {2d, 3d, 4d} × **`WEIGHT`** (optional `weight_tensor`) × **`DIVISOR`** (optional `divisor_tensor`) × `fp32_dest_acc_en` (formats only). The two optional-tensor axes drive almost every non-mechanical decision below — read *Construct → CB endpoints* before you touch the CB specs.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — one `static ProgramDescriptor create_descriptor(...)` on `Factory` (`device/moreh_nll_loss_backward_device_operation.hpp:38-43`).
- **Op-owned tensors:** none. The output is either the caller's preallocated `input_grad_tensor` or an ordinary `create_device_tensor` (`...device_operation.cpp:90-98`).
- **Target concept:** `ProgramSpecFactoryConcept` — the **base** concept. `Override runtime args method?` is `no`, so there is no `override_runtime_arguments` to translate and this is *not* a `CustomProgramSpecFactoryConcept` port.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none` `TensorParameter relaxation` (the cell reads `none`) · `get_dynamic_runtime_args` (hook absent from the device-op). A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list — none of them gate — and on this op all three happen to be absent anyway: **no** custom `compute_program_hash` to preserve, **no** override method to translate, and **no** pybound `create_descriptor` to delete, so this port carries no user-visible API change from that column.

## Construct — to do

### Tensor bindings — five, all Case 1, two conditional

| Binding | Legacy delivery | Kernel use | Case |
|---|---|---|---|
| `target` | `Buffer*`, reader RTA 0 | `TensorAccessor(target_args, target_addr)` → donor `read_tile` | **1** |
| `output_grad` | `Buffer*`, reader RTA 1 | `TensorAccessor(output_grad_args, output_grad_addr)` → `read_tile` | **1** |
| `weight` | `Buffer*` **or `nullptr`**, reader RTA 2 | `TensorAccessor(weight_args, weight_addr)` → `read_line`, inside `#if defined(WEIGHT)` | **1**, conditional |
| `divisor` | `Buffer*` **or `nullptr`**, reader RTA 3 | `TensorAccessor(divisor_args, divisor_addr)` → `read_tile`, inside `#if defined(DIVISOR)` | **1**, conditional |
| `input_grad` | `Buffer*`, writer RTA 0 | `TensorAccessor(input_grad_args, input_grad_addr)` | **1** |

Express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::…)`. The legacy `Buffer*` RTA slot and its `TensorAccessorArgs` CTA block both disappear.

**No Case 2 — and this op will try to convince you otherwise.** The readers use raw typed pointers heavily:

```cpp
CoreLocalMem<volatile int32_t>            target_l1_ptr(dfb_target_obj.get_read_ptr());
CoreLocalMem<volatile uint16_t>           weight_l1_ptr(dfb_weight_obj.get_read_ptr());
CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_weight_l1_ptr(dfb_tmp_weight_obj.get_write_ptr());
```

Every one of those is a pointer into **CB/L1 memory obtained from a DFB method** — not a tensor base address obtained from an RTA. No kernel does address arithmetic on a tensor base, so **no binding needs the `get_bank_base_address` bridge**. Leave this raw L1 walking exactly as it is; it is CB access, which the DFB binding already covers.

Delivery note: the bases arrive as `Buffer*` entries (the factory says why at `:197-198`), so the framework already patches them on cache hits. Routine conversion, not a stale-pointer repair.

### Conditional bindings and the placeholder CTA chain — this all goes away

The factory appends **four** `TensorAccessorArgs` blocks to the reader CTAs *unconditionally*, passing `nullptr` for an absent optional (`:111-114`, `:325-328`, `:542-545`). A null block still emits two words (`args_config_.raw()` and `aligned_page_size = 0`), which is what keeps the kernel's offset chain aligned across configs:

```cpp
constexpr auto target_args      = TensorAccessorArgs<0>();
constexpr auto weight_args      = TensorAccessorArgs<target_args.next_compile_time_args_offset()>();
constexpr auto divisor_args     = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
constexpr auto output_grad_args = TensorAccessorArgs<divisor_args.next_compile_time_args_offset()>();
```

Under Metal 2.0 the framework builds accessor args from the bindings, so **the placeholder blocks and the entire offset chain both disappear**. Express `weight` and `divisor` as conditional bindings instead — do not carry a null placeholder binding forward to preserve the chain; there is no chain left to preserve.

### CB endpoints — read this before writing the DFB specs

Four plain 1P+1C pairs, four self-loops, and **three CBs that are allocated in configs where nothing touches them**. That last group is the substance of this port: a DFB with neither a producer nor a consumer binding is rejected by the spec validator, so these cannot be carried across as-is.

| CB | Role | Disposition | Configs |
|---|---|---|---|
| `c_0` | `output_grad` | plain 1P+1C — reader produces, compute consumes (holds, never pops) | all |
| `c_1` | `target` | **self-loop** — reader produces, consumes *and* peeks it; no other toucher | all |
| `c_2` | `weight` | **self-loop** — reader produces via `read_line`, then `wait_front` + `get_read_ptr` (never pops) | `WEIGHT` only |
| `c_3` | `divisor` | plain 1P+1C — reader produces, compute consumes | `DIVISOR` only |
| `c_7` | `weight_scratch` | **self-loop** — sync-free: NoC-written and `get_write_ptr()`-read inside `read_line`, **no FIFO ops at all** | `WEIGHT` only |
| `c_8` | *(intended output_grad scratch)* | **drop — confirmed dead CB** | 2d only |
| `c_16` | `input_grad` | plain 1P+1C — compute produces, writer consumes | all |
| `c_24` | `tmp_weight` | plain 1P+1C — reader produces (`reserve_back` + raw write + `push_back`), compute consumes | all |
| `c_25` | `tmp1` | **self-loop** under `DIVISOR`; **dead** without it → **gate the allocation** | see below |
| `c_26` | `tmp2` | **self-loop** under `DIVISOR`; **dead** without it → **gate the allocation** | see below |

**Drop `c_8`** @ `program_factory.cpp:107` (2d path). Confirmed unreferenced: no kernel names `c_8`, no CB-index CTA exists to thread it through, every `read_tile`/`read_line` call site passes an explicitly named DFB object, and the 2d output_grad read is the **3-argument** `read_tile` overload that takes no scratch at all. The factory's comment claiming a scratch is needed is stale — a full-tile read is naturally aligned, which is why only `read_line`'s sub-tile slices need `c_7`. A dead CB has no behaviour, so removing it changes none. Record the drop with `file:line` in the port report.

**Gate `c_25` and `c_26` on `divisor_has_value`** — do **not** drop them, and do not keep them unconditional. Both are allocated unconditionally (`:96-97`, `:312-313`, `:529-530`) but every *use* is inside `#if defined(DIVISOR)` (`moreh_nll_loss_backward_kernel.cpp:36,46,49,57,67,70,74,75,78,87`). Dropping them breaks the divisor path; keeping them unconditional leaves the non-divisor spec with two bindingless DFBs that will not validate. Make the `DataflowBufferSpec` conditional on the same predicate the factory already uses for `c_3` — which `push_cb` implements today by early-returning on `num_tiles == 0` (`:29-31`).

**And move two kernel-side constructions inside the guard.** This is the part a touch-based census does not show you:

```cpp
// moreh_nll_loss_backward_kernel.cpp:22-25 — constructed unconditionally,
// but every USE is inside #if defined(DIVISOR)
constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
DataflowBuffer dfb_tmp1_obj(cb_tmp1);
constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
DataflowBuffer dfb_tmp2_obj(cb_tmp2);
```

Constructing a `DataflowBuffer(dfb::x)` requires the binding to exist. Once the specs are conditional, these two constructions must move inside the existing `#if defined(DIVISOR)` guard — otherwise the non-divisor build names two DFBs the spec no longer declares. Reusing the op's own existing define is whitelist rule 6 territory, not a new conditional you are inventing.

**No multi-binding anywhere.** Both faces were hunted and came back empty — you do not need to re-run the hunt. In particular, the reader's raw `get_write_ptr()` write into `c_24` is bracketed by its own `reserve_back`/`push_back`: that is the producer's own peek, not a hidden second writer. There are no semaphores in this op at all.

### The dead `get_dataformat(cb_id)` locals — delete, do not modernise

Each reader opens with three calls whose results are **never read** (`reader_..._2d.cpp:34,36,38`; `_3d.cpp:35,37,39`; `_4d.cpp:35,37,39`):

```cpp
const DataFormat weight_data_format      = get_dataformat(cb_weight);   // c_2 — absent unless WEIGHT
const DataFormat divisor_data_format     = get_dataformat(cb_divisor);  // c_3 — absent unless DIVISOR
const DataFormat output_grad_data_format = get_dataformat(cb_output_grad);
```

Whitelist rule 7 would normally have you move a metadata lookup onto the object — **do not do that here.** Two of the three query CBs that are *not allocated* in the non-`WEIGHT` / non-`DIVISOR` configs, so `dfb::weight` and `dfb::divisor` will not exist to read a format from; and all three sit outside the guards that wrap every real use of those CBs. They are provably dead, so deleting them is behaviour-preserving. The audit has asked the ops team to confirm the deletion (Question 2) — if that confirmation has not landed when you start, raise it rather than guessing, because the alternative (keeping them) has no expressible form.

### Per-core-group compute pair — keep both specs

Each rank path emits the compute kernel **twice** from one source over **disjoint** core groups, differing only in the leading CTA: `{units_per_core_group_1, divisor_has_value}` vs `{units_per_core_group_2, divisor_has_value}` (`:169`/`:185`, `:383`/`:399`, `:600`/`:616`, each guarded by `has_core_group_2`). Port as **two `KernelSpec`s of the same source in two `WorkUnitSpec`s** over disjoint `target_nodes`, both binding the same DFBs. Collapsing them into one spec by demoting that CTA to an RTA is the *Demoting per-group CTA to RTA* anti-pattern (`port_patterns.md`) — real kernel-perf cost, false premise. Because the node sets are disjoint each node still sees one compute instance: ordinary single-role bindings, **not** the multi-binding flag.

### Donor call sites need no bridging work

The shared-pool helpers this op calls take either a `DataflowBuffer` **by value** — `read_tile`, `read_line` (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:666`, `:739`), and `copy_tile_init_with_dt` / `pack_tile_with_dt` / `mul_bcast_scalar_init_with_dt` (`kernel/compute/moreh_common.hpp:35`, `:28`, `:121`) — or the accessor as a **template parameter** (`AddrGen`, instantiated with `TensorAccessor`). Both bridge directly: the implicit `DataflowBuffer(DFBBindingToken)` constructor (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:106`) takes `dfb::name`, and `AddrGen` takes a `TensorAccessor(tensor::name)`. Pass the handles straight through — no `.id` extraction, no temp DFB wrappers. The LLK compute calls (`init_sfpu`, `copy_tile`, `mul_tiles_bcast_scalar`, `recip_tile`, `negative_tile`) take raw `uint32_t` ids by design and are covered by the token's `constexpr operator uint32_t()` (`:89`).

### Sanctioned free function you may retire on the object

The writer looks up its tile size with `get_tile_size(cb_input_grad)` (`writer_moreh_nll_loss_backward.cpp:26`), and the donor's `read_tile`/`read_line` use `get_tile_size(cb.get_id())` internally. `DataflowBuffer::get_tile_size()` exists as a member (`dataflow_buffer.h:201`), so moving the writer's lookup onto the object is ordinary port work under whitelist rule 7. This was **not** a Device 2.0 finding — the free function is sanctioned at that stage. (The donor internals are out of your scope; leave them.)

## Watch for

- **CB endpoints (multi-binding):** none — see above. Nothing to flag, nothing to re-hunt.
- **Cross-op / shared kernels:** **none.** All five kernel sources live in this op's own `device/kernels/` and are bound only by this op's single factory; the only outside reference to the directory is the family CMake glob (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:42`). No op borrows these files, this op borrows none, and no `_metal2` fork exists beside any of them — **so no fork question arises at all.** The writer and compute kernel each have three binding call sites, but all three are rank branches of the *same* factory, so one port converts every binder at once.

  The real consequence is narrower and easy to trip over: **any edit to `writer_moreh_nll_loss_backward.cpp` or `moreh_nll_loss_backward_kernel.cpp` must satisfy all three rank paths**, and those paths do not supply identical args — the 2d reader takes **10** runtime args, 3d and 4d take **11** (3d/4d add `num_inner_tile`). The writer's three args and the compute kernel's args are the same across paths, but verify per path rather than porting against the 2d impl alone.
- **RTA varargs:** none — name every live arg. The readers pull 10–11 args through a running `i++` counter at the top of `kernel_main` (`get_arg_val<uint32_t>(i++)`), which is a fixed run over a fixed set — ordinary positional plumbing that dissolves into named args, **not** a loop. No counted loop over arg indices, no data-selected read. Names are legible from the kernel locals: `target_addr`, `output_grad_addr`, `weight_addr`, `divisor_addr`, `ignore_index`, `num_tiles_per_core`, `start_id`, `C`, (`num_inner_tile` in 3d/4d), `weight_num_tile`, `element_size`; writer `input_grad_addr`, `num_tiles_per_core`, `start_id`.
- **Four dead args and one dead CTA — do not invent names for them.** Naming is your job, so here is what has no meaning to name:
  - Reader `element_size` — read into an unused local in all three readers (`_2d.cpp:22`, `_3d.cpp:23`, `_4d.cpp:23`); computed host-side at `:205`, `:419`, `:636`.
  - Compute RTA index **0** — never read at all (the tile count comes from CTA 0).
  - Compute RTA index **1** (`tile_offset`) — read into an unused local (`moreh_nll_loss_backward_kernel.cpp:14`). Note the kernel reads index 1 while never reading index 0, so the arg vector exists only to position a value nothing uses.
  - Compute CTA index **1** (`divisor_has_value`, `:169` etc.) — never read; the kernel branches on the `DIVISOR` define, which the factory also supplies (`:129`).

  Removal routes to the ops team, not to the port — but do not carry them across as named args either. Flag whichever you leave in the port report.
- **`reduction_mean` is unused by every impl.** It is public, stored in `operation_attributes_t`, threaded into `create_descriptor`, and then ignored — all three impls take it as `const bool /*reduction_mean*/` (`:52`, `:265`, `:480`). Do not go looking for the behaviour it implies; there isn't any. The audit has flagged it to the ops team (Question 3) as a possible functional bug independent of this port. Leave the attribute alone.
- **`experimental/quasar/` has no copy of this op or its kernels.** Nothing there to mistake for prior art or for a fork to reuse — the tree is out of bounds either way.
