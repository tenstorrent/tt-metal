# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

## Scope — what you are porting

Two DeviceOperations, **five factories**, in one directory. They share the writer kernel, so they port as one unit:

- **`LayerNormPreAllGatherDeviceOperation`**
  - `LayerNormPreAllGatherProgramFactory` — `layernorm_pre_all_gather_program_factory.cpp:25`
  - `LayerNormPreAllGather2DProgramFactory` — `layernorm_pre_all_gather_program_factory.cpp:295`
  - `LayerNormPreAllGatherWelfordProgramFactory` — `layernorm_pre_all_gather_welford_program_factory.cpp:23`
- **`LayerNormPostAllGatherDeviceOperation`**
  - `LayerNormPostAllGatherProgramFactory` — `layernorm_post_all_gather_program_factory.cpp:29` — **two configs in one factory**: 1D work split, and the `use_2d_core_grid` 2D split (`:149`)
  - `LayerNormPostAllGatherWelfordProgramFactory` — `layernorm_post_all_gather_welford_program_factory.cpp:44`

**Ten kernels** are in scope. Eight live in `device/kernels/`; two are file-path-instantiated from the sibling family on the `is_rmsnorm` branch — `rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` and `.../rmsnorm_post_allgather.cpp`. Those branches are live: `ttnn::rms_norm_pre_all_gather` / `rms_norm_post_all_gather` call straight into these prim device ops, so RMSNorm is a config of this op, not a separate op.

**Configs that change the CB set** (the port must hold all of them): `is_rmsnorm` (selects a different compute kernel), `fuse_pre_add` (residual present), `gamma` / `beta` present-or-absent independently, `use_2d_core_grid`, and Welford-vs-not. Two combinations are rejected by validation and need no port coverage: RMSNorm + Welford (`layernorm_post_all_gather_device_operation.cpp:166-171`, `layernorm_pre_all_gather_welford_program_factory.cpp:46`) and Welford + 2D core grid (`layernorm_pre_all_gather_device_operation.cpp:62-69`).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — five `create_descriptor` entry points, no workload form.
- **Op-owned tensors:** none. Note the one thing that looks like one and isn't: the Welford pre factory backs `c_2` with a reciprocal-LUT tensor (`layernorm_pre_all_gather_welford_program_factory.cpp:369`), but that tensor is **caller-supplied** — it arrives through `tensor_args_t` (`layernorm_pre_all_gather_device_operation_types.hpp:25`) and is created by the separate `ttnn.create_layer_norm_reciprocals` API. It is an ordinary `TensorParameter`, not an op-owned buffer, so it does **not** force the `WorkloadDescriptor` shape.
- **Target concept:** `ProgramSpecFactoryConcept` (no op-owned tensors) — the readiness sheet's own `Porting Target` column says the same on all five rows.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus **other migration-risky pybind**, which surfaces as a `safe` warning that also fails the gate. All `no` on the sheet's five rows and confirmed by grep.
- **Already named:** the Welford pre factory uses `named_compile_time_args` for one CTA (`layernorm_pre_all_gather_welford_program_factory.cpp:146-148,280`, read back as `get_named_compile_time_arg_val("welford_unpack_fp32_active")` at `layernorm_pre_allgather_welford.cpp:43`). That one carries over as-is.

## Construct — to do

### Tensor bindings

Seven bindings, six Case 1 and one clean. **No Case 2 anywhere** — no kernel does hand-rolled NoC arithmetic on a tensor base, so you never need the `get_bank_base_address` bridge in this op.

| Binding | Where | Today | Action |
|---|---|---|---|
| `input` | all 5 factories | `Buffer*` in reader RTA slot 0 (`…pre_all_gather_program_factory.cpp:174`; `…post_all_gather_program_factory.cpp:316,351`) | **Case 1** → `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::input)` (replaces `reader_…pre_allgather.cpp:34`, `reader_…post_allgather.cpp:100`, `reader_…2d.cpp:48`) |
| `residual_input_tensor` | Pre ×3, `fuse_pre_add` only | `Buffer*` in reader RTA (`:179`, `:438`, welford `:192`) | **Case 1** → same treatment (`reader_…pre_allgather.cpp:44`, `reader_…2d.cpp:61`) |
| `stats` | Post ×2 | `Buffer*` in reader RTA (`…post_all_gather_program_factory.cpp:324,359`; welford `:357,393`) | **Case 1** (`reader_…post_allgather.cpp:101`) |
| `gamma` | Post ×2, **optional** | `Buffer*` **or `nullptr`** in reader RTA (`:322,358`) | **Case 1** + drop the 3rd arg (below) |
| `beta` | Post ×2, **optional** | `Buffer*` **or `nullptr`** in reader RTA (`:323,358`) | **Case 1** + drop the 3rd arg (below) |
| `output` | all 5 factories | `Buffer*` in writer RTA slot 0 (`:184,444`; post `:328,362`) | **Case 1** (`writer_…:26`) |
| `recip_tensor` | Pre-Welford only | **borrowed-memory CB** — `.buffer = recip_tensor.buffer()` on `c_2` (`…welford_program_factory.cpp:362-369`); never an RTA | **clean** → `DataflowBufferSpec::borrowed_from` the `recip_tensor` binding. The kernel's raw read (`get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0)`, `layernorm_pre_allgather_welford.cpp:75`) stays as it is — the DFB *is* the tensor access |

All six RTA-delivered bases arrive as `Buffer*` (the framework's interim binding hack), which is already patched on cache hits — so this is **routine port work, not a correctness fix**.

**Settle the optional-binding shape first.** `gamma` and `beta` are independently optional and the current code leans on `nullptr`-as-`0u`, including a matching `TensorAccessorArgs(nullptr)` (`layernorm_post_all_gather_program_factory.cpp:111-112,251-254`). Four of the post factories' six configs exercise at least one absent case, so decide how an absent `TensorParameter` is expressed before writing the post-allgather reader's bindings rather than after.

### TensorAccessor 3rd arg

**Drop the page-size argument at both sites** — both Class 2 (redundant/inert), pure no-ops:

- `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp:104` — `TensorAccessor(gamma_args, gamma_addr, gamma_stick_size)`
- `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp:107` — `TensorAccessor(beta_args, beta_addr, beta_stick_size)`

Metal 2.0 supplies the `aligned_page_size` implicitly and it equals what is being passed: validation pins gamma/beta to BFLOAT16 or FLOAT32 (`layernorm_post_all_gather_device_operation.cpp:78-81,123-126`), so the TILE branch's `element_size() * 1024` is exactly `tile_size()` (2048 / 4096 B) and the ROW\_MAJOR branch's `padded_shape[-1] * element_size()` is exactly the stick (64 / 128 B, with the width pinned to `tile_width` at `:107`). No `dynamic_tensor_shape` relaxation — that is Class 1, and this is not it.

Once dropped, the host-side `gamma_stick_size` / `beta_stick_size` computation (`layernorm_post_all_gather_program_factory.cpp:219-242`, `…welford_…:259-282`) is still needed for the `gamma_is_row_major` / `beta_is_row_major` flags but its byte values become dead; check whether the CTAs carrying them are still read by anything before removing them.

> **Note if you also touch the triage doc:** `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md` still lists this op as **Class 3 — latent bug** over a block-float gamma. That row is stale — validation now rejects BFLOAT8_B gamma/beta, so the branch it flagged is unreachable. The audit routed the correction to the triage-doc owner; do not re-derive it, and do not let the stale row stop you dropping the arg.

### TensorParameter relaxation

**None.** Sheet says `none` on all five rows, and none *could* apply — a relaxation is a custom hash excluding a property from the cache key, and neither DeviceOperation declares `compute_program_hash`.

### CB endpoints

Most CBs are ordinary 1:1 reader→compute→writer FIFOs and need nothing. Four things do:

**Self-loop** (one toucher — bind the single kernel PRODUCER *and* CONSUMER; legal on Gen1 for compute and DM alike):

| Factory | CBs |
|---|---|
| Pre 1D | `c_3` *(fuse only)*, `c_6` |
| Pre 2D | `c_3` *(fuse only)*, `c_6` |
| Pre Welford | `c_3`, `c_4`, `c_6` *(all fuse only)*, and `c_2` — the borrowed recip LUT, raw-peek only |
| Post default | `c_6`, `c_7` *(LN only)*, `c_8`, `c_10`, `c_11` *(LN only)*, `c_12`, `c_13` *(LN + beta)* |
| Post Welford | `c_5`, `c_6`, `c_10`, `c_11`, `c_12`, `c_13` *(beta only)* |

`c_5` in the Post-Welford factory is a self-loop for an unusual reason worth knowing: the shared reader fills it (`reader_…post_allgather.cpp:111-115`) and the Welford compute kernel never reads it — an orphan producer, one toucher. Bind it both roles; don't go hunting for the missing consumer.

**1P+1C assignment:** not needed anywhere. Every two-toucher CB in this op is already one locked producer + one locked consumer.

**Multi-binding advanced option — exactly one CB:**

- **`(c_1, LayerNormPreAllGatherWelfordProgramFactory, all configs)`** — two locked producers, which no relabelling removes. The reader unconditionally `reserve_back(1)`/`push_back(1)`s a reduce-scaler tile into `c_1` (`reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp:24,31-32` → `reduce_helpers_dataflow.inl:163,203`), because that reader is **shared with the 1D factory, where `c_1` genuinely is the scaler CB**. In *this* factory `c_1` is the compute kernel's post-Welford transpose scratch, with its own `reserve_back(2)`/`push_back(2)`/`wait_front(2)`/`pop_front(2)` (`layernorm_pre_allgather_welford.cpp:215-220,277-283,290-293`). Set the flag; it self-documents the Quasar debt. The underlying sloppiness is a real (though currently harmless) defect on the ops team's plate — see `METAL2_PREPORT_AUDIT.md` *Misc anomalies* #2 — but it is **not yours to fix in the port diff**.

**Dead-CB drops — four allocations, all confirmed unreferenced by every kernel in every config:**

| CB | Allocation site | Scope |
|---|---|---|
| `c_9` (`var + epsilon`) | `layernorm_post_all_gather_program_factory.cpp:490-496` | dead in **all** configs of the default post factory |
| `c_9` | `layernorm_post_all_gather_welford_program_factory.cpp:554-560` | dead in all configs |
| `c_7` (`mean²`) | `layernorm_post_all_gather_welford_program_factory.cpp:583-589` | dead — Welford factory only (**live** in the default post factory) |
| `c_8` (`var`) | `layernorm_post_all_gather_welford_program_factory.cpp:545-551` | dead — Welford factory only (**live** in the default post factory) |

A dead CB has no behavior, so removing it changes none — and a bindingless DFB cannot be expressed in Metal 2.0 at all, so these must go. **No dead CTA accompanies any of them** (neither post factory passes a CB index through a CTA, an RTA, or a named CTA), so the drop is the allocation only. Record each drop with `file:line` in the port report. Two kernel-side declarations become dead alongside `c_9` and can go with it: `layernorm_post_allgather.cpp:115` and `rmsnorm_post_allgather.cpp:52`.

## Watch for

- **CB endpoints (multi-binding):** the one flagged CB above. Its extra producer is **visible, not hidden** — no semaphore-gated raw co-fill is involved — so you do not need to hunt for it. The hidden-second-writer hunt was already run across every CB in all five factories and found nothing else: the only raw `get_write_ptr()` sites in the op are `reader_…post_allgather.cpp:167,178,201,212` (a kernel peeking a buffer it is itself the FIFO producer of — one toucher, not two) and `reader_layernorm_preallgather_2d.cpp:127`, which is a **remote** node's address, not a local endpoint (next bullet).
- **The 2D pre factory's cross-core merge is not a hidden co-fill.** `reader_layernorm_preallgather_2d.cpp:120-127` writes to `dfb_x2_merge_buf.get_write_ptr() + worker_offset` at `reduce_core_noc_x/y` — i.e. into a **different node's** `c_15` instance; the local `get_write_ptr()` is only a peek used to compute the identical offset. So `c_15`'s census on a merge node is just the local reader (locked producer via its `push_back` at `:137`, once the semaphore confirms all peers landed) plus the local compute (locked consumer, `:108,124`) → plain 1:1. The semaphore itself (`SemaphoreDescriptor{.id = 0, …}` at `layernorm_pre_all_gather_program_factory.cpp:455-456`, kernel-side `Semaphore<> reducer_sem` at `:54`) is not a CB toucher and ports as an ordinary `SemaphoreSpec`.
- **DFB core range narrower than its binding kernel's core range (2D pre factory).** `c_14` is declared over `merge_cores` (`layernorm_pre_all_gather_program_factory.cpp:579-585`) while the compute kernel that produces into it runs over `all_cores` (`:484`) and reaches it only under a runtime `if (is_merge_core)` (`layernorm_pre_allgather_2d.cpp:100-133`). `c_13` has the same asymmetry on its producer side. Legal with legacy CBs; **confirm the spec validator accepts it early**, before you build out the rest of that factory, rather than discovering it at first build.
- **RMSNorm + gamma + beta will not port as-is — and that is a pre-existing bug, not something you introduced.** `rmsnorm_post_allgather.cpp:63-65` sets `cb_times_gamma_out_idx = tt::CBIndex::c_13` when both gamma and beta are present, and drives it at `:153,160,173,182` — but `layernorm_post_all_gather_program_factory.cpp:517-545` allocates `c_13` **only** under `if (!is_rmsnorm)`. So that config already drives an unconfigured CB index today, and in Metal 2.0 you cannot bind a DFB no spec declares. The config is reachable (`rmsnorm_distributed/rmsnorm_post_all_gather.cpp:43-53` forwards a `bias` straight through, and nothing validates it away). **Do not fix it in the port diff** — it needs a functional decision (allocate the CB, or reject the config) that belongs to the ops team. Raise it and get a ruling before you commit to how that config binds; the audit filed it as *Questions* #2.
- **Cross-op / shared kernels:** **no `_metal2` fork exists beside any kernel this op uses** — a search for `*_metal2*` under `normalization/`, `kernel_lib/`, and `kernel/` returns nothing — so this port creates the first fork of whatever it touches. Two borrowed kernel files, both in-family: `rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` (from `layernorm_pre_all_gather_program_factory.cpp:143`) and `.../rmsnorm_post_allgather.cpp` (from `layernorm_post_all_gather_program_factory.cpp:287` and `…welford_…:316`). **Sunset list: empty beyond this op** — no other op binds either file, so the legacy copies retire with this port. *(This is a sunset list, not authorization to convert either kernel in place; the in-place rung needs an explicit bundled-port assignment you do not have.)* Two near-miss basenames that are **different files, not shared code**: `normalization/layernorm/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` (that op's own copy, `layernorm_op_multi_core.cpp:639`) and `experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_{pre,post}_allgather.cpp` (that op's own copies). Nothing outside this directory instantiates any kernel this op owns.
- **Donor call shapes are all ✓** — nothing needs a boundary workaround. `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb_id,…>()`, `prepare_zero_tile<dfb_id>()`, and `compute_kernel_lib::reduce<…, in_cb, scaler_cb, out_cb, …>()` take the CB index as an **NTTP** (the `dfb::name` constexpr cast covers template-parameter position); `pre_add::one_row(...)` and `combine_welford_partials(...)` take `DataflowBuffer&`; `generate_bcast_col_scalar(CircularBuffer cb, …)` takes the wrapper **by value** (not the flagged `CircularBuffer&` shape); `norm::…::memory::get_pointer_to_cb_data<To>(uint32_t cb_id, …)` takes a plain `uint32_t` (constexpr cast covers it). No `uint32_t sem_id` / `sem_addr` donor parameter, no `TensorAccessorArgs<N>` or CTA-offset-NTTP parameter, no old-style addr-gen anywhere.
- **RTA varargs: none.** Every kernel reads each runtime arg at a fixed literal index as a distinct field (`reader_…post_allgather.cpp:59-69`, `reader_…2d.cpp:22-29`, `writer_…:15-17`, one-arg compute kernels). No counted loop, no `arg_index++` run, no data-selected index. **Name every RTA** — do not reach for the vararg mechanism anywhere in this op.
- **`packer_l1_acc` is destructured and then ignored in all five factories** (e.g. `layernorm_pre_all_gather_program_factory.cpp:55-56`; no `ComputeConfigDescriptor` in the op sets it). Preserve that behavior exactly — it is a pre-existing quirk on the ops team's list (`METAL2_PREPORT_AUDIT.md` *Misc anomalies* #7), and "fixing" it during the port would be a functional change.
