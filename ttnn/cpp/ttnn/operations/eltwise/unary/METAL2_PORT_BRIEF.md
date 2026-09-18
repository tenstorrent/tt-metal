# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/eltwise/unary`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (no site)

**Recipe docs:** *not pinnable* — the `metal_2.0/` doc tree is staged-but-uncommitted in this checkout (`git log -1 -- .../metal_2.0/` prints nothing). Working tree: branch `anasuya/metal2_port_unary`, HEAD `6e4a9a9b588 2026-09-17`. *(Carry this line into the port report's Provenance section.)*

**One open question before you start.** The readiness sheet could not be fetched during the audit (non-interactive session, Drive connector unauthenticated); `Is able to port? = yes` and `TensorParameter relaxation = dynamic` were supplied by the user, and the `Known op issues` cell is unverified. Confirm that cell is empty before committing to the port — see `METAL2_PREPORT_AUDIT.md` → *Questions for the user* #1.

---

## The shape of this op, in one paragraph

One `DeviceOperation` (`UnaryDeviceOperation`), one `ProgramFactory`, three kernels: `reader_unary.cpp`, `writer_unary.cpp`, and one of nine compute kernels chosen by `get_compute_kernel_path(op_chain[0].type(), input.dtype())`. Three CBs: `c_0` input, `c_2` output, and `c_1` a LOGIT-only scratch. Behaviour forks on three compile-time-fixed axes — `SRC_SHARDED`/`DST_SHARDED` (always equal), `RM_INTERLEAVED`, and whether `op_chain[0]` is LOGIT — none of which can flip on a cache hit. Everything below is organised against those axes.

---

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` — `ProgramFactory::create_descriptor` returns a `ProgramDescriptor` (`device/unary_device_operation.hpp:44`, `device/unary_program_factory.cpp:337`).
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`**, because `Override runtime args method? == yes`. Translate `ProgramFactory::override_runtime_arguments` (`device/unary_program_factory.cpp:570-666`, declared `device/unary_device_operation.hpp:51`) into one returning a `ProgramRunArgs`, per the port recipe's *Translating `override_runtime_arguments`* step. This method owns the **entire** cache-hit refresh today and must continue to — the adapter deliberately bypasses both `resolve_bindings` and `get_dynamic_runtime_args` when it is present (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:451-457`).
- **Custom hash:** present — `compute_program_hash` (`device/unary_device_operation.cpp:179`) and the backdoor `operation_attributes_t::to_hash()` (`:16`). **Leave both exactly as they are.** Read the comment block at `:197-213` before touching anything nearby: it was written specifically about the Metal 2.0 relaxation contract, and the `tensor_layout()` terms it explains are what make the relaxation below legal.
- **Pybind `create_descriptor`:** none — no user-visible API change in this port.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. A custom hash and an `override_runtime_arguments` are **not** in this list — neither gates, and both are present here.

---

## Construct — to do

### Tensor bindings

Two `TensorParameter`s, `src` (input) and `dst` (output). Each classifies differently by config, and **both halves are real** — do not flatten them:

- **`src` — Case 1** *(configs: interleaved TILE, interleaved ROW_MAJOR)*. The base arrives in reader RTA slot 0 (as a `Buffer*` on the miss path, `device/unary_program_factory.cpp:531,536,555`; as a raw `uint32_t` on the hit path, `:613`) and is fed straight into `TensorAccessor(src_args, src_addr)` at `device/kernels/dataflow/reader_unary.cpp:26`. → Express as a `TensorParameter`/`TensorBinding`; the kernel builds `TensorAccessor(tensor::src)`. RTA slot 0 **and** the host-side `TensorAccessorArgs` plumbing at `:462-463` both disappear.

- **`src` — clean** *(config: sharded)*. `c_0` is borrowed memory backed by `src_buffer` (`device/unary_program_factory.cpp:428`), and the accessor is compiled out by `#if SRC_SHARDED` (`reader_unary.cpp:20`). The DFB **is** the tensor access. → `DataflowBufferSpec::borrowed_from` the `src` `TensorParameter`. No Case-1/Case-2 work here.

- **`dst` — Case 1** *(interleaved TILE, interleaved ROW_MAJOR)*. Symmetric: `Buffer*` on miss (`:532,546,557`), raw on hit (`:616`), consumed via `TensorAccessor(dst_args, dst_addr)` at `device/kernels/dataflow/writer_unary.cpp:28`. → `TensorAccessor(tensor::dst)`; drop the plumbing at `:481-482`.

- **`dst` — clean** *(sharded)*. `c_2` borrowed from `dst_buffer` (`:452`). → `borrowed_from`.

Declare both `TensorParameter`s unconditionally; the `borrowed_from` linkage is what varies with config.

### TensorParameter relaxation

Source: `analyses/relaxations/eltwise_unary.md`. **Verdict `dynamic` — CONFIRMED.** The audit re-ran all five of that document's validity checks and all five now pass; check 1, which was failing when the analysis was written, has since been fixed in the op's code by exactly the route the doc prescribed (`device/unary_device_operation.cpp:217-218` hashes `tensor_layout()` for both slots). Do not act on the doc's §1 "report UNCONFIRMED" paragraph — it is stale.

Apply to **both** `TensorParameter`s, unconditionally:

```cpp
.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
```

- **`dynamic_tensor_shape` is mandatory, not optional.** The TILE-path cache key omits `padded_shape` entirely, so one entry legitimately serves many shapes. Without the flag the *first* cache hit at a different shape throws.
- **`relax_logical_rank` is required for the same reason** — the TILE key omits rank along with the rest of the shape. Hashing `tensor_layout` does not cover this: rank lives in `logical_shape`, on the other side of the split.
- **Do not set `match_page_size.`** Unary would arguably be *entitled* to it (it hashes `padded_shape` on the ROW_MAJOR branch, pinning the last-dim width), but it is declined on precedent grounds — no shipped factory in the tree sets it, and unary should not introduce an untravelled flag while also being the first shipped op to declare any relaxation at all.
- **Do not set `match_padded_shape_only.`** Strictly weaker than `dynamic_tensor_shape`; pins nothing this op needs.

**Expect thin ice, and read a throw carefully.** At the time of analysis, **zero** non-experimental shipped factories declared any `TensorSpecRelaxations`, and `relax_logical_rank` has no shipped precedent at all. The only in-tree factory precedent is `experimental/quasar/transpose`, where five factories set `.relaxations = {.dynamic_tensor_shape = true}` on both slots. If validation throws during the port, treat a framework-side gap as a live hypothesis rather than assuming the declaration is wrong.

> ### One stop condition — stop and ask
>
> If you are porting a configuration where the **input tensor is sharded but the op took the interleaved code path**, stop and raise it. That is the one regime where the accessor is live *over a sharded buffer*, so the relaxation's distribution-geometry term does real work instead of pinning dead code, and the analysis rates it **Low confidence**.
>
> It is plainly reachable, not theoretical. `get_shard_specs` returns `nullopt` on three documented fallbacks (`device/../common/unary_utils.cpp:66,73-91`): `is_native_L1_sharding` failing (DRAM buffer, mismatched in/out grids, or an uneven input), an uneven *output*, and a ROW_MAJOR shard whose element count is not tile-aligned — that last one even emits a `log_warning`. In that state `has_sharding` is false, `SRC_SHARDED` is `0`, and the accessor runs over sharded storage.

### TensorAccessor 3rd arg

**None** — no accessor in this op passes a third argument. Nothing to drop.

*(Not to be confused with:* the host builds both accessors' args with `tensor_accessor::ArgConfig::RuntimeTensorShape` at `:462` and `:481`. That is the legacy runtime-shape arg config — the very thing `dynamic_tensor_shape` replaces — not a page-size override.*)*

### CB endpoints

Full census in the audit. Dispositions:

- **`c_0` (input) — plain 1:1 in every config.** Bind reader PRODUCER, compute CONSUMER. Under **sharded**, additionally `borrowed_from` the `src` `TensorParameter`.
- **`c_2` (output) — plain 1:1 in every config.** Bind compute PRODUCER, writer CONSUMER. Under **sharded**, additionally `borrowed_from` the `dst` `TensorParameter`.
- **`c_1` (tmp0) — self-loop, LOGIT config only.** `logit_kernel.cpp` is its *only* toucher: it both packs into it (`:41-45`) and copies out of it (`:49-56`). Bind that one compute kernel **PRODUCER and CONSUMER**. Legal on Gen1 for a compute kernel; the kernel code is untouched and runtime behaviour is identical. The kernel's own comment at `:31-32` explains why it interleaves produce and consume one tile at a time — do not restructure that loop.
- **No multi-binding flag anywhere.** Both hunts were run and came back empty: there is no raw-pointer CB access (`get_write_ptr` / `get_read_ptr` / `fifo_*_ptr`) anywhere in scope, the op uses **no semaphores at all**, and there is no dual-instance work-split (all three `KernelDescriptor`s carry distinct `kernel_source`s).
- **No dead CB, and no conditional-DFB retrofit.** `c_1`'s conditional already exists host-side (`device/unary_program_factory.cpp:431`, gated by the same predicate that selects `logit_kernel.cpp`), so you mirror an existing conditional rather than inventing one.

### Runtime args → named args

**No varargs mechanism needed** — every RTA and CRTA is nameable. Suggested names, taken from the kernels' own locals:

| Kernel | Slot | Name |
|---|---|---|
| reader | 0 | *(disappears — becomes the `tensor::src` binding)* |
| reader | 1 | `num_pages` |
| reader | 2 | `start_id` |
| reader | 3–7 | `chunks_per_row`, `chunk_size`, `last_chunk_size`, `rows_per_tile`, `total_rows` |
| writer | 0 | *(disappears — `tensor::dst`)* |
| writer | 1–7 | same set, with the output-side chunk sizes |
| compute | 0–2 | `num_tiles`, `packed_scalar1`, `packed_scalar2` |

Two notes on the tail slots. Slots 3–7 are read only under `#if RM_INTERLEAVED`, and the legacy factory writes them as literal zeros otherwise (`:555-557`) so that `override_runtime_arguments` can rewrite every slot a core might need when the split flips a core between active and no-op (see its comment at `:590-591`). Preserve that guarantee — whatever schema you choose, a core that flips must not retain stale args. Likewise compute slots 1–2 are read only by `logit_kernel.cpp`, `where_tss_kernel.cpp` and `mac_tss_kernel.cpp`.

The **common** runtime args are the `TensorAccessorArgs` payload only (`:472`, `:491`). The framework auto-builds these from the binding in Metal 2.0, so they disappear — and so does their bespoke cache-hit refresh at `:638-653`.

### Two legacy blocks that simply go away

- `apply_descriptor_runtime_args(program, cb_addr_only)` (`:657-665`) exists only to re-point the two tensor-backed CBs on a cache hit, and its comment at `:655-656` describes a positional-CB-matching hazard. `borrowed_from` makes the framework own this; delete the block and the hazard with it.
- The accessor common-arg refresh loops (`:638-653`) go the same way. Note they are bounded by `i < common_args.size() && i < reader_common.size()`, which would silently truncate rather than throw — one more reason not to try to preserve them.

---

## Watch for

- **Cross-op / shared kernels — one file, and it is a compute kernel.** `device/kernels/compute/eltwise_sfpu.cpp` is file-path-instantiated by **three external C++ factories** and two tests:

  - `ttnn/cpp/ttnn/operations/examples/example/device/single_core_program_factory.cpp:91`
  - `ttnn/cpp/ttnn/operations/examples/example/device/multi_core_program_factory.cpp:89`
  - `ttnn/cpp/ttnn/operations/examples/example_multiple_return/device/single_core_program_factory.cpp:80`
  - `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:246`
  - `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1436`

  **No `_metal2` fork exists beside it** — this port creates the first one, per *Caution: Porting a shared kernel*. That list is a **sunset list, not authorization to convert the kernel in place.**

  Everything else is unary-exclusive and converts in place with no fork: `reader_unary.cpp`, `writer_unary.cpp`, and the other eight compute kernels (`eltwise_identity_kernel`, `hardswish_kernel`, `lgamma_kernel`, `lgamma_fast_kernel`, `logit_kernel`, `logsigmoid_kernel`, `mac_tss_kernel`, `where_tss_kernel`).

- **A `_metal2` name trap in the directory you are editing.** `device/kernels/dataflow/` contains `reader_unary_interleaved_start_id_metal2.cpp`, `reader_unary_sharded_metal2.cpp` and `writer_unary_interleaved_start_id_metal2.cpp`. **None of these forks `reader_unary.cpp` or `writer_unary.cpp`.** They fork the *other*, similarly-named kernels in the same directory — kernels this op does not use at all, which exist here only because other families (`untilize`, `tilize`, `transpose`, `copy`, …) borrow them. The fork test is per-stem, not per-directory.

- **…but do read them — they are the best local precedent you have, and they are not quasar code.** All three are checked-in, shipped, non-experimental Metal 2.0 kernels sitting in the very directory you are working in, and they answer most of the idiom questions directly:
  - `dfb::in` / `dfb::out` and `tensor::src` / `tensor::dst` binding tokens;
  - `get_arg(args::name)` named args via `experimental/kernel_args.h`;
  - `TensorAccessor(tensor::src)` construction;
  - the fork-note comment convention to copy when you create `eltwise_sfpu_metal2.cpp`.

  `writer_unary_interleaved_start_id_metal2.cpp:13-19` also records a live duplication problem (issue #52228 — a second fork of the same kernel living in `copy/typecast/`); read it before adding another fork to the tree.

  There is **no** `unary` directory under `experimental/quasar/`, so there is no quasar copy of this op to be tempted by.

- **Device 2.0 → Metal 2.0 breadcrumb: confirm, don't swap blind.** `reader_unary.cpp:57` and `writer_unary.cpp:60` read the page size via `get_local_cb_interface(cb_id).fifo_page_size`. That is a *sanctioned* Device 2.0 idiom (it did not fail the gate), and moving it onto the DFB object is **port-stage** work under kernel-side whitelist rule 7. The established in-tree equivalent is `dfb.get_entry_size()` — declared at `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:113`, and used for exactly this purpose by both neighbouring forks (`reader_unary_interleaved_start_id_metal2.cpp:53`, `writer_unary_interleaved_start_id_metal2.cpp:37`), each commented as working for both TILE and ROW_MAJOR.

- **Compile the compute kernels early — the CB ids travel in NTTP position, inside a struct.** Seven of the nine compute kernels bind CBs through `compute_kernel_lib`, where the id sits inside an `InputSpec`/`OutputSpec` used as a *non-type template parameter*: `ckl::CopyTile<ckl::input(dfb_input_id, ckl::WaitPolicy::PerTile, …), ckl::Dst::D0>{}`. This should work — `ckl::input`/`ckl::output` are `constexpr` and take `uint32_t cb_id` (`ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:356,383`), and `DFBBindingToken::operator uint32_t()` is `constexpr` (`tt_metal/hw/inc/api/dataflow/dfb_binding_token.h:36`) — but it is a deeper nesting than the plain `template<uint32_t cb>` case, so get one of these kernels through the compiler before converting the other six. `eltwise_sfpu.cpp` and `mac_tss_kernel.cpp` are the easy pair: they use `DataflowBuffer` objects directly and pass raw ids only to Gen1 compute LLKs (`copy_tile`, `pack_tile`, `compute_kernel_hw_startup`).

- **The CB-id declarations are already `constexpr`** — `constexpr auto cb_id_src = tt::CBIndex::c_0;` (`reader_unary.cpp:15`), `cb_id_dst = tt::CBIndex::c_2;` (`writer_unary.cpp:81`), and the `dfb_input_id` / `dfb_output_id` / `dfb_tmp0_id` declarations in every compute kernel. These are the sites that become `dfb::` tokens, and the `constexpr` spelling is what keeps the NTTP uses above valid. Don't demote any of them to `const`.

- **RTA varargs:** none.

- **Do not fix the non-32×32 tile bug, even though you will see it.** `create_descriptor` sizes both CBs with `tile_size(cb_data_format)` (`device/unary_program_factory.cpp:356,358`), which assumes 32×32, while `enumerate_core_rt_args` reads the tensor's *real* tile for the work split (`:167-169`). The two disagree, and it is a confirmed live bug (`ttnn.relu` on a 16×32-tile bf16 tensor returns wrong data — verified on silicon in `analyses/relaxations/eltwise_unary.md` §4). It is **out of scope**: it is family-wide, `copy/typecast` already shipped a Metal 2.0 port carrying the identical defect, and fixing it here would change behaviour the port's sentinels are supposed to hold fixed. It is recorded and routed to the eltwise team in the audit's *Misc anomalies*. Leave it alone.

- **Several RTA and CTA slots in this op are dead, and most of them are dead on purpose.** The uniform 8-slot reader/writer layout and the always-present compute scalar slots exist so the cache-hit override can rewrite every slot a flipped core might hold. One is *not* purposeful: `:506` appends `cb_data_format` to every compute kernel's CTA list and no in-scope compute kernel reads it. All are itemised in the audit's *Misc anomalies* (#1–#4) and route to the ops team — **none is yours to remove**, since dropping an arg is a functional change.
