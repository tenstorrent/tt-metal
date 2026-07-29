# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_mean`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓\* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

> **\* One confirmation outstanding before you start.** The readiness sheet could not be fetched during the audit (the Google Drive connector cannot be authorized from a non-interactive session). Every TTNN-gate conjunct that is checkable in code was verified clean, but two sheet-only cells were not read, for each of the three factory rows:
>
> - **`Is safe to port?`** — a gate conjunct. Expected `yes`; the smuggled-pointer failure mode it usually catches is structurally absent here (the op never calls `->address()`; every buffer pointer rides the framework-patched `Buffer*` binding). Confirm before porting.
> - **`TensorParameter relaxation`** — expected `none` (the op has no custom hash). If it reads anything else, this brief is missing a Construct item and the custom-hash gate is in play — stop and re-audit.
>
> Nothing else in this brief depends on those cells.

**Recipe docs:** `5fcf2963d45 2026-07-29 docs(metal_2.0): follow main's MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename` *(carry this line into the port report's Provenance section)*

---

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all three factories. Each is a `static ProgramDescriptor create_descriptor(...)` on `MorehMeanOperation` (`device/moreh_mean_device_operation.hpp:34-53`), selected by reduced-dim position in `select_program_factory` (`device/moreh_mean_device_operation.cpp:34-47`). Exactly one runs per invocation.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (no op-owned tensors), for all three factories.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All verified directly in code — the nanobind surface is one `bind_function<"moreh_mean">` over `&ttnn::moreh_mean` (`moreh_mean_nanobind.cpp:19-31`), exposing no internals.

**Three factories, one shared writer.** `MorehMeanHFactory` and `MorehMeanWFactory` both instantiate `kernels/writer_moreh_mean_unary_interleaved_start_id.cpp`; `MorehMeanNCFactory` has its own `kernels/writer_moreh_mean_nc.cpp`. The op owns all 8 kernel files — see *Watch for*.

**Per-factory compute kernels come in pairs over disjoint core groups.** Each factory pushes its compute kernel source into two `KernelDescriptor`s, for `core_group_1` and `core_group_2`, differing only in the `units_per_core_group_N` CTA. These are **disjoint node sets** — each node hosts one compute instance — so this is the ordinary per-group split, *not* a dual-instance work-split. No 1P+1C assignment question arises from it.

---

## Construct — to do

### Tensor bindings

All six are **Case 1** (via `TensorAccessor`) — mechanical, no raw-pointer bridge anywhere. Today the base is delivered by pushing a `Buffer*` into `emplace_runtime_args` at RTA index 0, and the accessor args ride the CTA list via `TensorAccessorArgs(*buf).append_to(...)`. Express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`, and **both** the index-0 `Buffer*` RTA and the `TensorAccessorArgs` CTA plumbing disappear.

| Factory | Binding | Host: `Buffer*` RTA + accessor-args CTA | Kernel: base read + accessor |
|---|---|---|---|
| H | `input` | `..._h_program_factory.cpp:214` · CTA `:119` | `reader_moreh_mean_h.cpp:12` → `:28,46` |
| H | `output` | `..._h_program_factory.cpp:216` · CTA `:137` | `writer_moreh_mean_unary_interleaved_start_id.cpp:11` → `:19,20` |
| W | `input` | `..._w_program_factory.cpp:219` · CTA `:123` | `reader_moreh_mean_w.cpp:12` → `:16,34` |
| W | `output` | `..._w_program_factory.cpp:221-222` · CTA `:141` | `writer_moreh_mean_unary_interleaved_start_id.cpp:11` → `:19,20` |
| NC | `input` | `..._nc_program_factory.cpp:191-199` · CTA `:116` | `reader_moreh_mean_nc.cpp:13` → `:38,39` |
| NC | `output` | `..._nc_program_factory.cpp:201` · CTA `:127` | `writer_moreh_mean_nc.cpp:13` → `:20,21` |

Two CTA-offset details that shift when the accessor args come off the CTA list:

- `reader_moreh_mean_h.cpp:28,32` — `TensorAccessorArgs<3>()` sits *between* `{Ht, Wt, HtWt}` and a trailing `origin_H`, which the kernel reads as `reduce_factor` through `src_args.next_compile_time_args_offset()`. Removing the accessor args makes that trailing CTA a plain named arg.
- `reader_moreh_mean_w.cpp:16,17` — same shape: `TensorAccessorArgs<0>()` first, then the packed `scaler` read via `next_compile_time_args_offset()`.

Both writers read a `cb_id_out` CTA that becomes a DFB binding: `writer_moreh_mean_unary_interleaved_start_id.cpp:15` takes it as `get_compile_time_arg_val(0)` (pushed at `..._h_program_factory.cpp:136` / `..._w_program_factory.cpp:140`), while `writer_moreh_mean_nc.cpp:17` hardcodes `16`. Both become `dfb::out`.

**TensorParameter relaxation:** none *(pending the sheet confirmation above)*.

**TensorAccessor 3rd arg:** none — all five accessor sites pass exactly two arguments. Nothing to drop.

### CB endpoints

Sixteen CBs across three factories. Ten are plain 1:1 (bind producer + consumer as they read); six need a **self-loop** — bind the single touching kernel as *both* PRODUCER and CONSUMER. No multi-binding advanced option anywhere, and no dead CB to drop.

**Self-loop — all configs** (the compute kernel both fills and drains these; nothing else touches them):

- `H / c_24` accum_dst — P `moreh_mean_h.cpp:54` (reduce output) + C `:84,92` (`Accumulate::at`)
- `H / c_25` masked_input — P `moreh_mean_h.cpp:72,76` + C `:81` (reduce input)
- `W / c_24` accum_dst — P `moreh_mean_w.cpp:67,71` + C `:101,122`
- `W / c_25` masked_input — P `moreh_mean_w.cpp:88,92` + C `:99,120`
- `NC / c_24` intermed0 — P `moreh_mean_nc.cpp:58,62` + C `:45,54,69,79`

**Self-loop — unmasked config only; plain 1:1 when masked:**

- `H / c_3` mask_h — masked: reader produces (`reader_moreh_mean_h.cpp:41-44`, under `DO_MASK_H`) + compute consumes (`moreh_mean_h.cpp:42,98`) → 1P+1C. Unmasked: the reader never binds it and compute's FIFO calls are compiled out, but compute still *constructs* the DFB at `moreh_mean_h.cpp:26` → one role-free toucher → self-loop.
- `W / c_3` mask_w — same shape: `reader_moreh_mean_w.cpp:25-27` under `DO_MASK_W`; compute binds at `moreh_mean_w.cpp:25`, FIFO at `:43,128`.

> **Do not read the unmasked config as a dead CB.** `c_3` (and `H / c_25`) are genuinely live under masking — the reader fills `c_3` whenever `origin_H % 32 != 0` / `origin_W % 32 != 0`. Dropping the allocation because one config never FIFO-touches it would break the other. Bind them; self-loop is the correct disposition.

**Plain 1:1 — bind as they read, no action beyond the usual:** `H/c_0`, `H/c_2`, `H/c_16` · `W/c_0`, `W/c_2`, `W/c_16` · `NC/c_0`, `NC/c_1`, `NC/c_2`, `NC/c_16`.

### Kernel-side, mechanical

- **`get_tile_size(cb_id)` → `dfb.get_tile_size()`** (whitelist rule 7). Five sites: `reader_moreh_mean_h.cpp:50` · `reader_moreh_mean_w.cpp:38` · `reader_moreh_mean_nc.cpp:43` · `writer_moreh_mean_nc.cpp:25` · `writer_moreh_mean_unary_interleaved_start_id.cpp:24`. `DataflowBuffer::get_tile_size()` is at `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167`.
- **`DataflowBuffer(tt::CBIndex::c_N)` → `DataflowBuffer(dfb::name)`.** The kernels are already on the `DataflowBuffer` object using its low-level `uint16_t` constructor; you are swapping the *argument*, not the type. The binding-token constructor is `DataflowBuffer(DFBAccessor)` at `dataflow_buffer.h:72`. Object, methods, and call sites all stay as they are.
- **RTAs → named args, everywhere.** No varargs in this op (see *Watch for*). Note `reader_moreh_mean_nc.cpp:12-19` uses a running `i++` counter over a **fixed run of 7** reads — name all seven; this is the recipe's explicit non-signal, not a vararg block.
- **Donor calls need no adaptation.** All five donor headers take either `DataflowBuffer` by value or a `uint32_t` dfb-id (NTTP or runtime); `dfb::name` reaches both — implicitly via `DataflowBuffer(DFBAccessor)`, or via `DFBAccessor`'s constexpr `operator uint32_t()` (`dataflow_buffer.h:55`). Pass the tokens straight through. Full per-function table is in the audit's Team-only section.

---

## Watch for

- **CB endpoints (multi-binding):** **none.** The hidden-second-writer hunt is negative for a structural reason worth knowing: this op has **no semaphores at all**, so the semaphore-gated raw co-fill that face describes has no coordination channel available. Every `get_write_ptr()` in play belongs to a donor producer peeking at the buffer it already FIFO-produces (`generate_mm_scaler.hpp:16`, `moreh_common.hpp:110,194`, `reduce_helpers_dataflow.inl:164`) — one toucher, not two. No `fifo_wr_ptr`, no `evil_set_*_ptr`.

- **⚠ Runtime-selected DFB in the W compute kernel — not a token-for-token swap.** `moreh_mean_w.cpp` keeps `cb_input` as a **mutable** variable initialised to `c_0` (`:21,51`) and reassigned to `c_25` (`cb_masked_input`) mid-loop when masking is active (`:95`), constructing a throwaway `DataflowBuffer(cb_input)` at each use (`:57,63,76,78,94,99,120`). Bind **both** DFBs to the compute kernel and keep the variable `uint32_t`-valued so the reassignment still compiles — `DFBAccessor`'s constexpr `operator uint32_t()` is what makes that legal. A blind `tt::CBIndex::c_N` → `dfb::name` substitution will not survive here.

- **⚠ Two compute-kernel CTAs are misnamed — do not infer the arg name from the kernel variable.** The port names arguments after the variable a kernel unpacks them into, and in these two cases that name is wrong:
  - `moreh_mean_h.cpp:16-18` reads CTA(1) into `Wt` and loops `for (wt < Wt)` (`:46`) — but the factory passes `units_per_core_group_1` / `units_per_core_group_2` there (`..._h_program_factory.cpp:164,185`). The compute kernel is never given the real `Wt`.
  - `moreh_mean_w.cpp:16-18` reads CTA(0) into `Ht` and loops `for (ht < Ht)` (`:47`) — but the factory passes `units_per_core_group_N` there (`..._w_program_factory.cpp:167,188`). Here CTA(1) *is* a genuine `Wt`, which makes the mismatch easy to miss.

  Name these after what they carry (`units_per_core`), not after the kernel's local. The kernel comments repeat the same confusion, describing a reduction over `Wt` / `Ht` tiles where the loop is over per-core units. Renaming the *locals* is out of scope; naming the *binding* correctly is not.

- **Cross-op / shared kernels:** **none — no coordination cost.** The op owns all 8 kernel `.cpp` files, and no other op instantiates any of them (the only external hits are in `ttnn/ttnn.egg-info/SOURCES.txt`, a packaging manifest). No `_metal2` fork exists beside any of them and none is needed: the shared-kernel fork convention does not apply, and there is no sunset list. Convert them in place. The 5 donor **headers** are function-call escapes only — no donor file is rewritten.

- **RTA varargs:** **none.** Every kernel reads a fixed set of RTAs as distinct fields — prefer named RTAs throughout.

- **Off-limits while porting:** `ttnn/cpp/ttnn/operations/experimental/quasar/` holds shortcut pre-port copies that carry idioms this recipe forbids (a stale `api/dataflow/circular_buffer.h` include, `cb_*` handle naming). Do not use anything there as a precedent, a naming source, or evidence that a construct ports.
