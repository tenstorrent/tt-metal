# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ ·
TensorAccessor 3rd arg ✓

**Recipe docs:** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`
*(carry this line into the port report's Provenance section)*

**Shape of this port.** Small and mechanical, with **two must-fix exceptions** that are not mechanical and
will stop the build if you miss them — both are CBs with zero endpoints in at least one config, which a
Metal 2.0 spec validator rejects outright. They are the first two items under *Construct*. Everything else
is the standard binding-layer swap: the kernels are already on `DataflowBuffer` / `Noc` / `CoreLocalMem` /
`TensorAccessor`, so you are changing bindings and arg names, not rewriting idioms.

**The three configs**, since several items below are config-scoped:

| Config | Meaning |
|---|---|
| **A** | small algorithm, no weight (`use_large_algorithm == false`, `weight_has_value == false`) |
| **B** | small algorithm, with weight |
| **C** | large algorithm, with weight |

`use_large_algorithm` implies `weight_has_value`, so *large without weight* does not exist — the audit
proves it from the `cb_usage` arithmetic. All file references below are relative to this op's directory;
`..._program_factory.cpp` is `device/moreh_nll_loss_step1_program_factory.cpp`.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `Factory::create_descriptor` returns `tt::tt_metal::ProgramDescriptor`
  (`device/moreh_nll_loss_step1_device_operation.hpp:34`; body at `..._program_factory.cpp:17-226`).
- **Factory set:** exactly one factory — `using program_factory_t = std::variant<Factory>`
  (`..._device_operation.hpp:40`). No per-factory divergence anywhere in this brief.
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** — the plain one. There is no
  `override_runtime_arguments` to translate, so the framework refreshes the tensor bindings on a cache hit
  and you write one method.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none`
  `TensorParameter relaxation` (the cell reads `none`) · `get_dynamic_runtime_args` (deprecated hook).
- **Also confirmed absent, none of which gate** — noted so you don't go looking: no custom
  `compute_program_hash` (nor a backdoor `attribute_values` / `to_hash`), no `override_runtime_arguments`,
  and **no pybound `create_descriptor`**. `moreh_nll_loss_nanobind.cpp` binds only the user-facing
  `ttnn::moreh_nll_loss`, so this port removes **no** user-visible Python API and the port report needs no
  entry for one.

## Construct — to do

### 1. Drop the dead CB `c_24` — and leave the arithmetic that reads its size alone

`c_24` (the "intermed" CB, allocated at `..._program_factory.cpp:104-112`) is referenced by **no kernel in
any config**. It cannot be carried into Metal 2.0 at all: a DFB with neither a producer nor a consumer
binding is rejected by the spec validator. A dead CB has no behavior, so removing its allocation changes L1
footprint and nothing else. No dead CTA carries its index, so there is nothing further to remove.

*(Why it exists: `c_24` is the conventional intermediate index for a compute kernel, and this op instantiates
none — both readers do the work in L1 with a scalar loop. It reads as a leftover.)*

> **This is the highest-risk line in the port.** `c_24`'s **size** is *not* dead. It feeds `cb_usage`
> (`..._program_factory.cpp:67-68`), which decides `use_large_algorithm` (`:70`), which selects **which
> reader kernel file gets compiled** (`:158-162`). If you drop the CB and also tidy away the
> `(intermed_num_tile * intermed_tile_size)` term from `cb_usage`, you move the small/large threshold and
> change which kernel runs for some shapes — a functional change, out of scope.
>
> **Drop the allocation. Leave `cb_usage` byte-for-byte, dead term and all.** Keep whatever
> `intermed_data_format` / `intermed_tile_size` locals (`:54`, `:58`) that term needs, even though the CB they
> were named for is gone. A comment noting *why* a size is computed for a buffer that no longer exists will
> save the next reader.

### 2. Make `c_7`'s DFB spec conditional — do **not** drop it

`c_7` (weight scratch, `..._program_factory.cpp:129-137`) is **dead under config C, live under config B**.
Both configs named deliberately: "dead CB" plus a drop instruction is exactly how a live buffer gets deleted.

The cause is a guard that is too loose. The allocation is gated on `weight_has_value` alone (`:125`), but the
only kernel that touches `c_7` is the **small** reader — via the donor `read_line`, which takes it as its
`cb_scratch` parameter (`reader_moreh_nll_loss_step1.cpp:57-58`;
`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:782-784`, `:797`). The **large** reader never names `c_7`
(its CB constants are `c_0`, `c_1`, `c_16` at `reader_moreh_nll_loss_step1_large.cpp:23-26`) and calls only
`read_tile` / `read_value`, neither of which takes a scratch buffer. So config C allocates it and nothing
binds it → validator reject.

**Tighten the condition to `weight_has_value && !use_large_algorithm`.** `use_large_algorithm` is already in
scope at the allocation site (computed at `:70`). Expect this to be **new structure** — the legacy factory
allocates unconditionally-ish and gates the *use* by which kernel file it compiles, so there is no existing
host-side conditional to translate.

Safe by inspection, unlike item 1: `c_7` does **not** appear in the `cb_usage` sum (`:67-68`), so tightening
its guard cannot feed back into algorithm selection.

### 3. Tensor bindings (per binding)

All three are **Case 1** — the kernel feeds the base into a `TensorAccessor` and does all access through it.
The mechanical swap applies to each: express as a `TensorParameter` / `TensorBinding`, the kernel builds
`TensorAccessor(tensor::name)`, and the legacy address-via-RTA plus its `TensorAccessorArgs` plumbing both
disappear.

- **`target`** — **Case 1**. Reader RTA idx 0 (`..._program_factory.cpp:206`) → `TensorAccessor(target_args,
  target_addr)` (`reader_...step1.cpp:34`, `..._large.cpp:34`), consumed via `read_tile(dfb, addrg_target,
  page)`.
- **`weight`** (optional) — **Case 1**. Reader RTA idx 1 (`:207`) → `TensorAccessor(weight_args,
  weight_addr)` under `#if defined(WEIGHT)`, consumed via `read_line` (small, `:58`) or `read_value` (large,
  `:79`). **See the optional-binding warning under *Watch for* — this is the one binding with a wrinkle.**
- **`output`** — **Case 1**. Writer RTA idx 0 (`:217`) → `TensorAccessor(output_args, output_addr)`
  (`writer_...step1.cpp:19`), written via `noc.async_write(dfb_out, output_addrg, …)` (`:30`).

Note on what you are replacing: the factory currently passes `Buffer*` handles (not `->address()`) through
`emplace_runtime_args(core, initializer_list<variant<uint32_t, Buffer*>>)` — the framework auto-registers
each as a `BufferBinding` and patches it on cache hits. So this op is **already correct on cache hits
today**; it is on the framework's interim fix, and your typed binding supersedes it. Routine port work, not a
correctness rescue.

**TensorParameter relaxation:** none — the only value that reaches a brief.

**TensorAccessor 3rd arg:** none — no accessor in this op passes a 3rd argument. Nothing to drop, no
`dynamic_tensor_shape` to set.

### 4. CB endpoints

| CB | Config | Disposition |
|---|---|---|
| `c_0` target | A, B, C | **self-loop** — one toucher (the reader, both roles). Bind the reader PRODUCER **and** CONSUMER. |
| `c_1` weight | B, C | **self-loop** — one toucher (the reader). *Not allocated in config A* — the `if (weight_cb_tiles > 0)` at `:90` skips it, and the reader's `c_1` references are all inside `#if defined(WEIGHT)`, so nothing dangles. |
| `c_7` weight scratch | B | **self-loop** — one toucher, entirely sync-free (raw `async_read` destination + `get_write_ptr()`, no FIFO ops). The producer/consumer label is cosmetic on Gen1. |
| `c_7` weight scratch | C | **conditional DFB** — see item 2. Not a drop. |
| `c_16` output | A, B, C | **legal 1:1 — no action.** Reader FIFO-produces (`reserve_back` `:69` / `push_back` `:95`), writer FIFO-consumes (`wait_front` `:29` / `pop_front` `:32`). Bind reader PRODUCER, writer CONSUMER. |
| `c_24` intermed | A, B, C | **dead-CB drop** — see item 1. |

**No CB in this op needs the multi-binding advanced option.** The maximum census on any node is 2 (`c_16`),
and that pair is one locked producer + one locked consumer. Recorded as a positive finding, not an untested
assumption: the audit hunted all three multi-toucher faces. The hidden-second-writer face in particular
**cannot** apply here — it requires a semaphore-gated raw co-fill, and this op has **no semaphores at all**.

One thing not to misread while binding `c_16`: the reader also calls `get_write_ptr()` on it
(`reader_...step1.cpp:72`). That is a peek on its own PRODUCER binding, **not** a third endpoint.

## Watch for

- **The `cb_usage` trap.** Restated here because it is the one way to silently break this port: drop `c_24`'s
  allocation, **not** its contribution to `cb_usage` (`..._program_factory.cpp:67-68`). See *Construct* item 1.
- **The optional `weight` tensor is expressed in three coordinated places, and the middle one is easy to
  lose.** Keep all three consistent:
  1. host — `weight_buf = nullptr` when absent (`..._program_factory.cpp:186`), which the framework turns
     into a literal `0u` with **no** binding registered, deliberately, so optional inputs don't invalidate the
     fast cache-hit path (`tt_metal/impl/program/program_descriptors.cpp:245-250`);
  2. host — `TensorAccessorArgs(nullptr).append_to(...)` still appends a **placeholder args block** for the
     absent weight (`..._program_factory.cpp:143`). This is load-bearing: it is what keeps
     `TensorAccessorArgs<target_args.next_compile_time_args_offset()>()` at a fixed CTA offset in the kernel
     (`reader_...step1.cpp:32`) whether or not a weight exists. Drop it and the offsets shift under config A.
  3. kernel — the `WEIGHT` define (`..._program_factory.cpp:151-153`) compiles the weight accessor and its
     DFB in or out entirely.
- **Cross-op / shared kernels: nothing to coordinate.** The op **owns all three** of its kernel `.cpp` files
  and **no other op instantiates them** (all three `kernel_source` paths at `:158-165` point inside this op's
  own `device/kernels/`). So: no `_metal2` fork to reuse, none to create, **no sunset list**. The single
  out-of-directory dependency is a *header*, not a borrowed kernel file:
  `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`.
- **The donor takes `DataflowBuffer`, which is the easy case — no donor-side change, no fork.** `read_tile`,
  `read_value`, and `read_line` all take `DataflowBuffer` **by value**, and the kernels already pass named DFB
  locals (`dfb_target_obj`, `dfb_weight_obj`, `dfb_weight_scratch_obj`). Construct those locals from the
  tokens — `DataflowBuffer dfb_target_obj(dfb::target);` — and every call site is unchanged. (Prefer the named
  local over passing `dfb::target` straight in, even though the conversion is non-explicit: no in-tree kernel
  does the latter yet.)
- **Two `get_tile_size(cb_id)` breadcrumbs — confirm, don't swap blind.** These are sanctioned Device 2.0
  today, and kernel-side whitelist rule 7 moves them onto the DFB object at port time:
  `writer_moreh_nll_loss_step1.cpp:25` and `reader_moreh_nll_loss_step1_large.cpp:37`. **Leave the donor's own
  internal ones alone** (`moreh_common.hpp:683`, `:709`, `:753`) — that is a shared header, and changing it
  reaches every moreh op.
- **These kernels are already part-modernized.** All three are on `DataflowBuffer` / `Noc` / `CoreLocalMem` /
  `TensorAccessor` with the current `api/dataflow/dataflow_buffer.h` include (not the stale
  `api/dataflow/circular_buffer.h`). Expect to touch bindings and arg names, not control flow.
- **`constexpr` vs `const` on the CB handles.** All CB indices are declared `constexpr uint32_t`
  (`reader_...step1.cpp:23-26`, `..._large.cpp:23-26`, `writer_...step1.cpp:15`) — the form that admits the
  token / constexpr-cast path. Worth a glance before assuming member-getter form.
- **RTA varargs: none.** Every arg is nameable. Both readers read a fixed run of **nine** through a running
  `i++` at the top (`reader_...step1.cpp:12-21`), and the writer reads three at constant indices
  (`writer_...step1.cpp:11-13`) — a sequential counter over a fixed set, not a loop, so it dissolves into
  named args. The names come straight off the declarations: `target_addr`, `weight_addr`, `ignore_index`,
  `num_units_per_core`, `start_id`, `C`, `weight_num_tile`, `element_size`, `target_element_size`;
  `output_addr`, `num_units_per_core`, `start_id`. **CTA varargs: none either** — all compile-time reads are
  at constexpr offsets, so no `compile_time_varargs`.
- **Two of those nine RTAs are dead — don't carry them across.** `element_size` (idx 7) and
  `target_element_size` (idx 8) are set by the factory (`..._program_factory.cpp:213-214`) and read by both
  readers (`reader_...step1.cpp:20-21`, `..._large.cpp:20-21`), then **never used**. Naming a dead arg is
  wasted plumbing; dropping them is the same class of change as the dead-CB drop (no behavior to preserve).
  If you would rather keep the port strictly binding-only, name them and note it in the port report — but say
  which you chose either way.
- **A declared-but-unused CTA mirror, so you don't "fix" half of it.** Both readers declare
  `constexpr bool weight_has_value = get_compile_time_arg_val(0) == 1;` (`:30`) and never use it — the weight
  paths are selected by `#if defined(WEIGHT)` instead. The CTA **slot** is *not* dead: it is positional
  padding that keeps `TensorAccessorArgs<1>` at its offset (`:31`). One host boolean, plumbed twice by two
  mechanisms (`..._program_factory.cpp:141` and `:151-153`). Remove one half without the other and the CTA
  offsets shift.
- **No quasar copy of this op exists** (`ttnn/cpp/ttnn/operations/experimental/quasar/` has no `nll` entry),
  so there is no shortcut-port lookalike to be misled by. A negative pointer, to save a wrong turn.
- **One open question the audit raised, worth a glance before you delete anything.** The audit asks the ops
  team to confirm nobody has a planned or reverted compute kernel for `step1` (which is what would make
  `c_24` live). The evidence for the drop is strong and the failure mode is safe — if the CB were somehow
  live, you'd hit a loud bindingless-DFB error, not silent wrong numerics. Proceed; just don't be surprised if
  that question comes back.
