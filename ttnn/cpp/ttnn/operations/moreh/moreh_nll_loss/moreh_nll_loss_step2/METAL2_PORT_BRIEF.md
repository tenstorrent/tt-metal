# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ ·
TensorAccessor 3rd arg ✓

**Recipe docs:** `28c1b0b4224 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`
*(carry this line into the port report's Provenance section)*

## Shape of this port — read before starting

One factory, **three internal rank paths**, **seven kernels**, **eleven CBs**. The kernels are already on
`DataflowBuffer` / `Noc` / `CoreLocalMem` / `TensorAccessor`, so this is a binding-layer change, not an idiom
rewrite. The work is concentrated almost entirely in **CB endpoints**, which carries four must-fix items.

`Factory::create_descriptor` (`device/moreh_nll_loss_step2_program_factory.cpp:701`) is a thin rank dispatcher
over three file-local builders, each producing its own `ProgramDescriptor` with its own kernel set and CB set:

| Path | Builder | Reader | Writer | Selected when |
|---|---|---|---|---|
| **2d** | `moreh_nll_loss_step2_impl_2d` (`:45`) | `reader_..._2d.cpp` | `writer_..._2d.cpp` | `rank == 2` |
| **3d** | `moreh_nll_loss_step2_impl_3d` (`:258`) | `reader_..._3d.cpp` | `writer_..._3d.cpp` | `rank == 3` |
| **4d** | `moreh_nll_loss_step2_impl_4d` (`:471`) | `reader_..._4d.cpp` | `writer_..._4d.cpp` | `rank >= 4` |

The **compute kernel** `moreh_nll_loss_step2_kernel.cpp` is shared by all three paths and instantiated
**twice** per path, once per `split_work_to_cores` core group.

**The config space is 12, and you must build all of it.** Three rank paths × weight present/absent × divisor
present/absent, all reachable. Two of the four must-fix items are **compile failures that only occur when an
optional tensor is absent** — so a port that builds and passes on the default path can still be broken in
eight of twelve configs.

**All file references below are relative to this op's directory**; `..._program_factory.cpp` is
`device/moreh_nll_loss_step2_program_factory.cpp`, and bare `compute:N` means
`device/kernels/moreh_nll_loss_step2_kernel.cpp:N`.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `Factory::create_descriptor` returns `tt::tt_metal::ProgramDescriptor`
  (`device/moreh_nll_loss_step2_device_operation.hpp:35`; body at `..._program_factory.cpp:701-732`).
- **Factory set:** exactly one factory — `using program_factory_t = std::variant<Factory>`
  (`..._device_operation.hpp:41`). **The three rank paths are internal code paths, not factories** — one
  factory concept, one spec factory; don't model them as three.
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** — the plain one. No `override_runtime_arguments` to
  translate, so the framework refreshes the tensor bindings on a cache hit and the factory writes one method.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none`
  `TensorParameter relaxation` (the cell reads `none`) · `get_dynamic_runtime_args` (deprecated hook).
- **Also confirmed absent, none of which gate** — noted so you don't go looking: no custom
  `compute_program_hash` (nor a backdoor `attribute_values` / `to_hash`), no `override_runtime_arguments`, and
  **no pybound `create_descriptor`**. `moreh_nll_loss_nanobind.cpp` binds only the user-facing
  `ttnn::moreh_nll_loss`, so this port removes **no** user-visible Python API and the port report needs no
  entry for one.

## Construct — to do

### 1. ⚠ Guard the compute kernel's `c_3` DFB declaration — or the no-divisor configs won't compile

`compute:15` constructs `DataflowBuffer dfb_divisor_obj(cb_divisor);` **outside** `#if defined(DIVISOR)`,
while every *use* of it is inside (`compute:36-52`, `:94`, `:124`, `:153`). But `c_3` is **not allocated at
all** when divisor is absent — `push_cb(..., CBIndex::c_3, divisor_has_value ? 1 : 0, ...)` returns early on
`num_tiles == 0` (`..._program_factory.cpp:88`, `:299`, `:529`; helper at `:22-41`).

**Move the declaration inside `#if defined(DIVISOR)`.** In Metal 2.0 the DFB is constructed from a
`dfb::divisor` binding token, which does not exist when there is no `DataflowBufferSpec` for `c_3` — so the
six no-divisor configs will fail to compile.

*(Why it has been harmless so far: today the constructor just records the id on the MATH thread; on the
unpack/pack threads it eagerly reads `get_local_cb_interface(3)` to feed a NoC-debug tracker
— `tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl:31-39` — reading a stale interface entry for a CB the
program never created, benign only because that tracker is normally compiled out.)*

**The readers already do this correctly** — 2d `:56`, 3d `:59`, 4d `:60` are all inside the guard. Only the
compute kernel is inconsistent, so fix it there and leave the readers alone.

### 2. ⚠ Keep `c_24`'s spec unconditional and self-loop it when weight is absent — do **not** drop it

`c_24` (tmp_weight) is allocated unconditionally in all three paths (`:90`, `:301`, `:531`) but FIFO-touched
only under `WEIGHT` (reader produces, compute consumes). It looks like a dead CB in the six no-weight configs.
**It is not, and dropping it breaks the build the same way item 1 does:** the compute kernel references its
index unconditionally at `compute:34` (`compute_kernel_hw_startup(cb_tmp_weight, cb_tmp_input, cb_output)`)
and constructs a DFB for it at `compute:18`. The `dfb::tmp_weight` token must resolve in every config.

**Build the spec unconditionally. Under `WEIGHT` it's an ordinary 1P+1C; without `WEIGHT`, self-loop it**
(bind the compute kernel PRODUCER and CONSUMER — cosmetic on Gen1, no runtime effect).

> One honest caveat, so you're not surprised if a reviewer raises it: whether
> `compute_kernel_hw_startup(dfb::tmp_weight, …)` counts as an endpoint *binding* — versus a
> format/hardware-config reference that needs the token but not a binding — is a framework question the audit
> recipe doesn't settle, and it's [Question 1](#) to the framework team in the audit. It changes the *label*
> (1 role-free toucher → self-loop, versus 0 touchers → conditional DFB) but **not** this instruction: either
> way the spec must exist in every config. The guidance above is the side that's safe under both readings.

### 3. Drop the dead `c_7` allocations in `impl_2d` and `impl_3d`

`c_7` (weight scratch) has **zero endpoints in every config** of the 2d and 3d paths:

- **`impl_2d` — drop `..._program_factory.cpp:102`.** The 2d reader never names `c_7` (its CB constants are
  `c_0`, `c_1`, `c_2`, `c_3`, `c_24`, `c_25`, `c_16` at `:25-33`); it reads weight via `read_value`, which
  takes no scratch buffer.
- **`impl_3d` — drop `..._program_factory.cpp:313`.** Same: constant list at `:26-34`, `read_value` again.
- **`impl_4d` — leave `:543` alone.** The 4d reader genuinely uses it (`:32`, `:73`, passed to `read_line` at
  `:74`); see the self-loop list below.

A DFB with neither a producer nor a consumer binding is rejected by the spec validator, so these can't be
carried across; and a dead CB has no behavior, so removing the allocation changes L1 footprint and nothing
else. No dead CTA carries the index (the readers' compile-time args are four `TensorAccessorArgs` blocks and
nothing else — `:107-110`, `:318-321`), so there's nothing further to remove.

> **Don't copy `step1`'s answer here.** `step1` has the same defect shape — a scratch CB allocated on
> `weight_has_value` alone while only one reader variant uses it — but there both variants live behind one host
> flag inside **one** program, so its `c_7` had to become a *conditional* DFB. Here the rank paths are three
> separate builders producing three separate `ProgramDescriptor`s, so in `impl_2d` and `impl_3d` the
> allocation is dead unconditionally → a straight drop, with no conditional to write.

### 4. Delete the four dead CB declarations — do **not** convert them

These are inert dead locals today. At port time they're a trap, because the mechanical conversion of a CB
constant is to a `dfb::name` binding — and **a binding is an endpoint**:

| Dead declaration | If mechanically converted | Consequence |
|---|---|---|
| `cb_output` = `c_16` in **all three readers** (2d `:33`, 3d `:34`, 4d `:37`) | adds a reader binding on `c_16` | per-node census 2 → **3** ⇒ you'd wrongly set the multi-binding flag on the output DFB |
| `cb_weight` = `c_2` in the **compute** kernel (`compute:13`) | adds a compute binding on `c_2` | per-node census 1 → **2** ⇒ turns a clean self-loop into a spurious 1P+1C |

They carry no behavior, so deleting them is zero-functional-change — the same reasoning as a dead-CB drop.

### 5. Delete the nine dead `get_dataformat` locals

Each reader computes three data-format values and uses none: `input_data_format`, `weight_data_format`,
`divisor_data_format` at `reader_..._2d.cpp:36`/`:38`/`:40`, `..._3d.cpp:37`/`:39`/`:41`,
`..._4d.cpp:40`/`:42`/`:44`.

**Delete them rather than converting them to DFB getters** — the values are unused, so there is nothing to
preserve. (Had any been live, kernel-side whitelist **rule 7** would apply: `get_dataformat(cb_id)` is named
there explicitly as metadata that moves onto the DFB object. All nine here are declared `const`, so they'd
take the **member getter** form, not the `constexpr` free-function carve-out.)

*Bonus reason to delete rather than convert:* `divisor_data_format` reads `c_3` **unconditionally**, a CB that
doesn't exist without a divisor — so converting it would add a third no-divisor compile failure alongside
items 1 and 2.

### 6. Tensor bindings (per binding)

All five are **Case 1** — the kernel feeds the base into a `TensorAccessor` and does all access through it.
The mechanical swap applies to each: express as a `TensorParameter` / `TensorBinding`, the kernel builds
`TensorAccessor(tensor::name)`, and the legacy address-via-RTA plus its `TensorAccessorArgs` plumbing both
disappear.

| Binding | Reader/writer RTA idx | Kernel consumption | Case |
|---|---|---|---|
| `input` | reader **0** (`:213`, `:424`, `:654`) | `TensorAccessor(input_args, input_addr)`; `read_value(dfb_input_obj, addrg_input, …)` | **Case 1** |
| `target` | reader **1** (`:214`, `:425`, `:655`) | `TensorAccessor(target_args, target_addr)`; `read_tile` | **Case 1** |
| `weight` (optional) | reader **2** (`:215`, `:426`, `:656`) | `TensorAccessor(weight_args, weight_addr)`; `read_value` (2d/3d) or `read_line` (4d) | **Case 1** |
| `divisor` (optional) | reader **3** (`:216`, `:427`, `:657`) | `TensorAccessor(divisor_args, divisor_addr)`; `read_tile` | **Case 1** |
| `output` | writer **0** (`:228`, `:440`, `:672`) | `TensorAccessor(output_args, output_addr)`; `noc.async_write(dfb_out, output_addrg, …)` | **Case 1** |

Note on what you're replacing: the factory currently passes `Buffer*` handles (never `->address()`) through
`emplace_runtime_args(core, initializer_list<variant<uint32_t, Buffer*>>)` (`:210`, `:225`, `:421`, `:437`,
`:651`, `:669`). The framework auto-registers each as a `BufferBinding` and patches it on cache hits, so this
op is **already correct on cache hits today**; your typed binding supersedes it. Routine port work, not a
correctness rescue.

**The compute kernel needs no tensor binding at all** — it only consumes from and produces to CBs, constructs
no `TensorAccessor`, and reads **no runtime args whatsoever**. So there is no Case-2-in-a-compute-kernel
problem here, which is the one shape that would have blocked this port.

**TensorParameter relaxation:** `none` — nothing to apply.

**TensorAccessor 3rd arg:** none — all 15 accessor constructions in the op are the 2-arg form. Nothing to
drop, no `dynamic_tensor_shape` to set.

### 7. CB endpoints — the remaining dispositions

After items 1-4, the rest is routine. **No CB in this op needs the multi-binding advanced option.**

- **self-loop** (one toucher): `c_0` input (all configs) · `c_1` target (all) · `c_2` weight (weight present) ·
  `c_7` weight scratch (**4d + weight only**) · `c_26` tmp1 (all) · `c_27` divisor_recip (all) ·
  `c_28` tmp3 (all)
- **legal 1:1, no action** (one locked producer + one locked consumer): `c_3` divisor (divisor present —
  reader produces, compute consumes) · `c_16` output (compute produces, writer consumes) · `c_24` tmp_weight
  (weight present — reader produces, compute consumes) · `c_25` tmp_input (all — reader produces, compute
  consumes)
- **multi-binding advanced option:** not needed anywhere.

`c_26`, `c_27`, `c_28` self-loop in *every* config: the compute kernel both produces and consumes them where
they're used, and constructs their DFBs unconditionally (`compute:22`, `:24`, `:26`) where they aren't — so
like `c_24`, their specs must exist in all 12 configs.

> **Three `KernelSpec`s will bind some of these DFBs, and that is legal — don't reach for the flag.** For
> `c_25`, `c_16`, `c_24`, `c_3` the reader/writer spec plus **both** compute specs reference the DFB. Counting
> *bindings* gives three; the census is **per CB, per node**, and because the two compute specs cover
> **disjoint** core groups each node sees exactly one, giving one producer and one consumer → ordinary 1:1.
> The framework validates the non-overlapping coverage. This is explicitly the legal single-role case, **not**
> `allow_instance_multi_binding`.

## Watch for

- **Do not collapse the two compute `KernelDescriptor`s into one spec with a runtime arg.** This op is a
  textbook instance of the **demoting-per-group-CTA anti-pattern**: `split_work_to_cores` plus two same-source
  compute descriptors carrying different per-group CTAs (`:163`/`:179`, `:374`/`:390`, `:604`/`:620`). The
  correct port is **two `KernelSpec`s of the same source in two `WorkUnitSpec`s**, one per core group — Metal
  2.0 supports that. The demotion costs compile-time loop unrolling on `per_core_tile_cnt` (`compute:11`, the
  loop bound at `compute:54`), a measurable kernel-perf regression the port is not entitled to make.
  **This op baits the trap unusually well:** the factory *already* populates a per-core compute RTA carrying
  exactly that value (`:235-243`, `:448-456`, `:678-686`) — and the compute kernel **never reads it** (zero
  `get_arg_val`). If you notice the RTA and "simplify" toward it, you land precisely on the anti-pattern.
  **Delete the dead RTA; don't adopt it.**
- **Build all twelve configs before calling this done.** Three rank paths × `(weight, divisor)`. Items 1 and 2
  fail *only* when an optional tensor is absent — eight of the twelve configs — and the default path won't
  catch either.
- **The optional-accessor placeholder chain is load-bearing, more so than in `step1`.**
  `TensorAccessorArgs(nullptr)` is appended for an absent `weight` / `divisor` (`:109-110`, `:320-321`,
  `:550-551`), which is what keeps the *following* accessors' CTA offsets fixed through the
  `next_compile_time_args_offset()` chain (2d `:42-45`). In `step1` the one optional tensor was last in its
  chain; here `weight` and `divisor` sit at positions 3 and 4 of four, so dropping a placeholder shifts every
  offset after it.
- **`weight`'s accessor is built outside its `WEIGHT` guard, unlike `divisor`'s.** `TensorAccessor(weight_args,
  weight_addr)` at 2d `:49`, 3d `:52`, 4d `:53` sits outside any `#if defined(WEIGHT)`, while every use is
  inside; `divisor`'s accessor is correctly inside its own guard (2d `:54`). Harmless today, but the
  `tensor::weight` binding must resolve in the no-weight configs — so either declare it there or move the
  construction inside the guard. Same class of issue as item 1; decide both the same way.
- **Compute LLK call sites take the rule-2 implicit conversion, not a getter.** `copy_tile`, `mul_tiles`,
  `mul_tiles_bcast_scalar`, `mul_bcast_scalar_init`, `reconfig_data_format`, and `compute_kernel_hw_startup`
  all take `uint32_t cb_id` and have **no** `DataflowBuffer` overload anywhere in
  `tt_metal/hw/inc/api/compute/` — the whole compute API is index-based by design. Pass `dfb::name` directly
  and let the conversion fire; do **not** extract `.get_id()`, and do **not** hunt for a getter that doesn't
  exist. (The donor's `*_with_dt` helpers already take `DataflowBuffer` — those keep taking your named local.)
- **Cross-op / shared kernels: nothing to coordinate.** The op **owns all seven** kernel `.cpp` files and **no
  other op instantiates them** (verified across `ttnn/` — the only hits are this factory's own `kernel_source`
  assignments). So: no `_metal2` fork to reuse, none to create, **no sunset list**. The two out-of-directory
  dependencies are *headers*, not borrowed kernel files:
  `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (readers, 3d/4d writers) and
  `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (compute).
- **Both donors take `DataflowBuffer` — the easy case, no donor-side change, no fork.** The dataflow donor's
  `read_tile` / `read_value` / `read_line` and the compute donor's `copy_tile_init_with_dt` /
  `pack_tile_with_dt` / `mul_tiles_init_with_dt` all take `DataflowBuffer` **by value**, and the kernels
  already pass named DFB locals. Construct those from the tokens
  (`DataflowBuffer dfb_input_obj(dfb::input);`) and every call site is unchanged. (Prefer the named local over
  passing `dfb::input` straight in: the conversion is non-explicit, but no in-tree kernel does that yet.)
- **`get_tile_size(cb_id)` breadcrumbs — confirm, don't swap blind.** Sanctioned today; whitelist rule 7 moves
  them onto the object at port time: `writer_..._2d.cpp:25`, `writer_..._4d.cpp:27`. Both are `const auto`, so
  the member-getter form applies. **Leave the donor's own internal ones alone**
  (`dataflow/moreh_common.hpp:683`, `:709`, `:753`) — that's a shared header, and changing it reaches every
  moreh op.
- **These kernels are already part-modernized.** All seven are on `DataflowBuffer` / `Noc` / `CoreLocalMem` /
  `TensorAccessor` with the current `api/dataflow/dataflow_buffer.h` include (not the stale
  `api/dataflow/circular_buffer.h`). Expect to touch bindings and arg names, not control flow.
- **`constexpr` vs `const` on the CB handles.** All CB indices are declared `constexpr uint32_t` — the form
  that admits the token / constexpr-cast path. The data-format and tile-size locals are `const auto`. Worth a
  glance before assuming a form.
- **Four dead RTAs you can drop while you're renaming args** (all confirmed unread by their kernels): the 2d
  writer's `origin_N` (`:231`; writer reads only indices 0-2 at `:11-13`), the 2d reader's `element_size`
  (`:222`, read at `reader_..._2d.cpp:23`, never used), the 4d reader's `element_size` (`:666`, read at
  `reader_..._4d.cpp:25`, never used), and the whole per-core compute RTA vector (above). The **3d** reader's
  `element_size` *is* used (`:89`) — keep that one. If you'd rather keep the port strictly binding-only, name
  them instead and say which you chose in the port report.
- **No quasar copy of this op exists** (`ttnn/cpp/ttnn/operations/experimental/quasar/` has no `nll` entry),
  so there is no shortcut-port lookalike to be misled by. A negative pointer, to save a wrong turn.
- **`step1`'s brief is not in this checkout.** It was written on branch
  `anasuya/metal2_port_moreh_nll_loss`; this branch is `anasuya/metal2_port_moreh_nll_loss_step2`. If you want
  to compare against the sibling op's port — the two share the dataflow donor header and repeat the `c_7`
  defect shape — recover it from that branch.
