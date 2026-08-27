# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/concat`

> ### ⚠ SUBSET BRIEF — read this box before anything else.
>
> The op is **RED at op level**. This brief covers **exactly three of concat's six program factories**:
>
> - ✅ **`ConcatProgramFactory`** (`device/concat_program_factory.cpp`) — the op's **default** factory:
>   all interleaved-input cases plus the ND-sharded fallback. **This is the bulk of the port.**
> - ✅ **`ConcatS2SRMProgramFactory`** (`device/concat_s2s_rm_program_factory.cpp`)
> - ✅ **`ConcatS2STiledProgramFactory`** (`device/concat_s2s_tiled_program_factory.cpp`)
>
> The other three are **gated and out of scope** — do not touch them, and do not carry anything in this
> brief across to them:
>
> - ⛔ `ConcatS2SMultiProgramFactory` — blocked: `DFB misuse; will need semi-manual port`
> - ⛔ `ConcatBlockShardedProgramFactory` — blocked: `DFB misuse; will need semi-manual port`
> - ⛔ `ConcatS2IProgramFactory` — blocked: dead code (references a kernel source that does not exist)
>
> `ConcatDeviceOperation` keeps all six factories in its `program_factory_t` variant
> (`device/concat_device_operation.hpp:29-35`) and `select_program_factory` keeps dispatching to all six.
> Only the three named above convert, so the variant ends up holding a **mix** of `descriptor` and
> `MetalV2` factories — **this is supported** (confirmed with the TTNN framework owner), so you need do
> nothing special about it. The full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared** (op-wide, not just for the subset): Device 2.0 ✓ · Features ✓ · Offset base pointers ✓ ·
TensorAccessor 3rd arg ✓ (N/A — no site) · TensorParameter relaxation ✓ (`none`) ·
TTNN factory concept ✓ **for these three factories**

**Recipe docs:** `0846547f407 2026-08-27 docs(metal_2.0): a run in flight freezes the kernel sources`
*(carry this line into the port report's Provenance section)*

## One stale readiness cell — don't let it stop you

The live readiness sheet still reads `Is able to port? = no` / `Known op issues = Awaiting variadic
tensor in Metal 2.0` on the **`ConcatProgramFactory`** row. **That support has merged**, and the sheet
owner has confirmed the row is green with the cell cleared; the sheet itself just hasn't been refreshed
yet. So if you re-fetch it and see `no` there, that is the stale cell — **not a new block**, and not a
reason to drop this factory from the port. Every other column on that row is clean and code-confirmed
(cross-check table in `METAL2_PREPORT_AUDIT.md` → Gate detail).

Nothing else is left open. Mixed `descriptor` / `MetalV2` factories on one `program_factory_t` variant
are supported, so the subset scope above is viable as written.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); all three factories port to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — each factory is a `static tt::tt_metal::ProgramDescriptor
  create_descriptor(...)` (`concat_program_factory.hpp:14`, `concat_s2s_rm_program_factory.hpp:14`,
  `concat_s2s_tiled_program_factory.hpp:14`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (the base concept — no `override_runtime_arguments`
  exists anywhere in the op, so this is *not* the custom concept).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none`
  `TensorParameter relaxation` · `get_dynamic_runtime_args`.
- **Also absent, though none of them gates:** a custom `compute_program_hash`, a backdoor
  `attribute_values` / `to_hash`, an `override_runtime_arguments`, and a pybound `create_descriptor`.
  `concat_nanobind.cpp` binds only the public `ttnn::concat` free function — so there is **no pybind
  binding to delete** and no user-visible API change on that axis.

---

# `ConcatProgramFactory` — the main body of work

The only factory in the op with tensor bindings that need real work, the only one with varargs, and the
only one with borrowed donor kernels. Config axes: **`rm_layout`** (RM vs TILE — swaps *both* the reader
and the writer kernel), **`WIDTH_CONCAT`** (set when `rm_layout && dim == rank-1`), **`sub_core_grids`**
present/absent, and CB depth 1 vs 2.

## Construct — to do

### Tensor bindings — `N+1`, all Case 1

`N` is the input tensor count, up to 47 for interleaved (`concat_device_operation.cpp:285`).

| Binding | Legacy delivery | Kernel consumption | Case |
|---|---|---|---|
| `input_0` … `input_{N-1}` | `Buffer*` pushed into the reader RTA list @ `concat_program_factory.cpp:276` | fed to `TensorAccessor` via `make_tensor_accessor_tuple(args, 3)` @ `reader_concat_interleaved_start_id.cpp:36`, `reader_concat_stick_layout_interleaved_start_id.cpp:35` | **Case 1** |
| `output` | `Buffer*` pushed into the writer RTA list @ `concat_program_factory.cpp:285` (RM), `:290` (TILE) | `TensorAccessor(dst_args, dst_addr)` in both donor writers | **Case 1** |

**These are `Buffer*`-form deliveries, not the silent-wrong `->address()` hazard.** The framework
auto-registers them as `BufferBinding`s and patches them on cache hits, so the op is correct today. This
is routine port work, not a correctness fix. (The whole op contains **zero** `->address()` sites.)

**Port work — the `N` inputs.** Declare `N` input `TensorParameter`s at ProgramSpec level and `N`
`TensorBinding`s on the reader `KernelSpec`, then declare **one `TensorBindingSequence`** listing them
in order (`KernelAdvancedOptions::TensorBindingSequence`,
`tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp:147-151`). That is the
variadic mechanism this factory was waiting on, and its documented purpose is exactly this shape: *"a
kernel that wishes to express a compile-time-variadic number of tensor bindings, and therefore needs to
access them positionally."* Concat's `N` is fixed at cache-miss time and already baked into CTAs, so
compile-time-variadic is sufficient.

Kernel side, the translation is close to mechanical — the two helpers the fork-side API names are the
same two the legacy readers already use:

```cpp
// legacy (both readers)
constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
constexpr uint32_t page_size_base_idx = 2;
constexpr auto tensor_accessor_args =
    make_tensor_accessor_args_tuple<num_tensors, page_size_base_idx + num_tensors>();
auto tensor_accessors_tuple = make_tensor_accessor_tuple(tensor_accessor_args, src_addr_base_idx);
auto abstract_tensor_accessor_wrappers = make_abstract_tensor_accessor_wrappers(tensor_accessors_tuple);

// Metal 2.0
auto accessor_tuple = make_tensor_accessors(tensor::inputs);
auto abstract_tensor_accessor_wrappers = make_abstract_tensor_accessor_wrappers(accessor_tuple);
```

Three consequences worth knowing before you write it:

- **The `num_tensors` CTA becomes redundant.** The sequence carries its own length —
  `std::tuple_size_v<decltype(tensor::inputs)>` (`advanced_options.hpp:138-139`). Drop
  `concat_program_factory.cpp:200`'s second element and take the count from the sequence.
- **The per-input `TensorAccessorArgs` CTAs disappear** (`concat_program_factory.cpp:203-205`) — the
  framework auto-builds accessor args from the bindings. So does the writer's (`:218`).
- **The `page_size_per_tensor[N]` CTA block stays** (`:201-202`) — it is *not* accessor args; the RM
  reader genuinely reads it, at a runtime index. See the CTA-vararg item below.

**Port work — the output.** Declare it as a `TensorParameter` and bind it on the writer `KernelSpec`
with `accessor_name = "dst"`. That name is **not yours to choose** — it is fixed by the `_metal2` fork
you will bind (both forks name it `tensor::dst`). Nothing on the kernel side to write; both forks already
do `TensorAccessor(tensor::dst)`.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — both readers and both donor writers construct two-argument accessors.

### CB endpoints — one CB, plain 1:1, no action

| CB | Backing | Producer | Consumer | Disposition |
|---|---|---|---|---|
| `0` (`src0_cb_index`) | L1 staging — **not** borrowed (no `.buffer`) | reader (`reserve_back`/`push_back`/`get_write_ptr`) | donor writer (`wait_front`/`pop_front`) | **plain 1:1 — no action** |

Two touchers, one locked to each FIFO role. Holds across all four config axes: RM vs TILE swaps *which*
donor writer is bound, but both are locked consumers; `WIDTH_CONCAT`, `sub_core_grids` and CB depth
change nothing. No self-loop, no 1P+1C assignment, no flag, no dead CB.

**The writer-side accessor name is fixed by the fork:** `dfb::out` on the TILE path, `dfb::out0` on the
RM path. The `DataflowBufferSpec` name itself is yours; the writer `KernelSpec`'s `accessor_name` is not.

One depth note to preserve: the CB is depth-**2** by default as a prefetch optimisation, falling back to
depth 1 when `2 * single_page_size` would exceed the L1 budget (`concat_program_factory.cpp:111-125`).
That is behaviour, not plumbing — carry the conditional into the `DataflowBufferSpec`.

### Runtime args — one vararg block, one CTA vararg, the rest named

**Reader RTA layout** (`concat_program_factory.cpp:270-281`):

| Slots | Contents | Disposition |
|---|---|---|
| `0` | `num_pages_per_core` | **name it** |
| `1` | `curr_tensor` | **name it** |
| `2` | `curr_tensor_id` | **name it** |
| `3 .. 3+N-1` | `Buffer*` per input | **gone** — becomes the `TensorBindingSequence` |
| `3+N .. 3+2N-1` | `num_pages_per_block[N]` | **RTA vararg** |
| `3+2N .. 3+3N-1` | `page_id_per_tensor[N]` | **RTA vararg** |

The two trailing `N`-element blocks are read in a **counted loop** — `arg_ptr[<base> + i]` inside
`for (i < num_tensors)` at `reader_concat_interleaved_start_id.cpp:39-43` and
`reader_concat_stick_layout_interleaved_start_id.cpp:38-42`. There are no per-argument names to infer,
so reach for the **RTA vararg mechanism** rather than trying to name `2N` slots. Note that `N` arrives as
a CTA rather than a runtime value — that still makes it a vararg, because it varies across
instantiations. **Do let slots 0-2 ride as named args**; they are distinct fixed fields and should not be
swept into the vararg block.

**CTA vararg — RM reader only.** `reader_concat_stick_layout_interleaved_start_id.cpp:57` and `:70` read
`kernel_compile_time_args[page_size_base_idx + curr_tensor]`, where `curr_tensor` is a **runtime** value.
That is a compile-time arg read at a varying index → `KernelAdvancedOptions::compile_time_varargs`, read
kernel-side with `get_compile_time_vararg(i)`. Genuine, and rare — don't try to name the elements.

The TILE reader (`reader_concat_interleaved_start_id.cpp`) has **no** equivalent: it takes its page size
from `get_tile_size(cb_id_in)` (line 28) and needs no per-tensor page-size CTA at all. So only the RM
reader gets `compile_time_varargs`.

**Writer RTAs → the forks' named args.** Both mappings are total — every value concat passes has a home,
and neither fork asks for anything concat lacks:

| Legacy writer RTA (concat) | RM fork named arg | TILE fork named arg |
|---|---|---|
| `dst_buffer` | → `tensor::dst` binding | → `tensor::dst` binding |
| `output.buffer()->page_size()` | `stick_size` | *(not passed — fork uses `dfb.get_entry_size()`)* |
| `num_pages_per_core` | `num_sticks` | `num_pages` |
| `num_pages_written` | `start_id` | `start_id` |
| `src0_cb_index` (CTA) | → `dfb::out0` binding | → `dfb::out` binding |

**No per-core-group CTA question here.** Unlike `ConcatS2SRMProgramFactory` below, this factory already
carries its per-core-group value (`num_pages_per_core`) in a **per-core RTA**
(`concat_program_factory.cpp:247-248, 272`), with single reader and writer `KernelDescriptor`s over
`all_cores`. There is nothing to demote, and nothing to promote — leave the shape as it is.

## Watch for — `ConcatProgramFactory`

- **Both donor writers are at rung 1: a `_metal2` fork already exists. Bind it; do not fork again.**

  | Path | Fork to bind | `dfb::` | `tensor::` | Named args | `#ifdef`s (concat sets none) |
  |---|---|---|---|---|---|
  | RM | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp` | `out0` | `dst` | `stick_size`, `num_sticks`, `start_id` | `BACKWARDS` |
  | TILE | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` | `out` | `dst` | `num_pages`, `start_id` | `OUT_SHARDED`, `BACKWARDS` |

  Fit was checked in both directions and **both forks fit concat exactly** (mapping table above) — so
  there is no handoff point, no fork edit, and no rung-2 carve-out in this port. You write nothing into
  either peer directory. Adopt the forks' names as-is; a fork with an existing consumer is read-only.

- **⚠ There are TWO forks of the TILE donor. Bind the one beside the original.**
  `copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` is a second,
  functionally identical fork living in a *consumer's* directory, and it names its accessor
  **`tensor::output`** instead of `tensor::dst`. The canonical fork — the one the locational rung-1 test
  finds, and the one to bind — is the sibling of the original under `eltwise/unary/`. Both are flagged
  for consolidation in the canonical fork's header (lines 13-16) and under issue #52228. Binding the
  wrong one silently inherits the wrong `tensor::` name.

- **Sunset lists — coordination, not authorization to convert in place.** The legacy originals keep
  serving their remaining binders: the RM donor has **1 other** consumer
  (`data_movement/copy/device/copy_same_memory_config_program_factory.cpp:37`); the TILE donor has
  **~23**, with the authoritative list and sunset plan at **issue #52228**. Record the still-unmigrated
  consumers in the port report so the port that turns out to be the last one can see that it was.

- **`get_tile_size(cb_id_in)` @ `reader_concat_interleaved_start_id.cpp:28` is a sanctioned Device 2.0
  free function** — the Device 2.0 gate is GREEN and this is not a holdover, despite a `DataflowBuffer`
  being in scope on line 45. Under the kernel-side whitelist a Metal 2.0 port moves such lookups onto
  the DFB object, so confirm the DFB equivalent rather than swapping blind. The analogous call in the
  TILE donor (`get_local_cb_interface(cb_id_out).fifo_page_size`) is **already done for you** — its
  `_metal2` fork replaces it with `dfb.get_entry_size()` (fork line 37).

- **A dead CTA slot disappears on its own.** `concat_program_factory.cpp:214` bakes
  `dst_buffer->page_size()` into RM writer CTA slot 1, which the *legacy* donor never reads (it takes
  `stick_size` from RTA slot 1 instead; the CTA slot exists only to satisfy the donor's
  `TensorAccessorArgs<2>` offset). The fork takes **no CTAs at all**, so don't carry the slot forward —
  and don't treat its disappearance as a behaviour change.

---

# `ConcatS2SRMProgramFactory`

## Construct — to do

### Tensor bindings — all clean

`input_0`, `input_1`, `output` — **all clean** (borrowed-memory DFB). Each is a `CBDescriptor` with
`.buffer = <tensor>.buffer()` (`concat_s2s_rm_program_factory.cpp:69, 86`), and the kernel reaches
tensor data through `input_dfb_0.get_read_ptr()` / `output_dfb.get_write_ptr()`. The DFB *is* the tensor
access → port via `DataflowBufferSpec::borrowed_from`. No Case 1, no Case 2, no work items.

**TensorParameter relaxation:** none. **TensorAccessor 3rd arg:** none — the kernel builds no accessor.

### CB endpoints — assign 1P+1C on all three CBs

| CB | Backing | Disposition | Config |
|---|---|---|---|
| `0` (`input_dfb_0`) | borrowed — `input_tensors[0]` | **1P+1C** | both configs |
| `1` (`input_dfb_1`) | borrowed — `input_tensors[1]` | **1P+1C** | both configs |
| `16` (`output_dfb`) | borrowed — `output` | **1P+1C** | both configs |

This is the **dual-instance work-split**: the factory pushes one kernel source
(`reader_height_sharded_width_concat_two_tensors.cpp`) into two `KernelDescriptor`s that differ only by
`ReaderConfigDescriptor` / `WriterConfigDescriptor` and their per-instance work-split CTAs, both over the
**same** `core_ranges` (`concat_s2s_rm_program_factory.cpp:166-188`). Both instances run on every node,
so each node has exactly two touchers of each CB — and **both touches are sync-free raw peeks**
(`input_dfb_0.get_read_ptr()` @ kernel:45, `input_dfb_1.get_read_ptr()` @ kernel:70,
`output_dfb.get_write_ptr()` @ kernel:42; the kernel contains **no** `reserve_back`/`push_back`/
`wait_front`/`pop_front` anywhere). Two role-free touchers → bind one instance PRODUCER, the other
CONSUMER. The labels are cosmetic on Gen1 and the kernel code is untouched.

### Runtime args — none

The factory sets **zero** runtime args. Four CTA vectors of 14 elements each
(`concat_s2s_rm_program_factory.cpp:104-164`), read as `get_compile_time_arg_val(0)`…`(13)`
(kernel:14-30). All fourteen nameable. No RTA varargs, no CTA varargs.

## Watch for — `ConcatS2SRMProgramFactory`

- **The per-core-group CTA split must become two `WorkUnitSpec`s — do not demote it to runtime args.**
  When `num_output_rows_per_core_last > 0` (`concat_s2s_rm_program_factory.cpp:190-197`), the factory
  splits `all_cores` into `first_cores` / `last_cores` and emits **four** `KernelDescriptor`s of the one
  source: a reader/writer pair per core group, each pair carrying its own CTA values
  (`compile_time_args_0/1` vs `compile_time_args_0_last/1_last`). Translate as **two `WorkUnitSpec`s over
  disjoint node sets**, each holding its own pair of same-source `KernelSpec`s with its own
  `compile_time_args`. Moving the per-group values (`num_output_pages`, `page_start`, `page_end`,
  `output_stick_offset`, `input_start_0/1`) into `runtime_arg_names` is the
  **demoting-per-group-CTA anti-pattern**: it sacrifices compile-time loop unrolling on those bounds for
  no benefit.

  **Note the two shapes stack here.** Disjoint node sets *across* the two groups (so each node sees one
  reader-config and one writer-config instance), and two touchers *within* each node (so each group's
  pair still takes the 1P+1C assignment above). Census per node is 2 in **both** configs — the
  disposition does not flip when the split fires.

- **`groups` is `const`, not `constexpr`, in this kernel — and `constexpr` divisors depend on it.**
  `reader_height_sharded_width_concat_two_tensors.cpp:28` declares
  `const uint32_t groups = get_compile_time_arg_val(11)`, while lines 32-35 use it in `constexpr`
  initialisers (`group_stick_size_0 = input_stick_size_0 / groups`, etc.). The tiled factory's kernels
  declare their `groups` `constexpr` (reader:30, writer:29, compute:48). Since a `constexpr`-vs-`const`
  declaration decides token-form versus member-getter on the Metal 2.0 side, check each declaration
  rather than assuming they match across the two factories.

---

# `ConcatS2STiledProgramFactory`

## Construct — to do

### Tensor bindings — all clean

`input_0`, `input_1`, `output` — **all clean** (borrowed-memory DFB;
`concat_s2s_tiled_program_factory.cpp:102, 116`). The compute kernel only consumes from / produces to
CBs, so it holds no tensor binding.

**TensorParameter relaxation:** none. **TensorAccessor 3rd arg:** none — no accessor in any of the three
kernels.

### CB endpoints — six plain 1:1 (no action), one self-loop

| CB | Name | Backing | Producer | Consumer | Disposition |
|---|---|---|---|---|---|
| `0` | input0 | borrowed — `input_tensors[0]` | reader (`push_back` @ reader:53) | compute (`wait_front`/`pop_front`) | plain 1:1 — no action |
| `1` | input1 | borrowed — `input_tensors[1]` | reader (`push_back` @ reader:54) | compute (`wait_front`/`pop_front`) | plain 1:1 — no action |
| `2` | output | borrowed — `output` | writer (`reserve_back`/`push_back` @ writer:43,61) | — none | **self-loop** — bind the writer PRODUCER **and** CONSUMER |
| `3` | input0_transpose | L1 scratch | compute (`reserve_back`/`push_back`) | reader (`wait_front`/`pop_front` @ reader:59,101) | plain 1:1 — no action |
| `4` | input1_transpose | L1 scratch | compute (`reserve_back`/`push_back`) | reader (`wait_front`/`pop_front` @ reader:103,145) | plain 1:1 — no action |
| `5` | concat | L1 scratch | reader (`reserve_back`/`push_back` @ reader:57,147) | compute (`wait_front`/`pop_front`) | plain 1:1 — no action |
| `6` | output_transpose | L1 scratch | compute (`reserve_back`/`push_back`) | writer (`wait_front`/`pop_front` @ writer:44,60) | plain 1:1 — no action |

Note the numbering: `cb_output_id = num_input_tensors` = **2**, and the scratch CBs run 3-6
(`concat_s2s_tiled_program_factory.cpp:106, 128, 140, 152, 163`). The reader's CTAs 0 and 1 are the
literals `0, 1` (`:191-192`) — the two input CBs.

### Runtime args — none

The factory sets **zero** runtime args. One shared 14-element compile-time arg list
(`concat_s2s_tiled_program_factory.cpp:190-205`) handed to all three kernels, every element read at a
literal index. All nameable. No RTA varargs, no CTA varargs.

## Watch for — `ConcatS2STiledProgramFactory`

- **Resolve the compute kernel's unused output DFB before writing its `dfb_bindings`.**
  `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp:57` constructs
  `DataflowBuffer output_dfb(output_dfb_id);` and **never uses it** — it is a dead local. That is why
  CB 2's census is 1 (writer only) and its disposition is a self-loop rather than 1P+1C. Read the kernel
  body before you bind: if you give compute a CB-2 binding to satisfy the construction, you have created
  a two-toucher where the code has one. The audit's recommendation is to bind CB 2 to the **writer only**
  (self-loop) and leave the dead local for the ops team.

- **Two `defines` axes to carry over:** `BF8` (set when inputs are `BFLOAT8_B` — it swaps the transpose
  CBs to a bf16 format and changes the reader's stride arithmetic) and `USE_SINGLE_PACKET_READ` (set when
  both input strides fit `NOC_MAX_BURST_SIZE`, which the factory computes per-arch: 16384 on Blackhole,
  8192 otherwise) — `concat_s2s_tiled_program_factory.cpp:182-213`. Neither changes the CB set or the
  endpoint census, but both must reach the **reader's** `compiler_options.defines`. The writer and
  compute kernels receive **no** defines.

---

# Op-wide watch-for

- **CB endpoints (multi-binding): none, anywhere in this op.** No CB in any of the six factories reaches
  ≥3 distinct touchers or has two kernels locked to the same FIFO role, so **do not set the
  multi-binding advanced option anywhere in this port.** The hidden-second-writer hunt was run and came
  back empty — and it is structurally impossible here: the op declares **no semaphores at all**, so
  there is no primitive for a semaphore-gated raw co-fill to coordinate with. If you find yourself
  reaching for the flag, recount.

- **Endpoint dispositions are yours to re-derive.** All of the above are mechanical enough to verify:
  run the census yourself and follow *it*. If your count disagrees with a table here, follow your count
  and note the disagreement in the port report.

- **Shared kernels: only `ConcatProgramFactory` has any exposure, and both are rung 1.** The two sharded
  factories bind exclusively concat-owned kernels, and the lent-shape census confirms **no other op
  binds any concat kernel** — so no concat kernel needs a fork, and nothing you change in
  `device/kernels/` can break another op.
