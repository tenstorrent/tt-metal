# Spec: porting unified kernel arguments to named arguments

Investigation and proposal. Written against `origin/main` at `9aa797dec5e`, after the rebase.

**STATUS: Phase 1 and the Phase 2 sentinel are DONE.** Decision 3 was settled first and in
the port's favour -- see the note at the end of §4. The rest of the document stands as
written; §8's decisions are answered, not open.

Motivation is hazard **D17** in `unified_api_hazards.md`: the runtime-argument list is
positional and untyped, a contract living in the kernel and in every launcher of it with no
compiler behind it. It has hung the device **three times**, each time by a kernel reading a
loop bound out of an argument slot nobody filled. D18 is the same defect on the compile-time
side: `TensorAccessorArgs<N>` offsets are chained by hand, so inserting one argument shifts
every one downstream.

---

## 1. The headline finding

**"Metal 2.0 argument passing" is three different things in this tree, and only one of them is
reachable from how we build programs.**

The half we can adopt today is the compile-time half — which is D18, the one that has never
hurt us. The runtime half, which is D17 and every hang, is **not reachable** from the
`ProgramDescriptor` path without either adopting a deprecated Blaze feature or migrating to a
host API that has no Python bindings.

That asymmetry is the whole decision, so it is stated first.

## 2. What exists, precisely

Three generations coexist. They are easy to confuse because two of them spell things
similarly.

### (a) Named compile-time args — `get_named_compile_time_arg_val("name")`

- Host: `KernelDescriptor::named_compile_time_args`, a `vector<pair<string, uint32_t>>`.
  **Python-bound** (`ttnn.KernelDescriptor.named_compile_time_args`).
- JIT: `write_named_ct_arg_map_header()` in `genfiles.cpp`, called **unconditionally** — the
  one piece not fenced behind the Metal 2.0 gate. Emits `KERNEL_COMPILE_TIME_ARG_MAP`,
  force-included by the compile recipe.
- Kernel: `constexpr auto x = get_named_compile_time_arg_val("x");` from
  `api/compile_time_args.h`.
- **In production use**, not just an example: ttnn's own matmul kernels use it
  (`reader_bmm_tile_layout_in1_sender_writer_padding.cpp`, `writer_bmm_tile_layout.cpp`).

### (b) Blaze named runtime args — `blaze_rt_args::get<blaze_ct_args::ns::field>()`

- Host: `KernelDescriptor::blaze_named_args`. **Python-bound** (`blaze_named_common_runtime_args`,
  `blaze_named_per_core_runtime_args`, and array variants).
- Covers RTAs and CRTAs, scalars and arrays, per-core and common.
- **Explicitly temporary.** Its own README: *"Temporary feature. This will be deleted when
  Blaze migrates to the Metal 2.0 `args::` system. Do not use in new non-Blaze code."*
  Removal tracked by tenstorrent/tt-metal#50953.

### (c) Metal 2.0 proper — `get_arg(args::name)` and `TT_KERNEL`

- Device: `tt_metal/hw/inc/experimental/kernel_args.h`. `RtaArg` / `CrtaArg` / `CtaVal`, one
  overloaded `get_arg()` covering all three kinds.
- The design goal is genuinely the right one: *"The kernel source is identical regardless of
  whether an arg is dispatched via RTA, CRTA, or CTA — moving an arg between kinds only
  requires a host-side schema change."* Compare our kernels, where the CTA/RTA split is baked
  into the source and moving an argument means editing the kernel.
- Plus `TT_KERNEL` (`tech_reports/NamedKernelArgs/kernel_args_as_parameters.md`): the entry is
  a plain typed function, template parameters are CTAs, function parameters are runtime args,
  and `genfiles` generates `kernel_main()` from the signature by tokenizing the source.
- **Gated.** `genfiles.cpp`: *"This is only invoked for Metal 2.0 kernels created via the new
  host API. Legacy kernels do not get `kernel_args_generated.h`."* The gate is
  `settings.is_metal2_kernel()`, set only by the ProgramSpec path.

### Reachability from `unified_harness.py` (`ttnn.generic_op` + `ProgramDescriptor`)

| mechanism | covers | reachable today | status |
|---|---|---|---|
| `get_named_compile_time_arg_val` | CTAs | **yes**, Python-bound | stable, in ttnn production kernels |
| Blaze `blaze_rt_args::get<>` | RTAs, CRTAs, arrays | yes, Python-bound | deprecated by design, deletion tracked |
| Metal 2.0 `get_arg(args::x)` / `TT_KERNEL` | CTA, RTA, CRTA uniformly | **no** | needs the ProgramSpec host API |

## 3. Why Metal 2.0 proper is out of reach

`is_metal2_kernel` is set only for kernels built through `experimental::metal2_host_api`. That
is not an argument-passing API — it is a different program model:

- `ProgramSpec` replaces `ProgramDescriptor`
- `DataflowBufferSpec` replaces circular buffers
- `WorkUnitSpec` + `NodeCoord` replace core ranges
- `TensorParameter` + `ProgramRunArgs` replace tensor arguments and runtime args
- `KernelSpec::RuntimeArgSchema` carries the named RTA/CRTA lists

None of `tt_metal/api/tt-metalium/experimental/metal2_host_api/*` is bound into Python. So
adopting (c) means porting `unified_harness.py`'s entire program construction to a C++ API
that our tests cannot currently call, and rewriting every CB declaration as a dataflow buffer.
**That is a project, not an argument port**, and it should not be smuggled in under this one.

## 4. Proposal

### Phase 1 — named compile-time args. Do this.

Move every scalar compile-time argument from the positional list to a name. Leave
`TensorAccessorArgs` positional (see §5).

Kernel, before and after:

```cpp
// now
constexpr uint32_t mt   = get_compile_time_arg_val(0);
constexpr uint32_t ktot = get_compile_time_arg_val(1);
constexpr uint32_t ntot = get_compile_time_arg_val(2);
constexpr uint32_t kt   = get_compile_time_arg_val(3);
constexpr uint32_t nt   = get_compile_time_arg_val(4);
constexpr auto a_args = TensorAccessorArgs<5>();

// after
constexpr uint32_t mt   = get_named_compile_time_arg_val("mt");
constexpr uint32_t ktot = get_named_compile_time_arg_val("ktot");
constexpr uint32_t ntot = get_named_compile_time_arg_val("ntot");
constexpr uint32_t kt   = get_named_compile_time_arg_val("kt");
constexpr uint32_t nt   = get_named_compile_time_arg_val("nt");
constexpr auto a_args = TensorAccessorArgs<0>();     // <- and it STAYS 0
```

**The second win is the bigger one.** With the scalars gone from the positional list, the
accessor block starts at 0 and stops moving. Today adding one compile-time argument shifts
every accessor offset in the kernel, which is D18 exactly.

Failure mode on a typo or a missing name: `get_named_ct_arg` walks the map and falls off the
end into `__builtin_unreachable()`. In a `constexpr` context — which is how we always use it
— that is a **build failure**, not a hang.

**The header's "fails with a segfault" caveat is STALE for our toolchain**, which was worth
the ten minutes to check rather than accept. Both compilers give a clean diagnostic naming
the line and the offending name:

    // riscv-tt-elf-g++ (what actually builds kernels)
    typo_test.cpp:10:42: in 'constexpr' expansion of 'get_named_ct_arg(..."mtt")'
    typo_test.cpp:7:26: error: '__builtin_unreachable()' is not a constant expression

    // clang-20
    error: constexpr variable 'bad' must be initialized by a constant expression
    note: in call to 'get_named_ct_arg({3, &"mtt"[0]})'

So decision 3 resolves in favour of the port: the diagnostic is good, not a compiler crash.

### Phase 2 — named runtime args. Decide, do not default.

This is D17, the one with three hangs. Three options, none free:

- **2a. Use Blaze named RT args now.** Python-bound, works today, covers per-core and arrays
  (`block_begin`/`block_count` are per-core, so we need the per-core variant). Cost: adopting a
  feature whose README says not to, with a deletion issue open — we would be volunteering for
  the migration later. Buys the hang class immediately.
- **2b. Wait for Metal 2.0 to reach the descriptor path**, or for Python bindings on the 2.0
  host API. Cost: unknown schedule; D17 stays live meanwhile.
- **2c. Close it ourselves without either.** A `runtime_args` *count* sentinel: the kernel
  asserts the argument count it was given matches what it expects. Cheap, catches the exact
  failure that hung us three times (a launcher passing too few), catches nothing about order.
  Roughly ten lines, no dependency on any experimental feature, and it composes with 2a or 2b
  later rather than blocking them.

**Recommendation: Phase 1 now, plus 2c as a stopgap. Defer 2a.** 2c gets most of D17's value
for a fraction of the risk, and does not commit us to a feature that is scheduled for
deletion. Revisit 2a only if 2c proves insufficient in practice.

## 5. What stays positional, and why

`TensorAccessorArgs<N>` consumes a **contiguous positional block** whose length depends on the
tensor's layout, and `next_compile_time_args_offset()` chains one to the next. Named args are a
name→`uint32_t` map with no notion of a run of slots. Metal 2.0 has `compile_time_varargs`
for exactly this, but that is on the 2.0 host API and unreachable (§3).

So accessors stay positional. This is fine, and arguably better than a name each: the chain is
self-describing (each accessor computes the next offset), and once the scalars leave, the block
starts at 0 and nothing perturbs it. The residual hazard is only that two accessors could be
chained in the wrong order — a real but much narrower failure than today's.

## 6. Scope

14 kernels in `unified_kernels/`, 8 launchers. The two forms coexist (`compile_time_args` and
`named_compile_time_args` are independent fields), so this is **incremental, kernel by kernel**,
with the suite green after each. No library change in `tt/unified/` — this is kernels and
harnesses only.

Suggested order, easiest first, each one a checkpoint:

1. `example_reduce.cpp` — 2 accessor blocks, no scalars, proves the accessor-offset-0 claim
2. `binary.cpp`, `unary.cpp`, `rope.cpp` — few scalars, several launchers each, so they also
   prove the "every launcher must agree" story
3. `rmsnorm.cpp`, `matmul_blocked.cpp` — the real shapes
4. `flash_attention.cpp` — most arguments, most to gain
5. The rest

## 7. What this does and does not buy

**Does:** kills D18 outright. Removes the offset arithmetic from every kernel. Makes a
mistyped or missing compile-time argument a build failure. Makes the kernel source say what
each constant *is*, which is worth something on its own — `get_compile_time_arg_val(3)` tells
a reader nothing.

**Does not:** touch D17 unless Phase 2 is taken. Named CTAs cannot hang the device, and the
device hangs were all runtime args. **Adopting Phase 1 alone leaves every hang we have
actually suffered still possible** — worth being blunt about, because the phase that is easy
is not the phase that hurts.

## 8. Decisions needed

1. **Phase 1: go?** Low risk, incremental, reversible per kernel.
2. **Phase 2: 2c stopgap, 2a Blaze, or defer entirely?** My recommendation is 2c now,
   revisit 2a later, but the appetite for depending on an explicitly-temporary API is a call
   about the project's relationship with Blaze, not a technical one.
3. **Is the compiler-segfault diagnostic acceptable** as the failure mode for a mistyped
   name? If not, Phase 1 should wait for that to be fixed upstream, or we wrap the accessor
   in something that produces a `static_assert` instead — worth ten minutes of investigation
   before committing to the port.

## 9. Explicitly out of scope

Migrating to the Metal 2.0 host API (`ProgramSpec`, dataflow buffers, work units, node
coords). That is a rewrite of how we build programs, it needs Python bindings that do not
exist, and it would replace the circular-buffer model the whole unified library is written
against. If it becomes desirable it deserves its own spec, and `TT_KERNEL`-style typed entry
points — genuinely the nicest thing in Metal 2.0 — arrive with it rather than before it.
