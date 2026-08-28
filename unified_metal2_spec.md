# Proposal: supporting Metal 2.0 in the unified programming model

Written by reading the tree at `8bb48ab0f1d`. Every claim is sourced to a file and line.

**THE GATE HAS BEEN RUN, on a Wormhole n150, and it passes.** See §7.1, which was written after the
fact and CORRECTS this document where the two disagree; the probes are in `unified_gate/`. Claims
still only read off the source, never executed, stay marked UNVERIFIED.

This is the document `unified_named_args_spec.md` §9 deferred: *"Migrating to the Metal 2.0 host
API ... is a rewrite of how we build programs ... If it becomes desirable it deserves its own
spec."*

---

## 1. The headline finding

**"Metal 2.0" is two independent migrations wearing one name, and only one of them is blocked.**

| layer | what changes | reachable from where we are today |
|---|---|---|
| **Kernel object model** — `DataflowBuffer`, `Noc`, `Semaphore`, `CircularBuffer` | free functions become objects | **yes.** Every one of them is constructible from a raw id under the legacy `ProgramDescriptor` path, and works on WH/BH |
| **Host program model** — `ProgramSpec`, `KernelSpec`, `DFBBinding`, `TensorParameter`, `ProgramRunArgs` | descriptors become a validated spec, and codegen emits `dfb::`/`sem::`/`tensor::`/`args::` binding tokens | **no.** No Python bindings; `ttnn.generic_op` accepts only `ProgramDescriptor` / `MeshProgramDescriptor` (`generic_op.hpp:20-26`) |

The `unified_named_args_spec.md` finding was that the *runtime-argument* half of 2.0 is gated
behind a host API with no Python bindings. That is still true. What that spec did not look at —
because it was scoped to arguments — is that **the kernel-side half of 2.0 is not gated at all.**

`DataflowBuffer` has a public low-level constructor taking a raw id:

    // tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:113
    // Low-level constructor: prefer DFBBindingToken overload above for new kernel code.
    DataflowBuffer(uint16_t logical_dfb_id);

and on tt-1xx that constructor is a one-line wrapper over the same `LocalCBInterface` our
`cb_page_bytes` already reads:

    // tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl:31
    : logical_dfb_id_(logical_dfb_id), local_dfb_interface_(get_local_cb_interface(logical_dfb_id))

`Noc` is the same story — `Noc()` defaults to `noc_index`, `Noc(uint8_t)` names one
(`noc.h:146-147`) — and we are *already* on 2.0's `Semaphore`: `api.h`'s `Semaphore<thread>` wraps
`::Semaphore<ProgrammableCoreType::TENSIX>` from `api/dataflow/noc_semaphore.h`.

So the shape of the answer is neither "wholesale port" nor "two implementations forever". It is
**one API, two implementations, with the split placed where the blockage actually is** — and the
first implementation step costs no host work at all.

---

## 2. What the library got right, and what this tests

`tt/unified/core` was written for this exact question and says so:

    // This file is the version selector. Today there is one target -- Metal v1
    // (Wormhole / Blackhole) -- so it picks the v1 adaptor and implementation
    // unconditionally. As other metal versions arrive the choice happens here, and
    // nothing above it changes.

The layering it describes is the right one, and the port is a test of whether it holds:

    core               -- selects a backend + implementation
      adaptor_v1.hpp   -- binds the model's intrinsics to metal v1     <- REPLACED
      api.h            -- core API declarations (version-agnostic)     <- mostly unchanged
        math.hpp       -- leaves, ops, fusion kinds, strategies        <- unchanged
          expr.hpp     -- op-agnostic tree, allocator, method syntax   <- unchanged
      impl_v1.hpp      -- definitions for the v1 target                <- REPLACED

**The compute layer does not move.** `math.hpp` (2049 lines) and `expr.hpp` (494) call
`api/compute/*` — `matmul.h`, `eltwise_binary.h`, `reduce.h`, `pack.h` — and those headers are
arch-generic, gating `ARCH_QUASAR` internally rather than being replaced. That is 2543 of the
library's 5522 lines that a Metal 2.0 port does not touch, and it is the part that took the
longest to get right.

**`impl_v1.hpp` is where the whole cost is.** Its 1253 lines contain 30 circular-buffer protocol
call sites and 33 NOC call sites, and 30 of its regions are behind `IS_DM_THREAD` against 3 behind
`IS_COMPUTE_THREAD`. Data movement is the port.

Two things the layering got *wrong*, both of which this exercise exposes:

1. **`api.h` is not as version-agnostic as it claims.** It hardcodes a two-DM-thread machine:
   `kMcastReadySem<thread> = kMcastSemBase + 2*thread`, `kCopyArrivedSem<thread> = base + 4 +
   thread`, and the documented convention "READS ON THREAD 0, writes on 1". Quasar has **six user
   DM cores** (`program_spec.cpp:47-49`: 8 per node, 2 reserved) and four Tensix engines
   (`:50`). See §6.
2. **`Storage` carries only a cb id.** Metal 2.0 wants a *role* — producer or consumer — declared
   per kernel, per buffer, on the host. See §5.1, which is the single hardest part of this and the
   most interesting.

---

## 3. The mapping, concept by concept

| unified concept | metal v1 (today) | metal 2.0 | note |
|---|---|---|---|
| `Storage<S>(cb_id)` | `cb_reserve_back(cb, n)` etc., free functions resolving per projection | `DataflowBuffer dfb(id_or_token)`; `dfb.reserve_back(n)` | **strictly better fit.** One object whose methods already resolve on every projection |
| `cb_page_bytes(cb)` | `get_local_cb_interface(cb).fifo_page_size << cb_addr_shift` — the shift written out by hand | `dfb.get_entry_size()` | the arch-unit shift moves behind the API (`dataflow_buffer.inl:46`) |
| `cb_num_pages(cb)` | `.fifo_num_pages`, DM-only, with a long comment on why it does not link on TRISC | `dfb.get_total_num_entries()` | same restriction, better name |
| `noc_async_read(...)`, trailing `uint8_t noc` | free functions, NOC implicit | `noc.async_read(src, dst, bytes, src_args, dst_args)` on a `Noc` object | §4 — this *is* `unified_explicit_noc_spec.md`'s answer |
| `noc_async_read_barrier()` | free function | `noc.async_read_barrier()` | the NOC travels with the object, so §3 of the noc spec ("handles must carry it") becomes "handles carry a `Noc`" |
| `noc_async_write_multicast_loopback_src` | separate function | `noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(...)` | `noc.h:44` |
| `Semaphore<thread>` | already `::Semaphore<TENSIX>` | same class, plus `sem::name` tokens | **already ported** |
| `TensorAccessor(TensorAccessorArgs<N>(), base)` | positional CTA block, offsets chained by hand (hazard D18) | `TensorAccessor(tensor::name)` | `tensor_binding_token.h:22-27`. Host-only; needs §5.2 |
| CT args | `get_compile_time_arg_val(N)` / `get_named_compile_time_arg_val("x")` | `get_arg(args::x)` — CTA/CRTA/RTA behind one spelling | `kernel_args.h:16-23`. Host-only |
| runtime args | positional + our sentinel | `get_arg(args::x)`, schema-checked | **this is hazard D17, the three device hangs.** Host-only |
| thread identity | `COMPILE_FOR_BRISC` / `COMPILE_FOR_NCRISC` | `COMPILE_FOR_DM=<id>` (`hal_2xx_common.cpp:28`) | gen2 gives an *index*, not two booleans — cleaner, and it extends past two |
| compute identity | `UCK_CHLKC_{UNPACK,MATH,PACK}` | **same on gen2** (`hal_2xx_common.cpp:36-58`) | unchanged, per NEO |

The one-line summary of that table: **Metal 2.0 replaces the parts of the unified library that are
thin wrappers, and leaves alone the parts that are load-bearing.**

---

## 4. Metal 2.0 answers `unified_explicit_noc_spec.md` outright

That spec proposes a `NocTag<N>` threaded through 25 entry points, because "our wrappers drop the
`noc` parameter and the unified API should not be less expressive than the API it wraps."

Metal 2.0's answer is a `Noc` object. It carries `noc_id_`, every operation is a method on it, and
its barriers are methods too — so the spec's §3 constraint ("each of the five handle types has to
carry the NOC it was issued on, and its `wait()` has to barrier on that one") is satisfied by
storing a `Noc` in the handle instead of a `uint8_t`. There is no ambiguity problem (§5 of that
spec: appending `uint8_t` breaks every existing multicast call site), because a `Noc` is not
constructible from an integer literal — which is exactly what `NocTag` was invented to achieve.

The spec also notes, of the hang that cost it two device resets:

> `program_spec.cpp:905` validates exactly this ... "both are dedicated-NOC data movement kernels
> pinned to NOC_{}, which **hangs the device**" ... **the 2.0 path would have turned this
> ablation's hang into a diagnostic.**

Confirmed at `tt_metal/impl/metal2_host_api/program_spec.cpp:906-916`, and the mode-agreement check
is immediately above it at `:899-905`. That is a hang class the legacy `ProgramDescriptor` path does
not check and the 2.0 path does.

**Recommendation: do not implement `NocTag`.** Implement step 1 of that spec (handles carry the
NOC they were issued on — the step that is silently wrong if half-done), but carry it as a `Noc`
object from the start, and let step 2's call-site spelling be `u::noc0` implemented as a `Noc`
rather than as a bespoke tag type. That deletes about a third of the noc spec's work and lands us
on metal's own vocabulary.

---

## 5. The four hard parts

Everything above is mechanical. These are not.

### 5.1 DFB endpoint roles vs. one-source-five-projections

This is the interesting one, and it cuts both ways.

Metal 2.0 requires each kernel to declare, per buffer, whether it is the **producer** or the
**consumer**:

    // kernel_spec.hpp:125-141
    struct DFBBinding {
        enum class EndpointType { PRODUCER, CONSUMER };
        DFBSpecName dfb_spec_name;
        std::string accessor_name;   // the name the kernel source uses: dfb::<accessor_name>
        EndpointType endpoint_type;
    };

and enforces one producer instance and one consumer instance per node
(`dataflow_buffer_spec.hpp:41-44`).

**The bad news:** our host side does not know this. `unified_harness.py`'s `make_cb` says only
"here is a CB of this size and format"; which threads produce and which consume lives entirely in
the kernel source, and is deliberately implicit — that is the model's whole trick.

**The good news, and it is better than it first looks:** the unified model *already names these
three categories*, in the first comment block of `api.h`:

      INPUT                    OUTPUT                   INTERMED
           DM    Compute            DM    Compute            DM    Compute
      reserve <- *               * -> reserve                   reserve
        write                          write                      write
         push ->    wait         wait <-  push                     push
                   read          read                              wait
            * <-     pop          pop -> *                         read
                                                                    pop

That table *is* the endpoint binding, drawn in ASCII. INPUT = DM produces, compute consumes.
OUTPUT = compute produces, DM consumes. INTERMED = compute self-loop — which Metal 2.0
**explicitly supports**: `program_spec.cpp:299-303` handles "both bindings target the same DFB with
opposite endpoint types ... This lets a kernel that both produces and consumes the same DFB", and
`:942` calls it a "compute self-loop DFB". Our `Accumulator`, `RetainedBlock`, and every scratch
buffer are exactly that.

So the port converts the model's own documentation into something the host validates. Concretely,
`make_cb` gains a role:

```python
make_cb(kCbIn0,  ranges, role=Role.INPUT)      # reader produces, compute consumes
make_cb(kCbOut,  ranges, role=Role.OUTPUT)     # compute produces, writer consumes
make_cb(kCbAcc,  ranges, role=Role.INTERMED)   # compute self-loop
```

and the harness emits the three `KernelSpec`s' `dfb_bindings` from that. **This directly kills
hazard D20** ("CB index collisions, or a Storage naming a CB the host never declared" — verified
live in `matmul_blocked.cpp`), because a DFB with no producer or no consumer is a host-side error
(`program_spec.cpp:393-394`) rather than a silent zero-page read.

**What genuinely does not fit**, and must be called out rather than glossed:

- **Which DM thread.** INPUT says "DM produces" but not whether it is thread 0 or thread 1. Today
  that is decided in the kernel by `template<int thread>`. The host would need it too — so `role`
  is really `(role, thread)`, e.g. `Role.INPUT_ON(0)`. Two facts that must agree with nothing
  checking them is precisely the class of hazard this project keeps logging; **the mitigation is
  that a mismatch is now a build/launch error** (the DFB's producer kernel would be the wrong
  KernelSpec and the correct one would find no binding named `dfb::in0`), not a hang. That is an
  improvement, but it is a new coupling and should be specced honestly as one.
- **`fill_reduce_scaler`.** DM pushes one page; compute waits and *never pops*. That is a legal
  producer/consumer pair, but the DFB never drains. UNVERIFIED whether any 2.0 validation objects.
- **Buffers touched by only one side.** A `Storage` used purely as scratch by one projection has no
  second endpoint, and `program_spec.cpp:393-394` requires both. UNVERIFIED whether we have one.

### 5.2 The host API has no Python bindings, and the harness is Python

`ttnn.generic_op` takes `ProgramDescriptor` or `MeshProgramDescriptor` and nothing else
(`generic_op_nanobind.cpp:44-56`). Nothing under
`tt_metal/api/tt-metalium/experimental/metal2_host_api/` is bound. The 2.0 hardware tests are C++
gtests using `MeshBuffer` and `MakeProgramFromSpec` directly
(`tests/tt_metal/tt_metal/api/metal2_host_api/test_program_spec_hw.cpp`).

That matters more than it sounds, because the 19 kernels in `unified_kernels/` are validated by 20
pytest files that compare against torch. Losing that is losing the project's entire safety net.

Three ways out, in increasing cost:

- **(a) Bind a narrow shim.** One nanobind function, `unified_generic_op(io_tensors, spec)`, taking
  a Python-side description that C++ turns into a `ProgramSpec`. We do not need general
  `ProgramSpec` bindings — the harness builds one shape of program. Smallest surface that keeps
  pytest.
- **(b) Bind `ProgramSpec` properly.** More useful to everyone else, much more work, and it is
  upstream's call whether that lands where we want it.
- **(c) Move the harness to C++.** Loses torch comparison and the whole suite. Not recommended.

**Recommend (a)**, and note that it is the step that makes 2.0 reachable *at all* — so it should be
prototyped early enough to fail cheaply, before any library code is written against it. A
throwaway shim that runs `unified_kernels/unary.cpp` end to end through a `ProgramSpec` is the
gate on this whole proposal.

### 5.3 Core-to-core CB pokes have no 2.0 story yet

`noc_core_read` / `noc_core_write` and the multicast load write **into a peer core's copy of a
circular buffer**, addressed through the writer's local view. `api.h` already flags this as
unsound-in-general:

    // NOTE: reserve/push act on the *local* view of the destination CB. For a genuine
    // peer buffer the far side's pointers have to be advanced too

Metal 2.0's sanctioned replacement is the cross-node DFB, and it does not exist:

    // dataflow_buffer_spec.hpp:139-141
    // NOTE: Cross-Node DataflowBuffer is not yet supported!
    //       A sketch is included in the experimental Metal 2.0 APIs for visibility.

On Gen1 this is fine — a DFB is a `LocalCBInterface` and the raw-address pokes work exactly as they
do now. On Gen2 it is not obviously fine: Quasar DFBs have hardware tile counters and implicit
sync, and a NOC write into another core's DFB L1 that bypasses the credit machinery is at best
unvalidated. `DataflowBuffer` even has a `scoped_write_lock` whose stated purpose is to "flag any
NOC write into the locked entries as `WRITE_TO_LOCKED_DFB`" (`dataflow_buffer.h:349-352`), which
reads like the machinery for catching exactly what we do on purpose.

**This is the one place where "port to 2.0" and "run on Quasar" come apart, and it should not be
allowed to block the rest.** Concretely: a 2.0 port targeting WH/BH can keep the current
core-to-core implementation verbatim; a Quasar port cannot, and needs cross-node DFB or an
explicitly sanctioned raw-L1 path. Scope the first port to Gen1 and log this as the known gap.

### 5.4 Quasar is a different machine, and `api.h` assumes it is not

Not strictly a 2.0 issue — 2.0 runs on WH/BH today — but the reason 2.0 exists, so it belongs here.

- **Six user DM threads, not two.** `template<int thread>` generalizes, but `kMcastReadySem<thread>
  = base + 2*thread` and `kCopyArrivedSem<thread> = base + 4 + thread` are written for exactly two,
  and the second overlaps the first the moment there are three. These need deriving from a
  `kNumDmThreads` the adaptor supplies.
- **Four Tensix engines**, each with its own UNPACK/MATH/PACK. `IS_COMPUTE_THREAD` is currently a
  single boolean; on Quasar there are four independent compute kernels per node.
- **`num_threads > 1` per KernelSpec** with STRIDED / ALL / BLOCKED DFB access patterns
  (`kernel_spec.hpp:129-134`). This is a genuinely new axis the unified model has no word for.
- **Semaphore initial values must be zero on Quasar** (`program_spec.cpp:1695`). Our reserved
  multicast semaphores are all zero; UNVERIFIED whether any test's user semaphore is not.
- **The NOC-per-thread convention dissolves.** "READS ON THREAD 0, writes on 1" — worth 2.6x on the
  blocked matmul — is a two-RISC-two-NOC fact. It does not survive to a six-DM machine, and
  `unified_llama_prefill.md`'s conclusions are Gen1 conclusions.

**Recommendation: do not try to reach Quasar in this port.** Make `api.h` stop *asserting* two DM
threads (derive the semaphore layout from a constant the adaptor supplies), and leave the rest.

---

## 6. The three options

### Option A — wholesale port, `impl_v1.hpp` becomes `impl_v2.hpp`

Delete the v1 path. One implementation, one set of behaviours, no `#if` sprawl.

**For:** the library stays as simple as it is now, which is a real asset — `impl_v1.hpp` is dense
but it is *one* story. Every hazard fix lands once. No risk of the two implementations drifting
into subtly different semantics, which is the failure mode that would hurt most (the model's whole
value is that a kernel's meaning is stable).

**Against:** it is all-or-nothing against a host API with no Python bindings. Until §5.2 is solved
the suite cannot run *at all*, so there is no green checkpoint between "start" and "done" — and
every other change in this project has been landed against a green suite. It also strands
core-to-core (§5.3) with no fallback, and it throws away a working, measured, tuned Gen1 path
whose performance numbers (2.6x on matmul NOC assignment, 2.4x rmsnorm) were expensive to find.

### Option B — two implementations behind one API, permanently

`core` selects `adaptor_v1`/`impl_v1` or `adaptor_v2`/`impl_v2` on a define.

**For:** exactly what `tt/unified/core` was written for, and the honest answer while Gen1 and Gen2
both matter. Every step has a green checkpoint: v1 keeps running while v2 is built. Lets the Gen1
performance work stand while Quasar is targeted separately.

**Against:** two implementations of 1253 lines each is a real maintenance tax, and the *semantic*
divergence is the danger, not the line count. §5.1's endpoint roles and §5.4's thread count are
places where the two versions want a genuinely different `api.h`, and papering over that with
`#if` in the shared header is how a version selector rots into a compatibility layer.

### Option C — one API, staged: v2 objects first under the v1 host, then the v2 host **(recommended)**

The observation in §1 makes a third option available that neither A nor B describes. The two
migrations are separable, so separate them:

**Stage 1 — `adaptor_v2` + `impl_v2` on the legacy host path.** Replace the free functions with
`DataflowBuffer`, `Noc`, and object-based barriers, still constructed from **raw ids** supplied by
today's `ProgramDescriptor` harness. No host change, no Python bindings needed, the entire suite
runs after every commit.

**Stage 2 — the host port**, behind the shim of §5.2, adding binding tokens (`dfb::`, `sem::`,
`tensor::`, `args::`) and endpoint roles. This is where D17, D18, D19 and D20 die.

**Stage 3 — Gen2**, if and when it is wanted, with §5.3 and §5.4 as its own scope.

**For:** stage 1 is a pure refactor with a green suite throughout and no dependency on anything
experimental — it is the cheapest possible way to find out whether the object model actually suits
the unified library, which is the question A and B both *assume* an answer to. It converges on
Option A (one implementation) rather than Option B (two forever), while getting Option B's
checkpointing during the migration. `impl_v1.hpp` stays in the tree during stage 1 only as a
rollback, selected by `core`, and is **deleted at the end of stage 1** rather than maintained.

**Against:** stage 1 buys no hazard fixes on its own — every hazard in the ledger dies in stage 2.
It is a refactor whose payoff is deferred, and if stage 2 never happens it was churn. That risk is
real and worth naming; the mitigation is that stage 1 is small (the 63 call sites of §2) and that
its intermediate state is strictly no worse than today's.

---

## 7. Recommended plan

**The gate is done, and it passed** (§7.1). What follows is the plan it unblocks, each step
ending with the full suite green:

1. **`Storage` holds a `DataflowBuffer`, not a `cb_id`.** Replace the 30 CB call sites. Still
   constructed from a raw id. `cb_page_bytes`/`cb_num_pages` become `dfb.get_entry_size()` /
   `dfb.get_total_num_entries()` and the hand-written `cb_addr_shift` goes away.
2. **Handles hold a `Noc`.** Replace the 33 NOC call sites; every `wait()` barriers on its own
   handle's `Noc`. This is step 1 of `unified_explicit_noc_spec.md`, done with metal's own type —
   and it is the step that is silently wrong if half-done, so it goes alone.
3. **Delete `impl_v1.hpp` and `adaptor_v1.hpp`.** End of stage 1: one implementation, still on the
   legacy host.
4. **Roles on the host.** `make_cb(..., role=Role.INPUT_ON(0))`, harness emits `dfb_bindings`,
   `Storage` takes `dfb::name`. Kills D20.
5. **`tensor::name` accessors.** Kills D18 outright — better than `unified_named_args_spec.md`
   Phase 1 could manage, since that spec had to leave `TensorAccessorArgs` positional (§5 there)
   precisely because named CT args cannot express a contiguous run of slots.
6. **`get_arg(args::x)`.** Kills D17 — the three device hangs — and retires our runtime-arg
   sentinel, which exists only because this was unreachable.
7. **`TT_KERNEL` typed entry points**, if wanted. Cosmetic next to 4-6.

Steps 4-6 are the entire justification. Steps 1-3 are the ramp that makes them landable.

### 7.1 The gate, run

On a Wormhole n150, against this tree. Sources in `unified_gate/`; the host programs are standalone
C++ linked straight against `build/lib/libtt_metal.so`, so nothing had to be added to the build.

| probe | question | result |
|---|---|---|
| **Gate A** -- `gate_a.cpp` + `gate_host.cpp` | one source, three `KernelSpec`s, five projections, two DFBs whose endpoints straddle DM and compute, compute squaring each tile | **PASS**, 0/4096 values wrong |
| **Gate A'** -- `gate_a_tokens.cpp` | the same kernel spelled with `dfb::` binding tokens | **compile error**, as predicted |
| **Gate B** -- `gate_b.cpp` + `gate_host_b.cpp` | the unified library itself, the shape of `unified_kernels/unary.cpp`, under a `ProgramSpec` | **PASS**, 0/16384 values wrong |
| **Validation probe** | drop compute's producer binding of `out`, build only | **rejected at build**: `DFB 'out' has no producer` (`program_spec.cpp:393`) |

Four things came out of it that this document did not know.

**1. The gate's headline question is answered YES, and it was never close.** A Metal 2.0
`ProgramSpec` compiles one source for all five projections and binds one DFB with a DM producer and
a compute consumer. Gen1 emits the same `COMPILE_FOR_BRISC` / `COMPILE_FOR_NCRISC` /
`UCK_CHLKC_{UNPACK,MATH,PACK}` defines on the 2.0 path as on the legacy one, so
`adaptor_v1.hpp`'s projection detection works untouched.

**2. Gate B is the load-bearing result: the unified library runs under a ProgramSpec with ONE
additive line of change.** `Storage`, `Block`, `ComputeBlock`, `store()`, `noc_load`, `noc_store`
and the whole expression layer went across unmodified. The one change is in `adaptor_v1.hpp`: the
compute-projection `TensorAccessor` stand-in gained a one-argument constructor so
`TensorAccessor(tensor::in)` compiles on a TRISC, exactly as `TensorAccessor(args, addr)` already
did. That reorders the plan's risk: **stage 2 is not the leap this document treated it as**, and
§6's Option C could just as defensibly be run the other way round -- host first, kernel objects
after -- since the host port is now the demonstrated-cheap half.

**3. `dfb::` binding tokens are INCOMPATIBLE with the unified model, and this is the sharpest
finding of the exercise.** §5.1 worried about roles; the real problem is narrower and harder.
`write_kernel_bindings_generated_header` (`genfiles.cpp:129`) emits a token only for the bindings
of the kernel being compiled. A DFB has exactly two endpoint roles and they are taken, so a buffer
cannot be bound to all three kernels -- and a unified kernel declares EVERY `Storage` on EVERY
projection, unconditionally, which is the model's whole trick. Spelled with tokens, the gate fails
to build:

    gate_a_tokens.cpp:59:29: error: 'out' is not a member of 'dfb'   <- on the reader build
    gate_a_tokens.cpp:58:28: error: 'in' is not a member of 'dfb'    <- on the writer build

So **buffer slots must keep arriving as compile-time VALUES**, as `unified_harness.py` passes CB
indices today. Both gates do that and both pass. This does not extend to the other three token
namespaces: `tensor::` has no exclusive role, so Gate B binds both tensors to all three kernels and
`TensorAccessor(tensor::in)` works everywhere -- which is D18 dead, with no positional block left
to drift.

**4. Two corrections to this document.**

*§8 overclaimed on D20.* The validation probe confirms the completeness half: a DFB with no
producer is refused at build with the error naming the buffer. But with slots passed as values,
nothing checks that the kernel's `Storage(kCbAcc)` names the DFB the host meant -- the number is
just a number. So D20 splits: **"a buffer the host never declared" dies; "the kernel names the
wrong slot" survives.** The `matmul_blocked.cpp` instance in the hazard list was the second kind.
It stays caught only by our own capacity assert.

*§3 was wrong that CT args are optional in 2.0.* `KernelSpec::CompileTimeArgs` is a
`Table<std::string, uint32_t>` (`kernel_spec.hpp:190`) -- **named only**. There is no positional
list on the 2.0 host API at all, so `get_compile_time_arg_val(N)` and `TensorAccessorArgs<N>` have
no route in except the deprecated `compile_time_varargs` escape. Named CT args are not a Phase-1
choice on this path, as `unified_named_args_spec.md` framed them; they are the only option. Since
tensor bindings remove the one construct that needed a positional block, that costs nothing.

**What the gate did NOT establish.** Only one core, one node, one block shape, and a pointwise op.
No semaphores, no multicast, no core-to-core copy, no `Accumulator`, no `RetainedBlock`, and
nothing at all about §5.3 or §5.4. The slot numbers the kernels were told (`in` = 0, `out` = 1)
were assumed from the allocator's lowest-free-slot rule and confirmed only by the results being
correct; a real harness should read the assignment back rather than predict it. And the mis-binding
case that §9.3 asks about -- host roles that contradict what the kernel does -- was deliberately
NOT run, because a structurally valid spec that disagrees with the kernel is a hang, not an
exception, and that costs a device reset to learn.

---

### 7.2 The shim and the first ported suite

`ttnn.program_spec` (`ttnn/cpp/ttnn-nanobind/program_spec.cpp`) binds the 2.0 host API, and
`unified_harness.py` grew `unified_program_spec()` / `run_unified_spec()` beside the existing
`unified_program()`. `test_unified_unary.py --metal2` runs the same kernel through a
`ProgramSpec`; both paths pass, with **identical numbers** on all five ops (recip PCC 0.999991
/ rel 0.00421, and so on down the list) -- which is a stronger agreement than a tolerance
check, since any difference in what reached the device would move them.

Three things the port settled that the spec had left open.

**Endpoint roles are the api.h table, and they were cheap.** `dfb_input(name, thread)` /
`dfb_output(name, thread)` / `dfb_intermed(name)` are the three columns of that comment block,
and the harness turns them into producer/consumer bindings. §5.1 called this the hardest part;
in practice it was about twenty lines. The cost it predicted is real but smaller than
described: the DM *thread number* has to agree between the role and the kernel's
`noc_load<0>` / `noc_store<1>`, and nothing states that in one place.

**The slot prediction is now CHECKED, and the check is what `dfb::` tokens are actually good
for.** The harness predicts slots from declaration order (metal's lowest-free-slot rule) and
passes them as named compile-time args. The kernel then verifies them on the COMPUTE
projection -- the one projection that binds every buffer, inputs as consumer and outputs as
producer -- with a `static_assert(kCbIn == uint32_t(dfb::in) && ...)`. So the tokens turn out
not to be useless to this model after all: they cannot *name* a buffer in a shared source
(§7.1), but they can *verify* one wherever they exist. Confirmed non-vacuous by perturbing the
prediction, which fails the build with the message naming the cause.

**Named compile-time args carried across unchanged.** `get_named_compile_time_arg_val` works
on both paths -- `named_ct_arg_map_generated.h` is emitted unconditionally
(`genfiles.cpp:120`) -- so only the accessor construction needed a `#if`. One kernel, four
guarded lines, both paths green. That is what makes the remaining suites a sweep rather than a
fork.

### 7.3 The second suite: semaphores, collectives, per-core named runtime args

`test_unified_mcast.py --metal2` passes on **8/8 configurations** -- rows of 2, 4 and 8, one
and two tiles, with and without `--barrier` -- with numbers identical to the legacy path on
every one, and `max |slice_i - slice_0| = 0` throughout, meaning the broadcast really reached
every core rather than the test agreeing with itself.

That covers three of §9's open items at once.

**§9.4 is answered: `sem::` bindings work, and the derived-id arithmetic is now CHECKED.**
`api.h` builds six semaphore ids from one base, which needs the reserved run to be contiguous
and to start where the harness predicted -- two facts only the host knows. The harness passes
the FIRST and LAST reserved names as `sem::` token *expressions* and `api.h` static_asserts
the arithmetic against them. Both ends together pin the whole run, since metal cannot issue a
duplicate id. Verified non-vacuous twice: perturbing the base and splitting the run each fail
the build, and each trips both assertions.

**Metal 2.0 refuses semaphore bindings on a compute kernel** (`program_spec.cpp:1088), which
was not anticipated anywhere in this document. It turns out to agree with the model rather
than fight it: `api.h` already says a `Semaphore` is projected onto one DM thread and is a
no-op elsewhere, and `impl_v1.hpp` keeps metal's `Semaphore` behind an `IS_DM_THREAD` guard,
so compute has never touched one. **The rule the model documented is now the rule the host
enforces** -- the same shape of win as the DFB endpoint roles, and again one the port
discovers rather than imports.

It does have a mechanical consequence: `sem::` tokens exist only on the two data-movement
kernels, so the check tokens are defined only there. Fine, because between them the two DM
projections check every reserved id, and all five projections see the same base.

**D17 is closed for this kernel, and that is the whole point of the exercise.** `mcast_bcast`
has a genuinely per-core runtime argument -- each core's output block index -- so it is the
first port to exercise the named runtime-arg path rather than just the compile-time one. The
legacy spelling is `get_arg_val<uint32_t>(2)` plus the sentinel; the 2.0 spelling is
`get_arg(args::out_block)`, and the two base addresses vanish entirely into the tensor
bindings. A name that is not in the schema, or one in the schema and not supplied, is an
error from metal rather than a garbage loop bound. The sentinel has nothing left to guard.

**`synchronize_cores()` survives the port untouched.** The `--barrier` configuration runs it
twice back to back on the reserved pair, which is the case that fails if the barrier clears
its arrival count in the wrong order, and it is bit-identical on both paths.

### 7.4 The wholesale port

Every kernel and every launcher is on the spec path. `unified_program()`, `make_cb()`,
`make_semaphore()` and `make_runtime_args()` are gone from the harness, and
`check_runtime_args` / `kRuntimeArgSentinel` are gone from `api.h` -- the sentinel existed
only to guard a positional runtime-argument list that no longer exists.

**A kernel now has one spelling for everything.** No `#if` selecting a path, no
`TensorAccessorArgs` offsets, no `get_arg_val` indices:

    constexpr uint32_t num_blocks = get_arg(args::num_blocks);   // compile-time
    constexpr uint32_t kCbIn      = get_arg(args::cb_in);        // compile-time
    const uint32_t     out_block  = get_arg(args::out_block);    // runtime
    const auto         in         = TensorAccessor(tensor::in);

That uniformity came from ttnn's own `redistribute_pages_row_major_reader.cpp`, which reads
its compile-time args through `get_arg` exactly as it reads its runtime ones. `kernel_args.h`
has a `constexpr get_arg(CtaVal<T>)` overload for it, and the goal is stated in that header:
moving an argument between CTA and RTA should be a host-side change with no kernel edit.

**Three defaults and one silent contract, found by porting rather than by reading.**

*Compute silently lost its optimisation level.* `KernelSpec::CompilerOptions::opt_level`
defaults to O2 for every kernel; the legacy path used O2 for data movement and **O3** for
compute (`kernel_types.hpp:82` against `:132`). For flash_attention that is not a slowdown but
a link failure -- constant propagation stops reaching `addr_mod_t::set()`'s inline-asm
immediate and LTO reports "impossible constraint in 'asm'". Every suite ported before that was
found had been building compute at O2 with nothing saying so, which would have been measured
as a model regression rather than a build one.

*A wrong endpoint role is silent on Gen1.* Binding a buffer's data-movement end to thread 0
while the kernel drives it on thread 1 runs, and passes, bit-identical: Gen1 circular-buffer
state is per core rather than per RISC, so either DM kernel can drive it whatever the host
declared. The masks only matter on Gen2. Since the kernel already states the thread in every
`noc_load<N>` / `noc_store<N>`, the harness now DERIVES the roles from the kernel rather than
having 24 launchers restate them -- removing the second end of the contract instead of trying
to check it.

*Metal 2.0 refuses semaphore bindings on compute kernels*, which agrees with what `api.h`
already documented (§7.3).

*A buffer the kernel DECLARES must be declared by the host*, even where no shape uses it.
This bit four times -- `matmul`'s bias, `reduction_tree`'s tmp1, `passcost`'s scratch chain,
`bcast`'s tmp -- so it is the rule, not a special case.

**What the port found in our own code**, all of it latent beforehand: two launchers whose
buffer NAMES had drifted from their kernels' (under the descriptor path the two sides only had
to agree on numbers); a buffer format that had to be carried explicitly, caught as 15 numeric
failures in `matmul_blocked`'s bfloat8 configurations, which is hazard D19 behaving exactly as
D19 says it does; and `passcost`'s scratch buffers, which had the old CB indices 1..7 hardcoded
in the kernel against a layout the launcher matched by hand.

**The real-time profiler needed one field, and the reason is worth knowing.** `bench()`
registered its callback, the callback reported itself ACTIVE, and no records arrived for
programs dispatched through `MakeMeshWorkloadFromSpec` + `EnqueueMeshWorkload`.

`ProgramImpl::runtime_id` is zero-initialised (`program_impl.hpp:522`), and zero is
`REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID` (`dispatch/kernels/realtime_profiler.hpp:23`).
The dispatch kernel gates the record FIFO on `program_host_id != UNPROFILED`
(`cq_dispatch.cpp:1432`), so a program nobody numbered is indistinguishable from one the host
deliberately excluded -- no error, no records, and a profiler that truthfully reports itself
active.

Nothing else assigns one. The only setter outside the profiler's own examples is ttnn's
device-operation path (`device_operation.hpp:186`), which every ttnn op passes through and a
program built straight from a ProgramSpec does not. **So this is a property of the 2.0 host
API, not of our shim: `MakeProgramFromSpec` and `MakeMeshWorkloadFromSpec` hand back a program
that is unprofilable until someone remembers to number it.** The shim now numbers every
program from `ttnn::CoreIDs`, the same counter ttnn uses, so ids stay unique across both.

`passcost` then passes in full, and its model is measurable again: ~0.86us per copy pass over
8 tiles, 0.148us per input tile for a reduce against 0.234us per input-plus-output tile, so
0.086us for the acquire and pack alone.

### 7.5 One open failure: mcast_share, and a methodology error that hid it

**`test_unified_mcast_share` hangs when kernel asserts are compiled in, and only then.**
Isolated to one variable, same device, same reset, nothing else changed:

    TT_METAL_HOME=$PWD                                          ./test_unified_mcast_share.py   8/8 configs ok
    TT_METAL_HOME=$PWD TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1     ./test_unified_mcast_share.py   hangs

Under the WATCHER -- asserts also on, different reporting -- it passes in 8s. That combination
is what `unified_api_hazards.md` already warns about under "the two assert modes are not
interchangeable": lightweight is an `ebreak` that halts the RISC and is indistinguishable from
a hang, so an assert firing and a deadlock look identical from the host. Which of the two this
is has not been established. mcast_share is also the suite built to be timing-sensitive --
handshake-pair sharing and deliberate skew -- so a build that changes kernel size changing its
behaviour is not surprising, and that cuts both ways: it could be a real assert, or a race the
watcher's timing avoids.

**Two earlier explanations were wrong, and both were mine.** The first was per-launch program
churn; caching the workload changed nothing. The second was a preceding multicast, from
bisecting that appeared to show `matmul_mcast + mcast_share` and `matmul_blocked +
mcast_share` failing where `custom_compute + mcast_share` passed. Re-run on a quiet machine,
`custom_compute + mcast_share` fails too -- and `mcast_share` ALONE through the runner fails.
There was never a prefix.

**The methodology error is worth more than the bug.** Every "passes standalone" run in this
session set `LIGHTWEIGHT_KERNEL_ASSERTS=1`, which metal does not read -- the variable is
`TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS`. So those runs had asserts OFF while appearing to have
them on, which is exactly why mcast_share looked fine standalone and broke under the runner,
and why two wrong theories survived as long as they did. The suite-level results stand,
because `run_unified_tests.sh` exports the right name; the individual spot-checks were weaker
than they looked.

## 8. What this buys, against the hazard ledger

| hazard | today | after |
|---|---|---|
| **D17** runtime-arg count/order — **3 device hangs** | positional, untyped; our sentinel catches *count*, never order | `get_arg(args::x)`, schema-validated per kernel |
| **D18** compile-time arg drift | offsets chained by hand | `TensorAccessor(tensor::name)` — no offsets at all |
| **D19** CB format vs tensor dtype | silent wrong bytes | `data_format_metadata` on the DFB spec, checked against the `TensorParameter` |
| **D20** CB collisions / undeclared Storage — **verified live** | caught only by our own capacity assert | **half of it.** A DFB with no producer or consumer is a host error, confirmed on hardware (§7.1); a kernel naming the WRONG slot is still unchecked, because slots stay values |
| the NOC-mode hang (noc spec §4) — **2 hangs, one tt-smi reset** | unchecked on the legacy path, by metal's own admission | `program_spec.cpp:906` names the fix in the exception |
| `NocTag` (noc spec §5) | 25 entry points of bespoke tag plumbing | `Noc`, already in metal |

Five hazard classes and eight recorded device hangs, against a library that today has a runtime-arg
sentinel and a set of asserts precisely because none of this was reachable.

---

## 9. Open questions

1. ~~**The gate (§7).**~~ **Answered: yes**, on hardware. See §7.1. It also settled the shape of
   the answer for buffer slots (values, never `dfb::` tokens) and cost the unified library exactly
   one additive line.
2. **Does a DFB tolerate a producer that never pops** (`fill_reduce_scaler`) and a buffer with only
   one live endpoint? §5.1.
3. **`role` carries a thread number the kernel also knows** (§5.1). Is a launch-time error good
   enough, or should the thread be derivable from one place? I think it is good enough and it is
   still an improvement on today, but it is a new two-places-must-agree contract in a project whose
   hazard list is mostly those.
4. **Does the multicast handshake survive 2.0's semaphore binding?** The class is unchanged, but
   ids come from `sem::name`, and `api.h` derives six ids arithmetically from one base
   (`kMcastSemBase + 2*thread`). Named bindings want six names; the arithmetic wants a base. UNVERIFIED
   which the 2.0 path permits.
5. **Is the `Noc` object as cheap as the free function?** It holds one `uint8_t` and every method
   forwards, so it should vanish under `-O2` — but the noc spec measured dynamic-NOC mode at +2.7%
   on a claim that also looked free, so this deserves a measurement rather than an argument.
6. **Should stage 1 happen at all if stage 2 is not funded?** §6 Option C's honest weakness. My
   answer is yes — the object model is where metal is going and the free functions will bit-rot —
   but it is a judgement call, not a finding.
