# Proposal: expose the base API's `noc` on the unified NOC APIs

Proposal, not implementation. Written against `origin/main` at `9aa797dec5e`, updated with
what the throwaway ablation found.

## 1. The ask

Metal's dataflow layer gives every NOC call an optional trailing NOC:

    void noc_async_read(uint64_t src, uint32_t dst, uint32_t len, uint8_t noc = noc_index);
    void noc_async_read_barrier(uint8_t noc = noc_index);

**Our wrappers drop that parameter.** The unified API should not be less expressive than the
API it wraps. This proposal restores it.

That is the whole justification. It is deliberately *not* "because configuration X is
faster" -- the ablation below shows the perf case is unproven and the mode it would need
costs 2.7%. Parity is the reason: a caller who can express something in `dataflow_api.h`
should be able to express it through us, and where we take that away we should be taking it
away for a stated reason, not by omission.

What parity unlocks, without our having to predict which pays:

- a thread reading on one NOC and writing on the other
- both DM threads issuing on the better NOC for a given direction
- multicast on NOC 0 specifically, rather than "whichever NOC thread N is welded to"

## 2. What the ablation established

Two rows, on `[512,2048]@[2048,2048]`, 8x8 multicast, kt=8 depth=3:

| configuration | median |
|---|---|
| baseline, A on thread 1 / NOC 1, B on thread 0 / NOC 0 | **153.8 us** |
| same shape, `DM_DYNAMIC_NOC` | 157.9 us |

So **dynamic mode costs ~2.7% before buying anything.** The case the original spec was
written to reach -- both reads on NOC 0 -- did not complete: it produced no output for
three minutes and was stopped, which is **inconclusive between a hang and a slow
compile** and is recorded here as unmeasured rather than as a result.

The load-bearing discovery is not a timing at all. It is §4.

## 3. Constraint one: barriers are per-NOC, so handles must carry it

    void noc_async_read_barrier(uint8_t noc = noc_index);
    void noc_async_writes_flushed(uint8_t noc = noc_index);

**Every handle in this library barriers in `wait()`.** If a read is issued on NOC 1 and the
barrier runs on NOC 0, the barrier returns immediately, `cb_push_back` publishes pages that
have not landed, and the consumer reads garbage. No hang, no assert.

So the parameter is not a pass-through. **Each of the five handle types has to carry the NOC
it was issued on**, and its `wait()` has to barrier on that one:

| handle | what its wait() barriers |
|---|---|
| `NocAsyncReadTx` | `noc_async_read_barrier` |
| `NocAsyncWriteTx` | `noc_async_write_barrier` |
| `NocAsyncMcastTx` | read barrier |
| `NocAsyncReadCoreTx` | read barrier |
| `NocAsyncWriteCoreTx` | writes flushed |

This is the largest part of the change and the one that fails silently if missed. It is also
the reason to do it as one change rather than API-by-API: a half-threaded `noc` is worse than
none, because it looks available and is wrong.

## 4. Constraint two: barrier accounting is per-RISC

**Under the NOC mode we build with, two RISCs cannot both issue reads on one NOC.** Not
"slowly" -- at all. The read barrier is:

    inline bool ncrisc_noc_reads_flushed(uint32_t noc) {
        return (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]);
    }

A **per-core hardware** response register compared against **`noc_reads_num_issued`, which
is a plain global defined once per RISC binary** (`brisc.cc:67`, `ncrisc.cc:36`, `drisc.cc:20`
each declare their own). So the hardware side counts every response that NOC delivered to
the core, and the software side counts only what *this* RISC issued. If BRISC issues 100
reads on NOC 0 and NCRISC then barriers NOC 0, NCRISC compares 100 against its own 0 and
**spins forever.**

This is not a subtlety we could work around in our own barrier bookkeeping; the counter is
metal's and the register is the hardware's.

**The thread-NOC welding this spec set out to break is therefore not an arbitrary
convention. It is the precondition that makes the barriers correct.** `unified_harness.py`
already records the same fact from the opposite direction, in the comment explaining why
`noc` must be set explicitly on the reader descriptor:

    // NOC_INDEX is emitted from this field (kernel.cpp:252), so BRISC would check
    // NOC 0's read counters against its own issued count and trip on reads NCRISC issued.

**Confirmed on hardware**, not just read off the source: the first ablation attempt hung on
exactly the configuration that has one RISC barrier a NOC the other RISC issued on.

### Metal states the rule itself, and only checks it on the 2.0 path

`tt_metal/impl/metal2_host_api/program_spec.cpp:905` validates exactly this and says why:

    "both are dedicated-NOC data movement kernels pinned to NOC_{}, which HANGS THE
     DEVICE. Give them distinct NOCs, or use DM_DYNAMIC_NOC mode to intentionally
     share a NOC."

and immediately above it, the constraint that settles whether `noc` can be per-call:

    "All data movement kernels on a node must use the same NOC mode -- it configures
     shared per-core NOC hardware."

**The mode is per-core, agreed across every DM kernel on it.** So it can never be a
per-call argument; the most it can be is a program-level choice that per-call `noc`
arguments are then legal under.

Two further things fall out of that file:

- It explains why we got a **silent hang instead of an exception**. The same comment records
  that "the legacy `CheckDataMovementConfig` intended this check but did not reliably
  enforce it for the common reader+writer pair (it runs before the second DM kernel is
  registered)". Our harness uses the legacy `ProgramDescriptor` path, so nothing checked it.
  This is an argument for the metal 2.0 migration in `unified_named_args_spec.md` that is
  independent of argument passing: **the 2.0 path would have turned this ablation's hang
  into a diagnostic.**
- It confirms the welding is deliberate: KernelGroup finalize "writes `brisc_noc_id = arg.noc`
  for RISCV_0 vs `1 - arg.noc` for RISCV_1, which agree only when the two NOCs differ".

### What unblocks it: `DM_DYNAMIC_NOC`

Metal has a second mode, and it exists for precisely this:

    inline bool ncrisc_dynamic_noc_reads_flushed(uint32_t noc) {
        uint32_t status_reg_val  = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
        uint32_t self_risc_acked  = get_noc_counter_val<proc_type,     READS_NUM_ISSUED>(noc);
        uint32_t other_risc_acked = get_noc_counter_val<1 - proc_type, READS_NUM_ISSUED>(noc);
        return (status_reg_val == (self_risc_acked + other_risc_acked));
    }

The counters move to shared L1 and the barrier sums **both** RISCs'. That is the mode in
which "two RISCs on NOC 0" is even expressible. It is per-kernel host state, not a per-call
argument -- `DataMovementConfigDescriptor::noc_mode` -- and it is exposed to Python as
`ttnn.NOC_MODE.DM_DYNAMIC_NOC`, so our harness can set it.

This shapes the design in three ways:

1. **`noc_mode` is program-scoped, not call-scoped**, so it cannot be part of the parameter
   this proposal adds -- it has to be part of how the program is built (§7). And a foreign
   `noc` in a dedicated-mode program is a **hang**, worse than §3's silent wrong barrier.
   Left unchecked that would make the new parameter a footgun; §6 makes it a build error,
   which is the single most valuable piece of this proposal.
2. **Dynamic mode is not free: measured at +2.7%** (§2), paid on every call whether or not
   any call uses a foreign NOC. So it must stay opt-in per program, never the default.
3. **ttnn does not use it.** `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2562`
   hardcodes `bool use_dedicated_noc = true`, having clearly considered the choice. Its
   answer for NOC preference is to pick the better NOC *per kernel* -- `preferred_noc_for_dram_read`
   for the weights reader, `preferred_noc_for_dram_write` for the writer -- which is exactly
   what our thread defaults already do. The deepseek_v3_b1 unit tests do use dynamic mode,
   so it is live, but the flagship matmul declines it.

## 5. The shape of the parameter: a typed tag, not a `uint8_t`

Metal spells it `uint8_t noc = noc_index`. **We cannot copy that spelling, and the reason is
concrete rather than stylistic: it breaks existing call sites.** Verified by compiling both
shapes, not reasoned about -- see the probe in §5.1.

Today, for one pair of `thread`/`pair` template arguments:

    noc_load(const Storage<S>&, PhysicalMcast, const Accessor&, uint32_t block_idx);  // 4 args
    noc_load(const Storage<S>&, PhysicalMcast, Fn fn);                                // 3 args

Append `uint8_t noc = noc_index` to both and the `Fn` form becomes 4 arguments of shape
`(Storage, Mcast, deduced, integral)` -- which is exactly the accessor form's shape. Both
`Accessor` and `Fn` are deduced template parameters, so both candidates match and the call
is ambiguous:

    noc_load<0>(st, mc, fn,  1);   // Fn form with noc=1, or accessor form with block_idx=1?
    noc_load<0>(st, mc, acc, 5);   // accessor form with block_idx=5, or Fn form with noc=5?

Both are ambiguous under gcc 20 with `-std=c++20`. **The second line is the serious one: it
is what every current multicast call site already looks like**, so appending a `uint8_t`
would not merely make new spellings ambiguous, it would stop today's kernels compiling. My
first probe missed this because it wrote `5u` and `uint8_t(1)`, whose exact matches break
the tie; with the bare `int` literals callers actually write, both calls fail.

A distinct type removes the ambiguity by construction, and it also answers the original
spec's open question 2 (bare positional literals, the complaint already logged against
`pair` in hazard 13b):

```cpp
template <uint8_t N>
struct NocTag {
    static constexpr uint8_t value = N;
};

inline constexpr NocTag<0> noc0{};
inline constexpr NocTag<1> noc1{};
```

At the call site:

```cpp
auto tx = noc_load<0>(a_storage, row, a_acc, idx, u::noc0);
noc_store<1>(out_storage.store(blk), out, idx, u::noc1);
```

Three properties worth having, all of which a `uint8_t` lacks:

1. **No ambiguity.** `NocTag<N>` is not constructible from an integer literal, so it can
   never be mistaken for a block index and a block index can never be mistaken for it.
2. **Not transposable.** `noc_load<0, 1>(...)` today is thread 0 / pair 1 and reads
   identically to thread 1 / pair 0. `u::noc1` cannot be confused with either.
3. **The value is compile-time**, which is what makes §6 a build error instead of a hang.

### 5.1 Verified, not assumed

Both halves of this section were compiled before being proposed. With the tag, all four
combinations resolve to the intended overload and **existing call sites are untouched**:

    existing accessor call      -> accessor form
    existing Fn call            -> Fn form
    accessor + explicit noc1    -> accessor form
    Fn + explicit noc1          -> Fn form

and with `noc_mode = DM_DEDICATED_NOC`, the foreign-NOC call fails at compile time with the
message that names the fix:

    error: static assertion failed: requesting a NOC other than this thread's
    requires DM_DYNAMIC_NOC (unified_program(dynamic_noc=True))

The probes are `scratchpad/noc_tag_probe.cpp` and `noc_assert_probe.cpp`. They are host
C++ standing in for the kernel's `noc_index`/`noc_mode` constants, so they prove the
overload resolution and the assert, not the device behaviour.

A runtime NOC (`NocTag` cannot express `noc = f(x)`) is deliberately not offered. No caller
we have wants one, and a runtime value downgrades §6's `static_assert` to an `ASSERT` that
only fires in assert builds -- which, per the lightweight-assert finding, means an `ebreak`
the host cannot distinguish from a hang. If a caller ever needs it, add it then, with the
runtime check, as a separately-named entry point.

## 6. The rule that makes it safe: `noc != noc_index` requires dynamic mode

§4 gives one rule, and it is checkable:

> Issuing on a NOC that is not this thread's own is sound only in a `DM_DYNAMIC_NOC`
> program.

Under `DM_DEDICATED_NOC` the *other* DM thread is pinned to the NOC you are reaching for,
so the moment both threads touch it the per-RISC counters and the per-core register
disagree and a barrier spins forever. Metal's own 2.0 validator says as much ("which hangs
the device").

Both sides of that test are compile-time constants in kernel scope:

    tt_metal/hw/inc/internal/dataflow/dataflow_api_common.h:10:  constexpr uint8_t noc_index = NOC_INDEX;
    tt_metal/hw/inc/internal/dataflow/dataflow_api_common.h:11:  constexpr uint8_t noc_mode  = NOC_MODE;

both emitted per kernel by `kernel.cpp:260-261`. So with the compile-time tag of §4:

```cpp
static_assert(
    N == noc_index || noc_mode == DM_DYNAMIC_NOC,
    "requesting a NOC other than this thread's requires the program to be built with "
    "DM_DYNAMIC_NOC (unified_program(dynamic_noc=True)); in DM_DEDICATED_NOC the other "
    "data-movement thread owns that NOC and the read barrier will never complete");
```

**This is the part of the proposal worth the most.** It converts the failure mode from a
silent device hang -- which cost this investigation two hangs and a tt-smi reset, and which
metal's legacy path does not check -- into a build error naming the fix. It is also why the
tag carries its value as a template parameter rather than a member.

Note what it does *not* claim: it does not verify that dynamic mode is *sufficient*, only
that dedicated mode is *insufficient*. Whether our multicast handshake works under dynamic
mode is the open question in §13.1.

### The narrow legal case it forbids

A kernel where only ONE data-movement thread issues anything can use both NOCs freely under
dedicated mode -- there is no second RISC to disagree with. The `static_assert` rejects it
anyway, because a kernel cannot see whether the other thread is idle.

That is the right trade. The check is conservative in the direction of refusing something
legal rather than admitting something that hangs, and the workaround is one host flag.

## 7. The host side, already built

`noc_mode` is per-core state that every DM kernel on the core must agree on (§4), so it
belongs to the program, not the call. `unified_harness.py` now takes it:

```python
program = unified_program(..., dynamic_noc=True)
```

which sets `noc_mode` on both `DataMovementConfigDescriptor`s. This is the one piece of the
proposal that is implemented, because the ablation needed it; it is useful on its own and
independent of everything else here.

## 8. The custom `Fn` contract changes, and it must change LOUDLY

The Fn forms hand the routine `L1Pages` and let it issue the traffic. Today's contract:

    // `fn` must issue ONLY READS, and only on this thread's assigned NOC.

With an explicit noc that becomes "on the NOC the handle was given", and the routine has to
know which. Two ways:

- **(a) `L1Pages` gains a `noc` field.** Every existing routine keeps compiling. That is the
  problem: a routine that ignores the field issues on `noc_index` while the handle barriers on
  the requested NOC. **Silently wrong**, in the exact failure mode of §3.
- **(b) `fn` takes it as a second parameter** -- `fn(L1Pages pages, uint8_t noc)`. Every one of
  the **12 routines** in `matmul_blocked` (9), `eltwise_add_exp` (2) and `flash_attention` (1)
  fails to compile until updated.

**Recommend (b).** The failure mode of (a) is precisely the class this project keeps getting
bitten by -- a contract with nothing checking it -- and 12 mechanical edits is a cheap price
for a compile error instead. (a) is defensible only if the noc is also plumbed into a
`pages.read(...)` helper so routines never name the intrinsic themselves, which is a bigger
redesign.

## 9. A multicast is one NOC for the whole collective

The sender's payload multicast, its flag multicast, and the receivers' semaphore increments
must all be on one NOC. The reason is in the code already:

    // ttnn's matmul sender does NOT flush here: its payload and flag multicasts go
    // out on the same NOC, VC and command buffer (NOC_CMD_STATIC_VC), so they cannot
    // reorder

Split them across NOCs and that ordering argument evaporates -- and ours already needs two
flushes it would rather not have. So the multicast forms take **one** `noc` for the collective,
not one per operation, and the handshake inherits it. A `static_assert` or an assert that a
receiver and its sender agree is not possible from one core, so this is a documented
requirement, not a checked one.

## 10. What this does NOT change

**The handshake pair still has to be distinct per rectangle.** Pairs are indexed by THREAD
(`kMcastReadySem<thread>`), not by NOC, so two collectives on one thread with different NOCs
still share a pair -- and hazard 13b still applies. Decoupling thread from NOC makes that
coupling *less* obvious, not more, so the `pair` documentation needs a line saying the NOC is
irrelevant to it.

Nor does it change which RISC executes: `thread` still selects that, and the
`if constexpr (thread == TT_DM_THREAD_ID)` gating is untouched. The change is only about which
wire the traffic uses.

## 11. Scope

| | count |
|---|---|
| `noc_load` overloads | 8 |
| `noc_store` | 2 |
| `noc_core_read` / `noc_core_write` | 6 |
| `synchronize_cores` | 3 |
| `Semaphore` remote ops (`inc_remote`, `inc_mcast`, `set_mcast`) | 6 |
| handle types needing a `noc` member | 5 |
| custom `Fn` routines in kernels | 12 |

25 entry points. The underlying metal calls all take the parameter already, and the
compute-projection stubs in `adaptor_v1.hpp` already carry a trailing `uint8_t`, so the
signatures line up with no shim work:

    inline void noc_async_read(std::uint64_t, uint32_t, uint32_t, uint8_t = 0) { ASSERT(false); }

## 12. Order of work

Each step ends with the full suite unchanged, since everything is defaulted.

1. **Handles carry the NOC.** Add the member, thread it into every `wait()` barrier,
   default `noc_index`. No API change yet. This is the step that silently breaks things if
   half-done (§3), so it goes first and alone.
2. **`NocTag` + the `static_assert`**, threaded through the 25 entry points, defaulted to
   `NocTag<noc_index>`. Nothing can request a foreign NOC yet except deliberately, and when
   it does the assert either passes or explains itself at compile time.
3. **`fn(L1Pages, uint8_t noc)`** and the 12 routines updated (§8). Mechanical, and a
   routine that ignores the NOC stops compiling rather than silently mis-barriering.
4. ~~**Answer §13.1** -- does the multicast handshake survive dynamic mode?~~ **Done, yes**
   (§13.1). It no longer gates step 2.
5. **Only then** revisit whether two RISCs on one NOC beats 153.8us. It is no longer the
   justification, so a null result costs nothing.

## 13. Open questions

1. ~~**Does our multicast handshake work under `DM_DYNAMIC_NOC` at all?**~~ **Answered:
   yes.** See §13.1 below; this is no longer open.
2. **Should the writes get the same treatment in the same change?** §3's silent-wrong-barrier
   argument applies identically to `noc_async_write_barrier`, and step 1 covers all five
   handle types, so it is already in scope. Worth stating explicitly so it does not get
   dropped as "reads were the point".
3. **Does the tag belong on `Storage` instead of the call?** Every current kernel would use
   one NOC per buffer for its whole life, so `Storage<S> a_storage(kCbA, u::noc0)` would be
   less repetitive than tagging each load. Against it: the NOC is a property of a transfer,
   not of a buffer, and putting it on the buffer makes a load's barrier depend on state
   declared far away. Recommend the call site, but the buffer form is worth a look if the
   call sites get noisy.
4. **`preferred_noc_for_dram_read` / `preferred_noc_for_dram_write`.** ttnn picks its NOC
   per kernel through these helpers rather than hardcoding 0 and 1, which is how it stays
   correct across architectures. If we are going to name NOCs explicitly, our defaults
   should come from the same place rather than from `MMB_IN0_THREAD`-style conventions.

### 13.1 Answered: the handshake is unaffected by dynamic mode

The worry was that under `DM_DYNAMIC_NOC` the flush predicates sum **both** RISCs'
counters, so the handshake's two `noc_async_writes_flushed()` calls wait on the other
thread's writes as well as their own -- over-waiting that ought to be slow rather than
wrong, but had nothing checking it. The ablation's NOC-0 case had stalled without a
verdict, so this could not be assumed.

`test_unified_mcast_share.py` now runs its whole matrix under both modes. Under
`DM_DYNAMIC_NOC`, **32/32 configurations are bit-exact** -- rounds 8 and 16, skew 0 / 5k /
50k / 200k, shared and distinct handshake pairs, two seeds each -- which is the same
coverage that proved the shared-pair result in `a27db04bbce`. Nothing hung, and the
outputs are exact rather than close, so a mis-ordered flag or a stale buffer would have
shown as a NaN slice or a wrong block, not as noise.

**What this does and does not establish.** It establishes that dynamic mode alone does not
break the collective: the counters moving to shared L1 and the flushes over-waiting are
survivable. It does **not** establish that a *foreign*-NOC multicast works, because
`mcast_share.cpp` still issues everything on `noc_index` -- the tag of §5 does not exist
yet, so there is no way to ask for anything else. That second half only becomes testable
after step 2, and §9's rule (one NOC for the whole collective, sender and receivers alike)
is still a documented requirement no test has exercised.

So step 2 is unblocked, and the residual risk it carries is §9's, not §13.1's.
