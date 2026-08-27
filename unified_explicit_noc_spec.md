# Spec: an explicit `noc` on the unified NOC APIs

Proposal, not implementation. Written against `origin/main` at `9aa797dec5e`.

The ask: give every NOC-generating unified API an optional trailing `uint8_t noc = noc_index`,
matching what metal's dataflow layer already does.

---

## 1. Why this is worth doing

Because **the two NOCs are not interchangeable and today we can only choose one by choosing a
thread.** Measured on the blocked matmul, `[512,2048]@[2048,2048]`, 8x8 multicast:

| A's thread / NOC | B's thread / NOC | |
|---|---|---|
| 1 | 0 | **155.6 us** |
| 0 | 0 | 216.2 us |
| 0 | 1 | 308.3 us |
| 1 | 1 | 403.5 us |

A 2.6x spread on the same reads. NOC 0 is much the better one for DRAM traffic, and the big
operand belongs on it -- which is why the default is A on thread 1 and B on thread 0.

But `thread` and `noc` are welded together: thread N issues on NOC N. So "put both operands on
NOC 0" is only expressible as "put both on thread 0", which serialises them on one RISC --
that is the 216.2us row, worse than the 155.6 we settled for. **The configuration we cannot
currently ask for is two RISCs both issuing on NOC 0**, and on this evidence it is the one most
likely to win. That hypothesis is untestable today, and making it testable is the point.

Secondary benefits: a thread could read on the fast NOC and write on the other, and the ttsim
multicast restriction (no multicast on NOC 1) becomes a per-call fact rather than a
whole-thread one.

## 2. The critical constraint: barriers are per-NOC

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

## 3. The custom `Fn` contract changes, and it must change LOUDLY

The Fn forms hand the routine `L1Pages` and let it issue the traffic. Today's contract:

    // `fn` must issue ONLY READS, and only on this thread's assigned NOC.

With an explicit noc that becomes "on the NOC the handle was given", and the routine has to
know which. Two ways:

- **(a) `L1Pages` gains a `noc` field.** Every existing routine keeps compiling. That is the
  problem: a routine that ignores the field issues on `noc_index` while the handle barriers on
  the requested NOC. **Silently wrong**, in the exact failure mode of §2.
- **(b) `fn` takes it as a second parameter** -- `fn(L1Pages pages, uint8_t noc)`. Every one of
  the **12 routines** in `matmul_blocked` (9), `eltwise_add_exp` (2) and `flash_attention` (1)
  fails to compile until updated.

**Recommend (b).** The failure mode of (a) is precisely the class this project keeps getting
bitten by -- a contract with nothing checking it -- and 12 mechanical edits is a cheap price
for a compile error instead. (a) is defensible only if the noc is also plumbed into a
`pages.read(...)` helper so routines never name the intrinsic themselves, which is a bigger
redesign.

## 4. A multicast is one NOC for the whole collective

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

## 5. What this does NOT change

**The handshake pair still has to be distinct per rectangle.** Pairs are indexed by THREAD
(`kMcastReadySem<thread>`), not by NOC, so two collectives on one thread with different NOCs
still share a pair -- and hazard 13b still applies. Decoupling thread from NOC makes that
coupling *less* obvious, not more, so the `pair` documentation needs a line saying the NOC is
irrelevant to it.

Nor does it change which RISC executes: `thread` still selects that, and the
`if constexpr (thread == TT_DM_THREAD_ID)` gating is untouched. The change is only about which
wire the traffic uses.

## 6. Scope

| | count |
|---|---|
| `noc_load` overloads | 8 |
| `noc_store` | 2 |
| `noc_core_read` / `noc_core_write` | 6 |
| `synchronize_cores` | 3 |
| `Semaphore` remote ops (`inc_remote`, `inc_mcast`, `set_mcast`) | 6 |
| handle types needing a `noc` member | 5 |
| custom `Fn` routines in kernels | 12 |

25 entry points. All of the underlying metal calls already take the parameter, and our
compute-projection stubs in `adaptor_v1.hpp` already have the trailing `uint8_t` -- so the
signatures line up and no shim work is needed:

    inline void noc_async_read(std::uint64_t, uint32_t, uint32_t, uint8_t = 0) { ASSERT(false); }

## 7. Order of work

1. **Handles carry the NOC.** Add the member, thread it into every `wait()` barrier, default
   `noc_index`. No API change yet, so the checkpoint is that all 19 suites are unchanged.
2. **`fn(L1Pages, uint8_t noc)`**, and update the 12 routines to pass it through. Still all
   defaults, so the checkpoint is again an unchanged suite -- but now a routine cannot silently
   ignore the NOC.
3. **Thread the parameter through the 25 entry points**, defaulted. Suite unchanged.
4. **Measure the thing this exists for**: `matmul_blocked` with A on thread 1 / NOC 0 and B on
   thread 0 / NOC 0, against the current 155.6us best. If it does not beat it, steps 1-3 are
   still worth keeping for the read-on-one-NOC-write-on-the-other case, but the headline
   justification is gone and the spec should say so.

## 8. Open questions

1. **Does a second RISC issuing on NOC 0 actually help, or does it just contend?** The whole
   perf case rests on this and it is unmeasured. Step 4 answers it, and it could be answered
   *first* with a throwaway hack -- one kernel, hardcoded `noc` arguments on the reads, no API
   change -- before committing to 25 signatures. **I would do that ablation first.**
2. **Is `uint8_t` the right type, or should it be an enum?** `noc_load<thread>(..., 1)` is a
   bare literal at a call site, which is the shape of complaint already logged against the
   `pair` parameter (hazard 13b): a small integer whose meaning is positional. A
   `Noc::First`/`Noc::Second` enum costs nothing and cannot be transposed with `thread` or a
   block index.
3. **Should the default be `noc_index` or the thread's NOC?** They are the same thing today,
   but spelling it `noc_index` inherits metal's global, while spelling it `thread` makes the
   coupling explicit and survives a future where a thread's default NOC changes. Prefer the
   latter.
4. **Does anything assume traffic and semaphores share a NOC?** §4 says a collective must, but
   `noc_core_write`'s arrival flag and `synchronize_cores` reuse the multicast semaphores, and
   those paths would need auditing rather than assuming.
