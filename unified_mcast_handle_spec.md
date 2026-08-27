# Spec: give the multicast load a deferred handle

Written against `origin/main` at `9aa797dec5e`.

**STATUS: step 1 done -- `NocAsyncMcastTx` exists and carries the state, behaviour
unchanged. Steps 2 to 4 (the assert, the deferral, the measurement) NOT done, held at the
author's request pending a different proposal for the detection half.**

The ask: make the multicast `noc_load` return a handle in the shape of
`NocAsyncWriteCoreTx` -- carrying the `data_sent` semaphore and a role flag -- with the
receiver's `data_sent.wait(1); set(0)` sunk out of `noc_load` and into `.wait()`. Then a
global per-semaphore in-flight bool, assertion-only, catches two operations live on one
semaphore.

---

## 1. The finding that reframes it

**The in-flight assert is not a bug detector here. It is the PRECONDITION that makes the
deferral sound.**

Today the receiver's flag wait happens inside `noc_load`, so when `noc_load` returns both
semaphores read 0 on every core. That invariant is load-bearing and the two `noc_async_writes_flushed()`
calls in the sender exist to protect it -- the code says so:

    // Back to 0 so BOTH semaphores read 0 on every core once this returns.

Defer the receiver's clear and that invariant is gone. Between `noc_load` returning and
`.wait()` being called, the receiver's flag is still 1. If the sender starts round b+1 in
that window it writes 1 into a flag that is already 1, and the receiver cannot tell round b
from round b+1: it clears once, round b+1's flag is lost, and the receiver waits forever.
A hang, one round late, which is the worst place to look for it.

`NocAsyncWriteCoreTx` **already documents this exact defect** in its own `wait()`:

    // Clearing after the count is a window: a writer already into the
    // next round has its increment erased here, hanging the round after.
    // Repeated pushes need the caller to keep the rounds apart --
    // synchronize_cores() between them is enough.

That answer -- barrier between rounds -- is affordable for `noc_core_write` (reduction_tree
barriers once per gather) and **is not affordable for a multicast load inside a k-loop**.

So the deferral needs one of:

- **(a) the caller never has two operations live on one pair**, which is exactly what the
  in-flight assert enforces, or
- **(b) a monotonic flag** -- an incrementing counter the receiver compares against a block
  number, so the sender never rewrites the word it just multicast. This is the protocol
  change already identified in `unified_llama_prefill.md`, built once and reverted for API
  churn with no perf gain at the time.

**Recommend (a).** It is cheap, it is the assert being asked for anyway, and it turns a
silent precondition into a checked one. (b) becomes necessary only if we later want two
rounds of the SAME collective in flight, which (a) forbids.

## 2. What the deferral actually buys, and what it does not

**Not** pipelining two rounds of one broadcast -- (a) forbids that by construction.

**What it does buy**: overlapping one collective's wait with another collective's work. The
two operands are already on different pairs, so this is legal under (a):

```cpp
// now: each handshake completes before the next begins
ComputeBlock a = noc_load<0, 0>(a_storage, row, a_acc, idx).wait();
ComputeBlock w = noc_load<1, 1>(w_storage, col, b_acc, idx).wait();

// after: A's flag wait overlaps B's entire handshake
auto ta = noc_load<0, 0>(a_storage, row, a_acc, idx);
auto tb = noc_load<1, 1>(w_storage, col, b_acc, idx);
ComputeBlock a = ta.wait();
ComputeBlock w = tb.wait();
```

**Receivers only.** The sender's `receivers_ready.wait(num_dests)` gates its broadcast and
cannot be deferred -- it must not multicast into a buffer nobody has freed, which is hazard
13b. So the sender stays fully synchronous and gains nothing here. That is not a small
caveat: in an 8x8 grid 7 of 8 cores are receivers for each collective, so the population
that benefits is most of the grid, but the critical path through the SENDER is untouched.

Which means **this is not the 17% the broadcast-deletion ablation bounded.** That 54us / 17%
was the cost of the whole broadcast including the sender's send. This change hides part of
the receivers' waiting behind the other operand's handshake. The honest expectation is
"some of the receiver-side wait", unmeasured, and it should be measured before the change is
justified on performance grounds rather than on shape.

## 3. A new handle type, not `NocAsyncWriteCoreTx`

Reusing it looks tempting and is wrong on two counts:

| | `NocAsyncWriteCoreTx` | what a multicast load needs |
|---|---|---|
| `wait()` signature | `wait(uint32_t num_writers)` | no argument -- the flag wait is always `wait(1)` |
| `src_cb` member | the block being pushed | meaningless; the payload arrives from DRAM into our own CB |
| semaphore | `arrived`, counted by N writers | `data_sent`, a 0/1 flag from one sender |

The `wait(num_writers)` argument is the decisive one: reusing the type **breaks all eleven
existing call sites**, every one of which is `noc_load(...).wait()`.

So: `NocAsyncMcastTx<thread, S>`, carrying

- `cb_id`, `num_pages` -- as `NocAsyncReadTx` does
- `Semaphore<thread> data_sent` -- mutable, as `NocAsyncWriteCoreTx`'s `arrived` is
- `bool sender` -- the role, decided at construction from the coordinate
- `waited` -- the assertion-only flag `NocAsyncReadTx` already carries, so a dropped handle
  still asserts

and `Block<S> wait() const` doing:

```
if (!sender) { data_sent.wait(1).set(0); }   // the part that moved out of noc_load
cb_push_back(cb_id, num_pages);
```

The sender's read barrier stays INSIDE `noc_load`: it broadcasts from its own L1, so the
payload must have landed before the multicast, not before `.wait()`.

**Every existing call site keeps working unchanged**, because `.wait()` stays no-argument
and still returns `Block<S>`. Call sites only change where a caller wants the overlap.

## 4. The in-flight assert

Assertion-only, per data-movement thread, indexed by semaphore id:

```
construction of NocAsyncMcastTx:  ASSERT(!in_flight[sem]); in_flight[sem] = true;
.wait():                          in_flight[sem] = false;
```

A `static bool` array in a function returning a reference, which is the pattern `pack_to`
already uses successfully under `-ftt-no-dyninit`. Index by `sem_id - kMcastSemBase` so the
array is six entries.

What it catches: two multicast operations live on one pair. Which is (a) above, so it is
enforcing the precondition rather than merely reporting violations of it.

**What it does NOT catch, and this must not be oversold: hazard 13b.** That bug has no core
with two operations in flight -- every core runs its two collectives strictly sequentially,
each fully waited -- and the damage lands on a different core. The rectangle-claim check
proposed in `unified_api_hazards.md` 13b is the one that catches that. These are two
different checks for two different failures and neither subsumes the other.

## 5. Scope

- **6 overloads** change return type: PhysicalMcast and LogicalMcast, each in accessor form,
  Fn form, and explicit-semaphore form.
- **11 call sites** across `example_matmul`, `flash_attention`, `matmul_mcast`,
  `mcast_bcast`, `matmul_blocked` -- all compile unchanged.
- **1 new type** in api.h + impl_v1.hpp; `NocAsyncReadTx` untouched, since the non-multicast
  loads keep using it.
- The `receivers_ready.inc_remote()` on the receiver side stays in `noc_load`: it means "my
  buffer is free", which is true at that moment and not at `.wait()`.

## 6. Order of work

1. **DONE.** `NocAsyncMcastTx` with the receiver wait still inside `noc_load` -- pure
   refactor, the handle does nothing new. Checkpoint met: 19 suites passed, 0 failed, and
   the multicast users (matmul_mcast, mcast, matmul_blocked, attention_proj, flash,
   example_matmul) verified individually including with asserts on.

   Two things the implementation needed that the spec had not called out. `Semaphore::id`
   was private with no accessor, and a reference would not do: every pair-derived multicast
   builds its two semaphores as LOCALS inside `noc_load`, so the handle has to carry the id
   -- hence a new `semaphore_id()` accessor. And the multicast form used to DELEGATE to the
   plain `Fn` form, inheriting `NocAsyncReadTx` along with it, so the producing half had to
   be extracted into `detail::issue_load` for the multicast form to run the same protocol
   and return a different handle.
2. The in-flight assert. Still nothing to catch, but it is in place before the window opens.
3. Sink the receiver's flag wait into `.wait()`. NOW the assert is load-bearing. Suite green
   with asserts on is the checkpoint.
4. Change one kernel -- `matmul_blocked` -- to issue both operands before waiting either,
   and MEASURE. If it does not pay, stop here: steps 1-3 are still worth having for the
   shape and the assert, and no kernel has to adopt the overlap.

## 7. Open questions

1. **Is the receiver's deferred window actually reachable in our kernels?** Under (a) a
   caller cannot have two live handles on one pair, and every current call site waits
   immediately. So the window only opens where a kernel deliberately opens it. That makes
   step 3 low-risk -- but it also means the assert cannot fire on any code we have today,
   so it is untestable without writing a kernel that deliberately violates it. That test
   should be written as part of step 2, not after.
2. **Does the sender want a handle at all?** Its work is complete when `noc_load` returns,
   so its `.wait()` is only the push. Keeping one type for both roles is simpler than two
   and matches how `NocAsyncWriteCoreTx` handles its reader/writer split.
3. **Should `noc_core_write` get the same in-flight assert?** It has the identical documented
   window and the identical "barrier between rounds" requirement, and `reduction_tree`
   relies on getting that right by hand. Same mechanism, so nearly free once built -- but it
   is a second change and should not be smuggled into this one.
