# Unified API: ways to misuse it that are hard to detect

A working list, for turning into negative tests and then into constraints. Nothing here is
a bug report about a kernel that exists -- it is an audit of what the API *permits*.

Two axes make this API unusually exposed:

- **one source runs as five thread projections** (BRISC, NCRISC, TRISC0-2), and
- **the same source runs on N cores.**

A statement is correct only if all five projections and all participating cores agree about
it, and C++ gives no way to see that from reading one path. Every entry below reads as
ordinary, correct code from a single thread on a single core.

Status markers are literal. **Verified** means checked against the headers or reproduced;
**unverified** means it looks reachable from the API but has not been demonstrated. Entries
without a marker are mechanisms the headers document as the caller's responsibility.

---

## A. CB protocol -- looks fine single-threaded, deadlocks in composition

1. **CB declared with fewer pages than `Storage<S>::num_pages`.** `cb_reserve_back` waits
   forever. Nothing connects the host's `make_cb(..., num_pages=N)` to the kernel's `Shape`.
   The canonical case, and a pure host/kernel contract gap.

2. **CB sized exactly `num_pages` where the kernel reserves block b+1 before popping b.**
   Self-deadlock. The flash kernel's `m`/`l`/`o` state buffers need `2 *` and only a comment
   says so.

3. **Two live blocks from one Storage.** `cb_wait_front` takes the FRONT, so reading a
   Storage whose previous block is still un-popped silently returns the wrong pages -- or
   hangs when the CB holds one block. `flash_attention.cpp` works around exactly this with a
   separate `mnow` buffer, and the reason is a comment rather than a type.

4. **A resident block placed at the wrong scope.** The reduce scaler, the column of ones and
   a fused bias must be held at kernel scope; inside a loop the destructor pops after the
   first iteration and the second hangs. Reads as ordinary RAII.

5. **Custom `Fn` load writing fewer pages than `pages.count`.** The handle pushes `count`
   whatever the routine touched, so the untouched pages are stale -- silently wrong, no hang.

6. **Subblock-sized reserve against a full-block reserve.** Noted in the subblocking work;
   a mismatch deadlocks.

## B. Thread-projection divergence -- the "legal from one thread" class

7. **CB work gated on `PhysicalCoord::this_core()`.** It returns the ORIGIN on a compute
   projection, which the header documents. The compute half then behaves as though every core
   were (0,0): guarded pushes happen everywhere, go unmatched, and deadlock.

8. **CB work inside any branch whose condition is not uniform across projections** -- a
   runtime arg only the DM side reads meaningfully, a data-dependent bound, an early `return`
   or `continue` on one path.

9. **A `noc_load` inside an `if` the compute projection does not take.** DM pushes, compute
   never waits, the CB fills, the next reserve hangs.

10. **Divergent loop trip counts** between the DM and compute halves of one loop.

## C. Collectives -- must be uniform across CORES, not just threads

11. **A multicast inside a core-dependent branch.** A broadcast is collective: every core in
    the row or column must make the same calls in the same order.

12. **Two broadcasts sharing a handshake pair.** The ready counter is waited with EQUALITY,
    so a third core's increment steps it past the target and the wait never matches.
    `matmul_mcast.cpp` spends twelve lines explaining this; nothing prevents it.

13. **`synchronize_cores()` (no-arg form) on a non-rectangular core set. VERIFIED LIVE.**
    `core_block(12)` returns 8 cores in row 0 plus 4 in row 1, while
    `TT_UNIFIED_CORE_GRID_H/W` are derived from the core range's BOUNDING BOX -- 2x8, so 16.
    The barrier then waits on four cores that were never launched. `reduction_tree.cpp` uses
    this form. The harness comment ("Bounding box, not num_cores: a barrier addresses a
    rectangle") acknowledges the mechanism without closing it.

13b. **`noc_load`'s `pair` template parameter.** Needed, but awkwardly shaped, and the
    combination is worth naming. It selects which handshake semaphore pair a collective
    uses, and it defaults to `thread` -- which is right when two collectives run on
    different threads and silently WRONG when they share one, because both then get the
    same pair and the ready counter interleaves (hazard 12). Four things compound:

    - `thread` and `pair` are adjacent small ints at the call site, so `noc_load<0, 1>` and
      `noc_load<1, 0>` are a transposition apart and both compile.
    - Getting it wrong is a HANG, not an error.
    - The invariant is whole-kernel -- "two concurrent collectives on one thread must
      differ" -- but is expressed at individual call sites, which in `matmul_blocked` sit
      inside a loop dozens of lines from each other.
    - The default is the bad kind: correct in the common case, hanging in the other.

    **The capability is load-bearing**, so this is not an argument for deleting it:
    `matmul_mcast.cpp` runs BOTH broadcasts on thread 0 whenever `MM_IN1_THREAD=0`, which
    is `test_unified_matmul_mcast.py`'s DEFAULT and exists because ttsim cannot multicast
    on NOC 1. With `pair = thread` those two would share a pair and hang.

    **Partly closed:** the index is bounded at 2 now. It never was, and pair 2 computed
    `base + 4`, which IS `kCopyArrivedSem<0>` -- a kernel asking for it would have shared
    semaphores with the `noc_core_write` arrival flag and hung. Verified by sabotage: it is
    a build error now.

    **Still open:** the shape. Moving the identity onto the region object -- so the two
    `LogicalMcast` declarations carry `Handshake::First` / `Handshake::Second` and sit side
    by side at the top of the kernel -- would put both halves of a whole-kernel invariant
    in one place, remove the adjacent-ints transposition, and close the range by
    construction. The redundant `noc_load<0, 0>` / `noc_load<1, 1>` in
    `example_matmul.cpp` (now dropped) is the small evidence that the parameter invites
    noise even from someone who knows what it does.

    **VERIFIED ON HARDWARE.** Asked
    whether a shared pair actually breaks on hardware, the sequential case was settled by
    reading: the handshake is entirely synchronous inside `noc_load` -- sender waits ready,
    barriers, broadcasts, sets and clears the flag; receiver increments, waits, clears --
    and BOTH semaphores are deliberately left at 0 on every core when it returns, which the
    two flushes exist to guarantee. `.wait()` carries no handshake state. So two loads on
    one core sharing a pair, even waited out of order, have nothing to cross: the second
    handshake cannot begin until the first returned.

    ### The mechanism, in one line

    **The ready counter counts; it does not identify.** An increment means "some core freed
    a buffer", not "the core you are waiting for freed its buffer". One counter per
    collective is enough because only that collective's receivers can increment it. Share
    one counter between two collectives with DIFFERENT RECTANGLES and a sender can be
    released by a core that is not in its rectangle at all.

    Note what this is NOT: nothing overlaps locally. Every core runs its two collectives
    strictly sequentially, each fully waited. The misuse is per-core (one pair, two
    rectangles) while the damage lands on a different core, which is why a per-core
    "is an async operation in flight" flag cannot see it and a per-core RECTANGLE claim can.

    ### Sequence of events

    A 2x2 grid, so `num_dests = volume - 1 = 1` and the arithmetic is trivial. The two
    rectangles `matmul_blocked` builds per core:

        row{ yx(me.y, 0), hw(1, GRID_W) }   // my row,    sender is (me.y, 0)
        col{ yx(0, me.x), hw(GRID_H, 1) }   // my column, sender is (0, me.x)

    | core | row role | col role |
    |---|---|---|
    | **(0,0)** | **SENDER** of row 0 | **SENDER** of col 0 |
    | (0,1) | receiver, increments (0,0) | SENDER of col 1 |
    | (1,0) | SENDER of row 1 | receiver, **increments (0,0)** |
    | (1,1) | receiver, increments (1,0) | receiver, increments (0,1) |

    Sharing a pair gives (0,0) ONE counter fed by (0,1) for the row and (1,0) for the
    column -- two different cores, two different collectives. Call it `C`. The skew has
    made (0,1) slow; (1,0) and (1,1) are fast.

    1. k=0 finishes. (0,1) enters the delay STILL HOLDING k=0's `a`/`w`: their
       `cb_pop_front` runs at the end of the iteration scope, which it has not reached.
    2. (1,0) and (1,1) finish k=0 and enter k=1.
    3. (1,0) runs k=1's ROW collective -- sender of row 1, waits for (1,1), broadcasts.
       Fast, and does not touch (0,0).
    4. (1,0) runs k=1's COL collective -- receiver of col 0: `inc_remote((0,0))`.
       **C: 0 -> 1.** It then blocks in `data_sent.wait(1)`.
    5. (0,0) runs k=1's ROW collective -- sender, `num_dests == 1`, so
       `receivers_ready.wait(1)` reads **C == 1 and MATCHES**. But that 1 came from (1,0),
       which is not in row 0's rectangle. (0,1) -- the only core (0,0) is actually waiting
       for -- is still delayed and has freed nothing.
    6. (0,0) broadcasts anyway: `set(0)`, read barrier, `noc_async_write_multicast` of
       k=1's A-block into row 0, i.e. **into (0,1)'s buffer, which still holds k=0's data
       (0,1) has not read or popped.** THIS IS THE CORRUPTION. At depth 1 there is no spare
       slot so it overwrites live data; at depth 2 it lands in the other slot harmlessly,
       which is exactly why the default masks it.
    7. (0,0) sets and broadcasts `data_sent`.
    8. (0,1) finishes its delay, pops k=0, enters k=1's row collective, `inc_remote((0,0))`.
       **C: 0 -> 1.**
    9. (0,0) runs k=1's COL collective -- sender, `wait(1)` reads C == 1 and matches, **off
       (0,1)'s ROW increment.** It broadcasts to (1,0), which is legitimately waiting, so
       this part is fine.
    10. Counts balance: two increments in, two waits satisfied, nothing spins forever.

    **That is why it is silent.** The bookkeeping is self-consistent -- every increment is
    consumed by some wait -- so no wait hangs. Only the PAIRING is wrong, and the single
    consequence is step 6.

    Two fixes, and why each works:

    - **Distinct pairs**: two counters. Step 4 increments the COL counter, which (0,0)'s row
      wait never reads, so step 5 blocks until (0,1) genuinely frees.
    - **Same rectangle on one pair is safe. TESTED, not assumed** -- the question was raised
      directly, on the grounds that two transactions on one semaphore might be unsafe
      regardless of rectangle. `test_unified_mcast_share.py` broadcasts one operand TWICE
      over the same 8x8 rectangle into two buffers on one pair: same sender, same receivers,
      same extent, no arithmetic to confuse a wrong answer with. Under the same conditions
      that break the differing-rectangle case -- 8 and 16 rounds, prefetch depth 1, skew
      holding half the receivers' buffers live at 5k, 50k and 200k iterations, repeated --
      **every run is bit-exact.**

      And there is a structural reason, which is why the result is believable rather than
      lucky. Same rectangle means the SAME SENDER, so every receiver is gated by that one
      core's flag: a receiver cannot reach collective B's increment until it has consumed
      A's flag, which the sender only sends after A's wait completed. Every increment is
      therefore in turn. The differing-rectangle case breaks precisely because a core can be
      a SENDER in one collective -- gated by nobody's flag -- while being a RECEIVER in the
      other, so its increment escapes that ordering.

      This is what makes the RECTANGLE the right thing to claim. The invariant is not "one
      transaction per semaphore"; it is "one rectangle per semaphore".

    And one more in the family: because the wait is for EQUALITY, the same sharing can also
    produce a HANG rather than corruption if increments overshoot the target. Ours balanced,
    so we got the silent version.

    **Twelve ordinary trials missed it and an adversarial one caught it**, which is the
    useful part. Sharing pair 0 across 2x2, 8x8 and 2x8 grids at depths 1 and 2 was
    correct every time. THREE conditions have to coincide:

    | condition | why |
    |---|---|
    | both collectives on one thread | otherwise the thread-derived pairs already differ |
    | a receiver holding a LIVE buffer | so an early broadcast has something to corrupt |
    | prefetch depth 1 | at depth 2 the early write lands in the spare slot, harmlessly |

    The second is where the first skew attempt went wrong: the delay sat BEFORE the load,
    which is after the previous iteration's pop, so the buffer was already free and nothing
    could break. Moved to after the loads, where the blocks are still live, it fires
    immediately.

    | 8x8 grid, thread 0 both, skew on, depth 1 | pcc |
    |---|---|
    | distinct pairs (control) | 0.999952 |
    | **shared pair 0** | **0.904540**, identical on 3 of 3 |

    A SILENT WRONG ANSWER, not a hang, and deterministic once the conditions are met. So
    the parameter is justified and is not deletable. `test_unified_matmul_blocked.py` now
    carries this as a regression test asserting the difference -- if shared pairs ever
    become safe it fails loudly, which is the right way to learn the parameter can go.
    `MMB_SKEW` and `MMB_SHARE_PAIR` stay in the kernel as the knobs it needs.

    The shape complaints above all still stand. The parameter has to exist; it does not
    have to be two adjacent ints at a call site far from its partner.

    **Do not expect the in-flight assert to catch this one.** A per-semaphore "an operation
    is live" bool -- specced in `unified_mcast_handle_spec.md` -- catches two multicast
    operations live on ONE pair, which is a real and different failure. 13b has no core with
    two live operations: each runs its two collectives strictly sequentially, fully waited,
    and the damage lands on another core. The rectangle claim is the check for 13b. Two
    checks, two failures, neither subsuming the other.

14. **Mismatched multicast rectangle** computed differently on sender and receiver.

15. **`noc_core_write` reused across rounds without an intervening barrier.** The documented
    `arrived.set(0)` window silently loses a writer's increment and hangs the round AFTER the
    one that lost it -- so the symptom is one round late.

16. **A program whose core range does not start at (0,0).** `LogicalCoord::this_core()` is
    sub-device-relative while `to_physical()` indexes the absolute worker tables; the two
    agree only at the origin.

## D. Host/kernel contract -- no compiler behind any of it

17. ~~**Runtime-arg count or order mismatch.**~~ **DEAD.** Three device hangs, all the same
    cause, and the cause no longer exists: there is no positional runtime-argument list. A
    kernel reads `get_arg(args::block_count)` against a schema its KernelSpec declares, so
    ORDER is not a thing that can be wrong -- arguments are looked up by name -- and COUNT is
    refused on the host before a kernel runs, at `program_run_args.cpp:242`
    (`provided == named_rta_names.size()`). `test_unified_negative` checks that refusal, having
    previously checked the sentinel.

    The sentinel is gone with it. `check_runtime_args` and `kRuntimeArgSentinel` existed only
    to catch a count mismatch in the positional list, and a check for a list that no longer
    exists is dead weight.

    One limit worth stating rather than glossing: metal's check compares the COUNT supplied
    against the count declared, so it catches a missing argument without saying which one. That
    is a diagnostic-quality complaint, not a hazard -- the failure is a refusal at a named line
    rather than a garbage loop bound.

18. ~~**Compile-time arg drift.**~~ **DEAD.** `TensorAccessorArgs<N>` is gone: an accessor is
    built from a binding token, `TensorAccessor(tensor::in)`, and the token carries its own
    offsets. There is nothing chained by hand and nothing downstream to shift.

    It is dead twice over, in fact. `KernelSpec::CompileTimeArgs` is a
    `Table<std::string, uint32_t>` -- the Metal 2.0 host API has no positional compile-time
    list at all, so there is no place for an offset to drift even if something wanted one.

19. **CB data format not matching the tensor's dtype.** Page size follows the format -- 1088
    bytes for bfloat8_b against 2048 for bfloat16 -- so a disagreement reads the wrong bytes
    with no error.

    **STILL LIVE, and the Metal 2.0 port produced a fresh instance rather than fixing it.** A
    dataflow buffer declares `data_format_metadata`, but nothing checks it against the
    TensorParameter the buffer is filled from. Porting `matmul_blocked`'s launcher, its weight
    buffer defaulted to bfloat16 while the tensor stayed bfloat8_b: fifteen configurations came
    back NUMERICALLY WRONG, with no error anywhere. Which is precisely what this entry
    predicts, now with a case attached.

20. **CB index collisions**, or a Storage naming a CB the host never declared.
    **VERIFIED LIVE, in our own kernel, and fixed.** `matmul_blocked.cpp` declares
    `Storage<Out> acc_storage(kCbAcc)` unconditionally while both its harnesses allocated
    that buffer only when `kb > 1` -- an L1 saving that quietly broke the contract. At
    `kb == 1` the kernel named a buffer that did not exist, which reports zero pages, so the
    capacity assert from hazard 1 fired. Found BY that assert, which is the argument for
    adding it: one check for hazard 1 caught an instance of hazard 20 that had been latent
    for the whole project. The harnesses now allocate it unconditionally.

    **HALF DEAD after the Metal 2.0 port, and it is worth being precise about which half.**
    A buffer the host never declared is now refused at build with the buffer named -- a
    dataflow buffer with no producer or no consumer fails `program_spec.cpp:393`, confirmed by
    probe -- and so is a kernel naming a buffer the launcher does not declare, which turned up
    two real cases where kernel and launcher had drifted to different NAMES for the same
    buffer (matmul_blocked's `a`/`b` against the kernel's `in`/`wo`). Under the descriptor path
    the two sides only ever had to agree on numbers, so names could drift and did.

    What survives is the original instance's own shape: slot numbers still reach the kernel as
    compile-time VALUES, and nothing checks that the number a Storage is given denotes the
    buffer that was meant. A wrong number is loud rather than silent -- every projection reads
    the same value, so they agree with each other and disagree with the host, which is wrong
    data or a hang on the first run -- but it is not caught before the run.

21. ~~**A user semaphore id colliding with the reserved multicast base.**~~ **CHECKED.** The
    harness names its semaphores and allocates the six reserved ones above the caller's, and
    `tt/unified/api.h` static_asserts the derived base against `sem::u_mcast_ready0` and the
    end of the run against `sem::u_copy_arrived1` -- both ends, which pins the whole run since
    metal cannot issue a duplicate id. A collision is a build error naming the arithmetic.
    Verified non-vacuous twice, by moving the base and by splitting the run.

30. ~~**A kernel returning with a nonposted write still in flight.**~~ **FIXED.** The DFB
    release condition is that the write has DEPARTED local L1, which is what
    `noc_async_writes_flushed()` waits for and what `~NocAsyncWriteTx` did. Kernel EXIT is a
    stricter contract, and how much stricter depends on the NOC mode: dedicated-mode firmware
    asserts `nonposted_writes_sent` (`brisck.cc:91-95`), which a flush satisfies, but under
    `DM_DYNAMIC_NOC` `brisc.cc:550-561` also asserts `nonposted_writes_flushed` -- landed, not
    merely sent -- so the NOC interface is idle for the next kernel. `noc_store(...)` without
    `.wait()` never barriered, so under dynamic NOC a store still in flight at return halted
    BRISC. `detail::release_writes` now pays the round trip in the mode that is owed it, which
    leaves dedicated-mode codegen unchanged.

    It is the rule `~NocAsyncWriteCoreTx` already stated for its atomic -- "leaving one
    outstanding is an inter-kernel data race... the ack lands after this kernel has finished,
    against whatever runs next" -- applied to the write it sits next to. The comment was right
    and the writes were the half that had not been finished.

    **What made it expensive to find is worth more than the fix.** It presents as a hang with
    a bit-exact result, because the payload does land; only the check that it had landed by
    then fails. It is invisible in dedicated mode, which is every other suite. And it is
    invisible under the WATCHER, which is the tool you reach for -- the watcher's own overhead
    gives the writes time to arrive, so the assert it exists to report does not trip. That
    combination is what kept `mcast_share` red for as long as it was.

## E. Silently wrong, never hangs

22. **`reduce_mean` with a scaler of 1.** That is a sum, with nothing to say so.

23. **`matmul_init`'s `TransposeB` disagreeing with `matmul<Tr>`.** The header documents this
    as uncheckable across two separate calls.

23b. **`.bias()` is the one thing in the FUSION position that does not run per k-block.**
    The rule `accumulate` documents is that the node runs every k-block and only the
    epilogue lambda is deferred to `finish`. `.bias()` sits in the node and is nonetheless
    gated on `finish` in both accumulator modes -- `kBiasFolded` folds it into the last
    subblock pass, `via_bias` routes it through `bias_finish`. So a reader applying the
    stated rule predicts `A@B + k_blocks*bias`, which is exactly the wrong answer
    `test_unified_matmul_bias.py` exists to catch.

    The reason it is a special case rather than an epilogue op: the epilogue is an
    `expr::UnaryChain`, whose `apply_in_place(slot)` composes unary SFPU ops on ONE DST
    slot. A bias add needs a second L1 operand, so the epilogue cannot express it at all.
    `.bias()` is a workaround for that missing capability, and its position in the
    expression is what makes the workaround invisible.

    Raised by the API's author reading `example_matmul.cpp` and predicting the documented
    semantics rather than the implemented ones -- which is the strongest possible evidence
    that the spelling misleads.

    **PARTLY FIXED, and the fix uncovered a second bug.** The first proposed remedy was to
    let the epilogue take a broadcast operand -- `sum + bcast<Rows>(bias_row)`. That is
    impossible, and the library already said so: a broadcast's left operand must be a
    stored buffer because `add_tiles_bcast_rows` reads BOTH operands from circular buffers
    and neither from DST, while an epilogue runs on DST. The only DST-legal add is the
    elementwise dest-reuse one, which is why the bias must be row-replicated. Hardware,
    not spelling.

    What works is the fluent form on the node the epilogue already receives:

        [&](auto sum) { return sum.bias(bias_row).relu(); }

    Every piece existed: `.bias()` is a `MatmulNode` method, `Fluent` supplies `.relu()`,
    and the unary-on-`MatmulNode` overloads carry `bias_cb` through a chain append. So this
    COMPILED ALREADY -- and silently produced an unbiased matmul, because `accumulate`
    only did `decltype(epilogue(declval<Bare>()))`. A chain is a type and survives that; an
    operand is a runtime member and does not. Measured 0.49312 max error on a +-0.5 bias,
    i.e. the whole bias missing.

    Fixed by EVALUATING the epilogue rather than only typing it, and threading the operand
    into the strategy where it already knew how to apply one at finish. Same dest-reuse
    add, same before-the-chain ordering: `max |epilogue - fusion| = 0.000000` at k_blocks
    1 and 3, with a relu epilogue, and in L1 mode. The bare node is built with `kNoBias` in
    every operand slot rather than `{}`, since a default-constructed 0 is a valid buffer
    index and would have added CB 0 to every output block.

    STILL OPEN: the node spelling remains legal under `accumulate`, so the misleading form
    is available even though the honest one now exists. Retiring 23b means porting the
    `accumulate` callers to the epilogue form and then rejecting a node-borne bias there --
    `.bias()` stays correct for a single-shot `store()`, where there is no epilogue and
    nothing to mislead about. `test_unified_matmul_bias.py` A/Bs the two spellings and is
    what makes that port safe; it only means something while both exist.

24. **A new pack path added without `pack_to`.** The class of bug fixed in the bfloat8_b work:
    the packer's output format is per-kernel state a pass must claim, and the fix is a
    convention rather than a constraint.

25. **Bias not replicated down all 32 rows of its tile**, or indexed with the wrong block per
    core. The second was a real bug in `example_matmul.cpp`, caught only because the launcher
    checked against torch.

26. **Using rows 1-31 of a reduction result.** Only row 0 is valid. Zeros are harmless to a
    sum and wrong for a max or a mean, which is why the two-stage tree is sum-only.

27. **Uninitialized output tensors plus allocator reuse.** A dropped output block returns the
    previous run's values, which look correct. Host-side, but a genuine detection hazard --
    it hid a missing attention head once.

## F. Worth investigating, not yet verified

28. **SFPU trees with more leaves than DST slots. UNVERIFIED.** `kPerAcquire = kMaxDstTiles /
    kLeaves` reaches 0 above 8 leaves and no `static_assert` bounds it. Whether the
    interleaved path misbehaves has not been demonstrated -- it needs a test before it is
    called a defect.

28b. **A `custom_compute` routine leaving the compute units reconfigured.** The escape
    hatch hands out raw circular-buffer ids so a routine can call any LLK, and unpacker,
    math and packer configuration is per-kernel state: whatever the routine leaves set, the
    next unified op inherits. The library already lives with this among its own passes --
    every matmul re-runs `matmul_block_init` because a broadcast or a reduction before it
    reconfigured the units, and `pack_to` exists for the same reason on the pack side -- so
    the hazard is not new, but the hatch is the first place a USER can create it. Nothing
    checks it, and the failure is a silently wrong answer in the op AFTER the routine, which
    is a bad place to look for a cause. Documented in the `custom_compute` contract in
    api.h; enforcing it would need the library to snapshot and restore the unit
    configuration around the call, which it does not do for its own passes either.

29. **Custom `Fn` issuing the wrong kind of traffic. UNVERIFIED.** The contract is writes
    only, on this thread's NOC; the handle's flush covers nothing else, so a read issued
    there would let the pop hand back pages mid-transfer. Unenforced, and untested.

---

## Already enforced

Not hazards, recorded so the list is not re-derived: elementwise shape agreement
(`node_shape<Bin>`), matmul inner-dimension agreement, `bcast` vector shape against its axis,
store destination shape against the node's, scaler-is-exactly-one-tile, `Block` consume
obligations, `RetainedBlock` occupancy, moved-from `Block` poisoning, and the
Logical/Physical coordinate split with named-only construction.

`unified_llama_prefill.md`'s claim that `Tiles<2,2> + Tiles<1,4>` is silently fine is STALE --
`node_shape<expr::Bin>` static_asserts it now.

## The shape of the list

**Everything enforced is compile-time shape algebra. Everything unenforced is page counts,
uniformity, or the host contract.** The type system was pointed at operand shapes, which is
where it worked, and nothing was pointed at the other three.

Where to start:

- **A** -- a page-count check is mechanical. The host knows `num_pages` per CB and the kernel
  knows `Shape::num_pages`; they simply never meet. Making the Shape the single source and
  deriving the CB from it would close 1 and 2 outright.
- **D** -- #17 alone has cost three hangs, which is the argument for the named kernel argument
  work surveyed and shelved earlier in `unified_llama_prefill.md` (Blaze's
  `blaze_rt_args::get<...>()`, which makes a missing argument a build error).
- **C13** is a one-line fix and a real hang available today.

---

# Triage: what to do first

Ordered by payoff over cost, with the feasibility of each checked rather than assumed.

## The finding that reorders the list

**`ASSERT_ENABLED` is 0 in a normal build.** It is 1 only under `WATCHER_ENABLED` or
`LIGHTWEIGHT_KERNEL_ASSERTS`; otherwise `ASSERT` expands to `(void(sizeof(...)))`. So the
safety net this library already has -- `Block` consume obligations, `RetainedBlock`
occupancy, moved-from poisoning -- **was dormant in every run of this work**. Written,
reviewed, paid for, never executing.

`TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` enables it through `ebreak`, with no watcher
overhead and no build change. Six suites run that way: 136 checks, zero failures, zero
spurious asserts. The net is correct and was simply switched off.

This has to come first because **adding runtime checks to a build that compiles them out is
theatre.** Every runtime item below depends on it.

## Ranked

1. **Run the suites with asserts enabled.** Zero library code, one environment variable,
   verified non-disruptive. Activates protection already written, and is the precondition
   for 2 and 4.

2. **`Storage` asserts its circular buffer is large enough (A1, A2).** DONE, and the
   feasibility probe that preceded it was WRONG -- see the correction below.

3. **C13 as a COMPILE error, which is better than the runtime check first imagined.** The
   harness can compute whether the core range is exactly rectangular -- bounding-box area
   equal to core count -- and emit `TT_UNIFIED_CORE_GRID_EXACT` only then; the no-argument
   `synchronize_cores()` then `static_assert`s on it. Zero runtime cost, catches a live hang,
   and independent of item 1. Roughly six lines.

4. **D19, circular-buffer format against tensor dtype.** `TensorAccessor::get_aligned_page_size()`
   exists, so the accessor-form load and store can assert it equals `cb_page_bytes(cb_id)`.
   About three lines. UNCHECKED CAVEAT: whether alignment padding can make the two
   legitimately differ for some dtype.

   **Now the top of this list rather than the fourth item.** Every hazard above it is dead
   (D17, D18, D21) or half dead (D20), and D19 has since produced a live instance of exactly
   the failure it describes -- fifteen numerically wrong configurations in `matmul_blocked`,
   no error. The Metal 2.0 host API also gives a second place to put the check: a dataflow
   buffer's `data_format_metadata` and the TensorParameter it is filled from are both declared
   on the host, side by side, and nothing compares them.

5. **F28, the DST leaf budget.** One `static_assert(kLeaves <= kMaxDstTiles)`. Free,
   compile-time, and it settles an open question in the list above rather than leaving it
   marked unverified.

6. **E22, `reduce_mean`'s scaler.** Give `fill_reduce_scaler` the pool and the shape so it
   computes 1 or 1/N itself, making a mean holding a sum's scaler unrepresentable. A small
   mechanical API change that removes a silent-numerics class outright.

## Worth knowing before writing negative tests

**The watcher already localises CB deadlocks precisely.** Reproduced here: with an undersized
CB, `generated/watcher/watcher.log` shows per core `CRBW` on BRISC (CB reserve-back wait) and
`CWFW` on NCRISC (CB wait-front wait), naming the kernel. It does NOT fail the run -- it polls
and keeps polling.

So for the deadlock classes the debugging tool already exists, and what is missing is anything
that turns a hang into a test failure. A negative-test harness that runs a case under the
watcher with a timeout and greps the log for stall waypoints would cover much of A and B with
no library change at all. That harness is worth building before the individual tests, because
it is what they all need.

## Explicitly not fruit

~~**D17**, named runtime arguments~~ -- **done, and the uncomfortable finding did not
survive contact.** `unified_named_args_spec.md` concluded that the runtime half was
unreachable: Metal 2.0 proper sat behind a host API with no Python bindings, leaving only a
Blaze feature whose own README says not to use it. What that spec did not look at, being
scoped to arguments, is that the *kernel* side of Metal 2.0 was never gated at all -- and the
host side turned out to need one narrow nanobind shim rather than a rewrite. Both D17 and D18
are dead; see `unified_metal2_spec.md`.

**The B and C uniformity classes** need a per-thread trace of circular-buffer operations,
cross-checked at kernel end -- push counts against wait counts, per CB, per projection. That
trace is the structural answer to most of both categories, and it is a project rather than a
fruit. Naming it here so it is not repeatedly rediscovered as an easy one.

Suggested first cut: **1 and 3**, neither of which touches library semantics and one of which
is free at runtime; then **2 and 5**; and let the negative tests for those build the harness
that later covers the rest.


---

# Outcome of items 1, 2 and 3

All three implemented. Two corrections to what the triage above claimed, both worth keeping
because each was a plausible-looking verification that did not verify what it appeared to.

## Correction 1: the capacity probe passed for the wrong reason

The triage recorded the `Storage` capacity assert as VERIFIED FEASIBLE because a probe
compiled and passed. It did -- with `ASSERT_ENABLED` at 0, where `ASSERT(x)` expands to
`(void(sizeof(not(x))))`. **`sizeof` is an unevaluated context**, so `cb_num_pages` was
never called, never emitted, and never linked. The probe demonstrated nothing.

Enabling asserts and building it for real failed to link every TRISC:

    undefined reference to `cb_interface'

`cb_interface` has no definition in a TRISC link. `cb_page_bytes` had appeared to prove
otherwise, and its comment claimed it works "on every projection" -- but every use of it in
shared code has a result that is dead on compute, so LTO deletes the call before the linker
sees it. A use compute genuinely evaluates fails.

The check is therefore behind `IS_DM_THREAD`, which loses nothing: the host configures one
circular buffer per core, so every projection would be checking the same number. The
adaptor's comment on `cb_num_pages` now records the link constraint rather than repeating
the claim that misled the probe.

**The general lesson: a check that compiles away cannot be tested by compiling it.**

## Correction 2: the two assert modes are not interchangeable

The triage proposed `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`, on the grounds that it enables
`ASSERT_ENABLED` cheaply. It does, and the six suites run that way passed. But a lightweight
assert is `ebreak`: it halts the RISC, and **the host cannot tell that from a hang**. The
undersized-CB case with lightweight asserts on still timed out at 200 seconds with no
diagnostic, exactly as it did with them off.

Under `TT_METAL_WATCHER` the same case throws to the host:

    BRISC tripped an assert on line 225 ... Current kernel: unified_kernels/example_reduce.cpp
    TT_THROW: Watcher detected tripped assert and stopped device.

So the watcher is the mode that produces a diagnostic, and per suite it costs about a
second on thirty -- 30s against 29s on rmsnorm.

The runner defaults to lightweight and takes `--watcher` as a flag. Lightweight turns a
tripped assert into a timeout -- a poor diagnostic but a real signal -- and the runner names
the suite and says to re-run that one under `--watcher` for the line. The negative suite
sets its own environment per case and ignores both.

## What landed

- **`run_unified_tests.sh`** runs every suite with the watcher on. The suites themselves are
  unchanged, so perf work runs the bench scripts directly and unwatched; every number in
  `unified_llama_prefill.md` was taken that way and stays comparable.
- **`Storage`'s constructor** asserts `cb_num_pages(cb_id) >= S::num_pages`, behind a
  data-movement guard. Greater-or-equal, since a deeper buffer is how prefetch depth works.
- **`TT_UNIFIED_CORE_GRID_EXACT`**, emitted by the harness only when the core set fills its
  bounding box, and `static_assert`ed by the no-region `synchronize_cores()`. Compile-time,
  zero runtime cost.
- **`test_unified_negative.py`**, which is the part that makes the other three mean
  something. Each case runs in a subprocess -- a tripped device assert aborts rather than
  raising, so it cannot be caught in-process -- and a case is only "ok" if it is refused BY
  THE EXPECTED CHECK. A timeout counts as a failure, since a hang is what these exist to
  eliminate.

### Verification status, stated exactly

RESOLVED. `run_unified_tests.sh` now completes end to end: **18 passed, 0 failed**, on a
rebase onto `origin/main` 876 commits newer, with asserts enabled. It had never done so
before, and the reason was hazard 20 rather than anything about the runner.

Both negative cases were confirmed to fail before the fixes and pass after:

| case | before | after |
|---|---|---|
| circular buffer one page too small | hangs, needs `tt-smi -r` | assert at `api.h:225`, host throws |
| no-region barrier on 12 cores in a 2x8 box | would wait on 4 cores never launched | static assertion, does not build |

## What this does NOT cover

The capacity assert catches A1. **A2 it does not** -- a buffer sized exactly one block is
large enough by this check and still self-deadlocks if the kernel reserves block b+1 before
popping b. Whether a block needs depth is a property of the access pattern, not of the
Shape, so `Storage` cannot know it. Still open.

The negative suite has two cases against a catalogue of twenty-nine. It is the harness that
matters more than the count: the subprocess runner, the watcher, and the rule that a
refusal must come from the named check.

## The trap that cost the most, and the three wrong theories it produced

Long runs began stalling, and it took FOUR theories to get it right. The watcher cannot
sustain many device cycles; no -- about twelve device open/close cycles in one shell; no --
hugepage exhaustion; no.

**The answer was this document's own subject.** With asserts enabled, `matmul_blocked` named
an unallocated circular buffer (hazard 20 above), the capacity assert fired, and a tripped
assert STOPS THE DEVICE. Every suite after it then failed on a stopped device. The stall
"moved" when suites were reordered or removed because what moved was the position of the
first suite to trip it -- remove `matmul_blocked` and `attention_proj`, which shares the same
harness bug, trips it instead. That is why it looked positional.

So the mystery was a real bug being correctly reported by a new check, wearing the disguise
of an infrastructure problem. Fixing the buffer allocation made the full run pass 18/18 end
to end, which it had never done.

The three wrong theories were each built from real observations, which is what made them
convincing -- the stall really did happen around the twelfth suite, twice.

The third theory was **hugepage exhaustion**, on the evidence that `HugePages_Free` was 0.
That was also wrong, and it is worth recording as wrong because it was the most plausible
of the three. `HugePages_Free` is 0 on this machine in normal operation: the fallback path
works, the device opens, and fifteen consecutive suites then ran clean with it still at 0.

The actual cause was duller. **A stopped device stays stopped until it is reset**, and each
failed run was being launched onto the wreckage of the previous one -- plus, once, a stale
`test_unified_binary.py` left holding the device after a `pkill` killed its parent shell but
not the child. `tt-smi -r` clears it. That is all it ever was.

**What identified it was stashing the library changes and finding the hang unchanged.** A
suite that hangs identically with and without your changes is not about your changes. That
check costs one minute and should come first, not after two false theories -- especially
when the alternative is bisecting a change that was never at fault.

Recovering needs the driver reset rather than `tt-smi -r`, which is a machine-level action.
The runner now says all of this in its header, and says not to hard-kill a run.

The lesser trap, still true and still worth the ordering: a negative test that trips an
assert STOPS the device, so the negative suite runs LAST and a suite failing right after
work on deliberate hangs deserves a reset before it is believed.
