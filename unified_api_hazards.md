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

14. **Mismatched multicast rectangle** computed differently on sender and receiver.

15. **`noc_core_write` reused across rounds without an intervening barrier.** The documented
    `arrived.set(0)` window silently loses a writer's increment and hangs the round AFTER the
    one that lost it -- so the symptom is one round late.

16. **A program whose core range does not start at (0,0).** `LogicalCoord::this_core()` is
    sub-device-relative while `to_physical()` indexes the absolute worker tables; the two
    agree only at the origin.

## D. Host/kernel contract -- no compiler behind any of it

17. **Runtime-arg count or order mismatch.** THREE device hangs so far, all the same cause.
    The list is positional and untyped: a kernel gains an argument and every launcher must be
    found by grep. The kernel reads a garbage loop bound and the device hangs -- no compile
    error, no assert.

18. **Compile-time arg drift.** `TensorAccessorArgs<N>` offsets are chained by hand, so
    inserting one argument shifts every one downstream.

19. **CB data format not matching the tensor's dtype.** Page size follows the format -- 1088
    bytes for bfloat8_b against 2048 for bfloat16 -- so a disagreement reads the wrong bytes
    with no error.

20. **CB index collisions**, or a Storage naming a CB the host never declared.

21. **A user semaphore id colliding with the reserved multicast base.**

## E. Silently wrong, never hangs

22. **`reduce_mean` with a scaler of 1.** That is a sum, with nothing to say so.

23. **`matmul_init`'s `TransposeB` disagreeing with `matmul<Tr>`.** The header documents this
    as uncheckable across two separate calls.

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
