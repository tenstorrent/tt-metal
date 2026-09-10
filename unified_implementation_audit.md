# Unified implementation audit

This audit covers `unified_overview.md`, the implementation under `tt/unified`,
and the standalone test harness. Items are ranked by expected correctness and
maintenance benefit relative to implementation cost.

## Stack rank

1. **Restore `unified_selftest.cpp` and require it in CI.** The documented DM0,
   DM1, and compute builds no longer compile: the harness lacks the current tile
   geometry hooks and still directly constructs `Storage`. This is a cheap fix
   that restores the fastest coverage for header compilation, protocol traces,
   DFB balance, and equivalent API spellings.

2. **Migrate unified transfers to the typed `Noc` API.** Normal load, store,
   multicast, and core-copy paths pass `DataflowBuffer::get_read_ptr()` or
   `get_write_ptr()` to the legacy free NOC functions. On Quasar DM those are
   uncached aliases. The legacy functions do not normalize them, while the typed
   `Noc` API explicitly converts local addresses back to their cached view.
   This is a silent wrong-address risk across most unified data movement.

3. **Validate DM-thread and multicast-pair template arguments.** `Input`,
   `Output`, `Semaphore`, and transaction types accept arbitrary integer thread
   IDs. A thread other than 0 or 1 compiles but executes on neither physical DM
   projection, which can hang compute. Multicast checks only `pair < 2`, so a
   negative pair can derive an invalid semaphore ID. Prefer a typed DM-thread
   enum and require `0 <= pair && pair < 2`.

4. **Close the public `Storage` and `Block` safety bypasses.** Public generic
   `Storage` NOC overloads accept endpoint base conversions, allowing an output
   to be loaded or an input to be used by storage-leading store despite the
   role-specific API. Public `Block(Storage)` and `Block(uint32_t)` constructors
   also let callers fabricate evidence that a buffer was produced. Move generic
   helpers into `detail`, expose endpoint-correct overloads, and make block
   construction private except for an explicitly named unsafe escape hatch.

5. **Add `[[nodiscard]]` to protocol obligations.** Discarding `Block`, read and
   multicast handles, or core-copy handles silently abandons a required consume
   or wait in ordinary builds because destructor validation is assertion-only.
   Compile-time warnings provide inexpensive protection. A normal asynchronous
   write handle may remain discardable because its destructor intentionally
   completes and releases the operation.

6. **Fix named `ComputeBlock` detection.** `is_compute_block` recognizes only
   `ComputeBlock<S, kNoDfb>`, so `custom_compute` rejects the supported named
   `ComputeBlock<S, DfbId>` form. Specialize the trait on both template
   parameters and remove the redundant operand specialization.

7. **Enforce `Accumulator` invariants.** The accumulator and output DFBs must be
   different, but the constructor does not check this. `clear()` can also abandon
   a live partial accumulation by resetting `reload`. Validate distinct buffers
   and reject clearing while a partial result is active.

8. **Correct or restrict `noc_core_read` sizing.** Its transaction type permits
   a destination larger than the source, but the implementation reads the full
   destination size from the peer source. This can read past the produced source
   block. Require equal sizes until a larger-destination read/publication
   protocol is explicitly designed.

9. **Fix logical-coordinate mapping for nonzero program origins.**
   `LogicalCoord::this_core()` returns program-relative coordinates, while
   `to_physical()` indexes global logical-to-virtual maps directly. Programs
   whose core range does not start at global logical `(0, 0)` can target the
   wrong physical cores. Translation needs the program/subdevice origin or an
   API that directly maps relative coordinates.

10. **Validate actual data formats.** Device-side accessor validation compares
    only DFB entry bytes with aligned tensor page bytes. Different formats with
    equal sizes can pass. Add host-side validation between tensor dtype and DFB
    data-format metadata, retaining the byte-size assertion as a sanity check.

11. **Harden shape and multicast-region construction.** Reject zero shape
    dimensions, out-of-range `dim()` indices, rank-one `with_hw`, zero multicast
    extents, and descending physical rectangles. Several currently underflow,
    divide by zero, or index outside compile-time arrays.

12. **Simplify reduction and custom-compute state APIs.** Make `reduce_max`
    scaler-free and derive the mean scaler from the reduction shape so a mean
    cannot silently become a sum. After `custom_compute`, invalidate both pack
    and unpack memoized state because a callback can reconfigure hardware behind
    unified's cache.

13. **Fix `RetainedBlock` object lifetime.** `release()` moves from a
    placement-new `Block` without destroying the moved-from object, and `get()`
    should use `std::launder`. Correct the manual lifetime handling, or replace it
    with an optional-like representation if device-library cost permits.

14. **Remove dead state and repetitive templates.** `NocAsyncMcastTx` stores a
    data-sent semaphore and sender flag that are never read. Remove them and the
    resulting accessor. Consolidate repeated unary wrappers and replace the broad
    `is_storable` detection with an explicit expression-node trait.

## Deferred item

`Strategy<FPUFusion>::bias_finish` appears to omit the initial unpack-geometry
configuration for mixed-geometry bias. Keep this below the items above until a
real test exercises that path: adding code to an unreachable path previously had
a measurable L1 cost.

## Suggested implementation order

Restore the standalone test first, migrate NOC addressing second, then add the
compile-time protocol checks and `[[nodiscard]]`. With that coverage in place,
close the ownership/role escape paths before changing the larger reduction and
coordinate APIs.
