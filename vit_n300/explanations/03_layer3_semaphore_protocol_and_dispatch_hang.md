# Layer 3: Semaphore Protocol and Dual-Path Failure Analysis

This document details the **prefetcher–dispatch semaphore protocol** that gates `CQ_PREFETCH_CMD_STALL`, and presents a **dual-path analysis** of how the hang can occur: either the prefetcher spins forever (Path A), or dispatch never reaches the point where it can signal (Path B).

**Read Layer 1 and Layer 2 first** for full context.

---

## The Semaphore Protocol for `process_stall`

### Command Layout for `add_dispatch_wait_with_prefetch_stall`

When the host calls `add_dispatch_wait_with_prefetch_stall()`, it emits three pieces in order:

| Order | Type | Purpose |
|-------|------|---------|
| 1 | `CQ_PREFETCH_CMD_RELAY_INLINE` | Inline payload = WAIT; relay sends it to dispatch; stride advances to stall |
| 2 | `CQ_DISPATCH_CMD_WAIT` (inside relay payload) | Dispatch waits (if needed) and **signals the prefetcher** when done |
| 3 | `CQ_PREFETCH_CMD_STALL` | Prefetcher blocks until the semaphore matches its expectation |

The relay packs the wait command into the stream sent to the dispatch core’s circular buffer. So from the prefetcher’s view:

```
process_cmd() → CQ_PREFETCH_CMD_RELAY_INLINE  (sends wait cmd to dispatch)
            → CQ_PREFETCH_CMD_STALL           (blocks until dispatch signals)
```

### Prefetcher Side: `process_stall()`

```cpp
uint32_t process_stall(uint32_t cmd_ptr) {
    static uint32_t count = 0;
    count++;

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore<fd_core_type>(my_downstream_sync_sem_id));

    uint32_t heartbeat = 0;
    do {
        invalidate_l1_cache();
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat, CQ_PREFETCH_CMD_BARE_MIN_SIZE);
    } while (*sem_addr != count);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}
```

- `my_downstream_sync_sem_id` is `DOWNSTREAM_SYNC_SEM_ID` (from prefetch kernel config).
- The semaphore lives in **prefetcher L1**.
- The prefetcher blocks until `*sem_addr == count`.
- `count` is a monotonically increasing static counter; each stall expects the next integer.

### Dispatch Side: Who Writes the Semaphore?

Dispatch never touches this semaphore directly. It is on the **prefetcher** core, so dispatch must do a **NoC write** to it.

The mapping:

- Prefetch kernel config: `downstream_sync_sem_id` → physical sem ID in prefetcher L1.
- Dispatch kernel config: `upstream_sync_sem` = prefetch’s `downstream_sync_sem_id`.
- Dispatch uses `upstream_noc_xy` (prefetcher core) for the target.

When dispatch runs `CQ_DISPATCH_CMD_WAIT` with `CQ_DISPATCH_CMD_WAIT_FLAG_NOTIFY_PREFETCH`:

```cpp
if (notify_prefetch) {
    noc_semaphore_inc(
        get_noc_addr_helper(upstream_noc_xy, get_semaphore<fd_core_type>(upstream_sync_sem)),
        1,
        upstream_noc_index);
}
```

So:

- **Writer**: Dispatch core.
- **Location**: Prefetcher L1, at `upstream_sync_sem` (same physical semaphore as prefetcher’s `my_downstream_sync_sem_id`).
- **Operation**: Atomic increment by 1 via NoC.

### Expected Flow

1. Prefetcher runs `relay_inline` → sends the wait command to dispatch’s circular buffer.
2. Prefetcher runs `CQ_PREFETCH_CMD_STALL` → enters `process_stall()`, increments `count`, spins on `*sem_addr == count`.
3. Dispatch reads and runs `CQ_DISPATCH_CMD_WAIT`:
   - Optionally waits on memory / stream (`wait_memory`, `wait_stream`).
   - If `notify_prefetch` is set, calls `noc_semaphore_inc(..., 1)` on the prefetcher semaphore.
4. Prefetcher sees `*sem_addr == count`, exits `process_stall()`, continues.
5. Prefetcher advances its fetch queue read pointer; host can proceed past `fetch_queue_reserve_back`.

For the protocol to work, dispatch must **reach and complete** the `CQ_DISPATCH_CMD_WAIT` that has `notify_prefetch` set.

---

## Dual-Path Failure Analysis

The hang shows up as **“device timeout in fetch queue wait”** from `fetch_queue_reserve_back()`. That means the prefetcher has stopped advancing its read pointer. The question is **why**.

### Path A: Prefetcher Stuck in `process_stall` (Semaphore Never Reaches `count`)

**Scenario**: The prefetcher correctly enters `process_stall()` and spins. The semaphore never reaches the expected `count`.

**Possible causes**:

1. **Dispatch never reaches the wait command**
   - The wait command is later in the dispatch stream.
   - Dispatch may be blocked on an earlier command and never processes the wait.

2. **Dispatch reaches the wait but never signals**
   - Bug in `notify_prefetch` handling.
   - Misconfiguration (`upstream_sync_sem`, `upstream_noc_xy` wrong).
   - NoC fault (write never reaches prefetcher L1).

3. **Signaling with wrong value**
   - `process_stall` expects `*sem_addr == count` (exact match).
   - Semaphore is a simple counter; dispatch only does +1.
   - If some other path also touches this semaphore, `count` and semaphore value could diverge.

4. **CQ interleaving**
   - CQ0 and CQ1 each have their own prefetcher and dispatch.
   - Semaphores are per-CQ.
   - Cross-CQ interaction (e.g., shared resources, ordering) could cause unexpected behavior.

### Path B: Dispatch Stuck Before It Can Signal (Cascading Failure)

If dispatch never reaches the `CQ_DISPATCH_CMD_WAIT` with `notify_prefetch`, the prefetcher will remain in `process_stall()`. So we need to ask: **what can block dispatch before that point?**

#### Potential Blocking Points in Dispatch

**1. `process_go_signal_mcast_cmd()` — waypoints `WCW` / `WCD`**

```cpp
// Wait for workers (Tensix) to have space before sending go signal
WAYPOINT("WCW");
while (!stream_wrap_ge(
    NOC_STREAM_READ_REG(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX), wait_count)) {
}
WAYPOINT("WCD");
cq_noc_async_write_with_state<CQ_NOC_sndl, CQ_NOC_wait>(0, 0, 0);
```

- Dispatch spins until `STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE >= wait_count`.
- That reflects worker (Tensix) readiness.
- If workers never free enough space (e.g., deadlock, hang), dispatch never leaves this loop.

**2. `process_wait()` — waypoints `PWW` / `PWD`**

```cpp
WAYPOINT("PWW");
if (wait_memory) {
    // Spin on semaphore until *sem_addr >= count
    do { ... } while (!wrap_ge(*sem_addr, count));
}
if (wait_stream) {
    // Spin on stream until space available
    do { ... } while (!stream_wrap_ge(*sem_addr, count));
}
WAYPOINT("PWD");
```

- If `wait_memory` or `wait_stream` is set, dispatch waits before doing anything else in that command, including `notify_prefetch`.
- If workers never satisfy the memory/stream condition, dispatch stays at `PWW` and never signals.

**3. Circular buffer starvation**

- Dispatch reads from its circular buffer; the prefetcher fills it.
- If the buffer is empty, dispatch blocks in `cb_acquire_pages` or equivalent.
- For `add_dispatch_wait_with_prefetch_stall`, the wait and stall are in the same relay block, so the wait should already be in the buffer when the prefetcher stalls.
- The more interesting case: an **earlier** command in the stream blocks dispatch (e.g., `process_go_signal_mcast_cmd`), so the wait is never reached.

#### Connection to Tensix / Trace Execution

- ViT uses CQ0 for trace (compute) and CQ1 for data movement.
- Trace execution involves many `CQ_DISPATCH_CMD_SEND_GO_SIGNAL` commands.
- If CQ0’s dispatch is stuck in `process_go_signal_mcast_cmd` (workers not progressing), CQ0’s stream stalls.
- CQ1 runs host→device copies with `add_dispatch_wait_with_prefetch_stall`. Its prefetcher can still hit `CQ_PREFETCH_CMD_STALL` and block in `process_stall`.
- CQ0 and CQ1 use different prefetcher/dispatch pairs, so a CQ0 dispatch hang does not directly block CQ1’s semaphore.
- The more relevant case: **CQ1’s own dispatch** is stuck (e.g., in a go-signal or wait) before the `notify_prefetch` wait. Then CQ1’s prefetcher stays in `process_stall`, and the host sees the fetch queue timeout for CQ1.

---

## Summary: Two Failure Paths

| Path | Where it hangs | Root cause idea |
|------|----------------|------------------|
| **A** | Prefetcher in `process_stall()` | Semaphore never reaches `count` — dispatch either never signals or signals incorrectly |
| **B** | Dispatch before `notify_prefetch` | Dispatch stuck in an earlier command (e.g. `process_go_signal_mcast_cmd` at WCW, or `process_wait` at PWW) so it never reaches the wait that would signal the prefetcher |

In both paths, the host observes:

- Prefetcher does not advance fetch queue read pointer.
- `fetch_queue_reserve_back` times out.
- Error: `"device timeout in fetch queue wait, potential hang detected"`.

Path B suggests that the underlying issue may be **dispatch or worker progress** (e.g., Tensix deadlock or stream backpressure), with the prefetcher hang as a consequence.

---

## Debugging Hints

1. **Waypoint dumps**: Inspect firmware waypoints (PSW/PSD for prefetcher stall, PWW/PWD for dispatch wait, WCW/WCD for go-signal) to see where execution stops.
2. **Semaphore values**: Read the downstream sync semaphore in prefetcher L1 and confirm it matches the expected `count` when the stall is active.
3. **Dispatch progress**: Check whether dispatch has reached the wait command (e.g., via `cmd_ptr` or equivalent) and whether `notify_prefetch` is set.
4. **Worker / stream state**: For Path B, inspect `STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE` and worker core state to see if Tensix is blocked.

---

## Next Step: Layer 4

Layer 4 can zoom into **specific commands and trace structure** (e.g., ordering of waits, go-signals, and stalls in CQ0 and CQ1) and how they might interact with CQ1’s `add_dispatch_wait_with_prefetch_stall` under concurrent execution.
