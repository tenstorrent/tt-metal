# Understanding the TT-Metal Fabric and the Reduce-Scatter Timeout Bug

This document explains the fabric communication stack in tt-metal, how reduce_scatter uses it, and the root cause of the non-deterministic device timeout we're investigating.

---

## 1. The Big Picture: Multi-Chip Communication

In a T3K system, 8 Wormhole B0 chips are arranged in a 2x4 mesh. Two MPI processes each control 4 chips. When running operations like reduce_scatter, data must flow between chips over **ethernet links**.

The communication stack has four layers:

```
 [Worker Kernel]       — runs on tensix cores, does the actual computation
       |
       v
 [Fabric Mux]          — runs on idle ethernet core, multiplexes worker data
       |
       v
 [EDM Router]          — runs on active ethernet core, manages ethernet links
       |
       v
 [Ethernet Hardware]   — physical link to remote chip
```

Each layer uses **credit-based flow control**: the sender can only send when the receiver has free buffer slots. Credits are returned after the receiver processes the data.

---

## 2. Key Components

### 2.1 Worker Kernel (tensix core)

The reduce_scatter writer kernel runs on regular compute cores. It:
1. Reads data from circular buffers (filled by the compute kernel)
2. Sends data through the fabric to remote chips
3. Disconnects from the mux when done
4. Coordinates termination with other workers

**Key file:** `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_writer.cpp`

The worker connects to the mux via a `WorkerToFabricEdmSender` handle. Every data send calls `wait_for_empty_write_slot()`, which is an **unbounded spin**:

```cpp
// edm_fabric_worker_adapters.hpp:270-273
void wait_for_empty_write_slot() const {
    WAYPOINT("FWSW");
    while (!this->edm_has_space_for_packet<1>());  // INFINITE SPIN
    WAYPOINT("FWSD");
}
```

The worker blocks here until the mux has a free buffer slot. If the mux is full (hasn't forwarded data downstream), the worker hangs.

### 2.2 Fabric Mux (idle ethernet core)

The mux kernel runs on an idle ethernet core. It accepts data from multiple workers (channels) and forwards it to the EDM router through a single downstream connection.

**Key file:** `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp`

**Configuration for reduce_scatter:**
- `NUM_FULL_SIZE_CHANNELS` = number of workers per direction (typically 2 for this test)
- `NUM_BUFFERS_FULL_SIZE_CHANNEL` = 1 (single buffer slot per channel!)
- `NUM_ITERS_BETWEEN_TEARDOWN_CHECKS` = 32
- `NUM_FULL_SIZE_CHANNELS_ITERS` = 1

The mux's main loop structure:

```
while (!IMMEDIATELY_TERMINATE) {
    if (GRACEFULLY_TERMINATE && all_channels_drained) break;

    for (i = 0..31) {                          // NUM_ITERS_BETWEEN_TEARDOWN_CHECKS
        for (channel_id = 0..N) {              // NUM_FULL_SIZE_CHANNELS
            forward_data(channel_id);           // may spin waiting for EDM space
        }
    }
}
close();  // disconnect from EDM router
```

The `forward_data()` function:
1. Checks if the channel has unsent data (`has_unsent_payload`)
2. If yes, waits for the EDM router to have space (`edm_has_space_for_packet`)
3. Forwards the packet to the EDM router
4. Returns credit to the worker (`notify_worker_of_read_counter_update`)
5. Checks for worker teardown requests (`check_worker_connections`)

### 2.3 EDM Router (active ethernet core)

The EDM router is a persistent kernel on active ethernet cores. It handles the actual data transfer across ethernet links. The mux sends data TO the router; the router sends it across the ethernet link to the remote chip.

**Key file:** `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`

The router has its own buffer slots. When it receives a packet from the mux:
1. Queues it for ethernet transmission
2. Sends across the ethernet link
3. Waits for the remote chip to acknowledge receipt
4. Frees the buffer slot
5. Returns credit to the mux

Credits flow back: EDM router -> mux -> worker. Each level can only operate when it has credits from the level below.

### 2.4 Worker Adapter (`edm_fabric_worker_adapters.hpp`)

This header provides the worker-side API for fabric communication:

- `open()` / `open_start()` + `open_finish()` — establish connection to mux
- `wait_for_empty_write_slot()` — spin until mux has buffer space (UNBOUNDED)
- `send_payload_*()` — write data to mux buffer
- `close()` / `close_start()` + `close_finish()` — disconnect from mux

`close_finish()` is another **unbounded spin**:

```cpp
// edm_fabric_worker_adapters.hpp:451-460
void close_finish() {
    WAYPOINT("FCFW");
    while (*this->worker_teardown_addr != 1) {  // INFINITE SPIN
        invalidate_l1_cache();
    }
    WAYPOINT("FCFD");
    noc_async_write_barrier(get_fabric_worker_noc());
    *(this->worker_teardown_addr) = 0;
}
```

The worker sends a disconnect request (`close_start`), then spins waiting for the mux to acknowledge (`close_finish`). The mux acknowledges via `teardown_worker_connection()`, which sends a `noc_semaphore_inc` to the worker's teardown address.

---

## 3. The Disconnect / Termination Protocol

When reduce_scatter finishes, the cleanup sequence is:

```
1. All workers finish sending data
2. Each worker calls fabric_client_disconnect() → close_start() + close_finish()
   - close_start(): writes disconnect request to mux's connection_live_semaphore
   - close_finish(): spins waiting for mux to ack via noc_semaphore_inc
3. Mux's check_worker_connections() detects the teardown request
4. Mux calls teardown_worker_connection() → sends ack to worker
5. Worker's close_finish() completes
6. Non-master workers signal the termination master (noc_semaphore_inc)
7. Termination master waits for all workers (noc_semaphore_wait)
8. Termination master sends GRACEFULLY_TERMINATE to mux
9. Mux tries to drain remaining data from all channels
10. Once drained, mux exits main loop
11. Mux calls fabric_connection.close() to disconnect from EDM router
12. Mux kernel returns → dispatch can proceed
```

**GRACEFULLY_TERMINATE** (value 1) — tells the mux to drain and exit
**IMMEDIATELY_TERMINATE** (value 2) — tells the mux to exit immediately (used by host for emergency shutdown)

---

## 4. Credit-Based Flow Control

Each connection uses credits to prevent buffer overflow:

```
Worker → Mux:
  - Mux has NUM_BUFFERS_FULL_SIZE_CHANNEL slots (typically 1)
  - Worker checks edm_has_space_for_packet() before writing
  - Mux returns credit after forwarding (notify_worker_of_read_counter_update)

Mux → EDM Router:
  - EDM has NUM_EDM_BUFFERS slots
  - Mux checks edm_has_space_for_packet() before forwarding
  - EDM returns credit after processing (notify_persistent_connection_of_free_space)
```

The `edm_has_space_for_packet()` function (line 256-268) checks:
```cpp
FORCE_INLINE bool edm_has_space_for_packet() const {
    invalidate_l1_cache();  // Ensure fresh read from L1
    if constexpr (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
        auto used_slots = this->buffer_slot_write_counter.counter
                        - *this->edm_buffer_local_free_slots_read_ptr;
        return used_slots < this->num_buffers_per_channel;
    } else {
        return get_ptr_val(worker_credits_stream_id) >= num_slots;
    }
}
```

The mux uses the L1 memory path (`I_USE_STREAM_REG_FOR_CREDIT_RECEIVE = false`). It compares its write counter against the free-slots counter updated by the EDM router via NOC writes.

---

## 5. The Bug: Channel Starvation in the Mux

### 5.1 The Symptom

~2-3% of reduce_scatter invocations hang, causing a 5-second device timeout:
```
RuntimeError: TT_THROW @ system_memory_manager.cpp:627
TIMEOUT: device timeout, potential hang detected, the device is unrecoverable
```

The hang always occurs in `run_reduce_scatter_impl` → `ttnn.from_device()`.

### 5.2 The Root Cause Chain

The timeout fires because dispatch firmware can't complete — it's waiting for kernel(s) that never finish. The chain of hangs:

```
1. EDM router is temporarily slow or congested
   (e.g., ethernet link busy, remote chip slow to ack)
        ↓
2. EDM router's input buffers fill up
        ↓
3. Mux can't forward data (edm_has_space_for_packet returns false)
        ↓
4. Mux's forward_data() spins in wait_for_empty_write_slot()
   *** THIS IS WHERE THE BUG MANIFESTS ***
        ↓
5. While channel 0 is spinning, channel 1 is never serviced
   - Channel 1's teardown requests are never checked
   - Channel 1's data is never forwarded
        ↓
6. Channel 1's worker is stuck in wait_for_empty_write_slot()
   (waiting for mux credit that never comes)
   OR stuck in close_finish() (waiting for teardown ack that never comes)
        ↓
7. Worker kernel never completes
        ↓
8. Dispatch waits for worker completion → no progress
        ↓
9. Host detects no progress for 5 seconds → TIMEOUT
```

### 5.3 Why It's Non-Deterministic

The bug requires a specific timing condition:
1. The EDM router must be temporarily congested when the mux processes channel 0
2. Channel 0's `forward_data` must enter the spin loop WHILE channel 1's worker needs service (credits or teardown)
3. The congestion must last long enough for the host timeout to fire

Under normal conditions, the EDM processes packets in ~5-10 microseconds, so congestion clears quickly. But occasionally (2-3% of the time), the timing aligns to create a deadlock.

### 5.4 The Starvation Mechanism

The mux processes channels **sequentially**: channel 0 first, then channel 1. The original `forward_data()` used `wait_for_empty_write_slot()` (unbounded spin). When channel 0 is spinning:

- **No other channel gets serviced** — the for loop is stuck on channel 0
- **Teardown requests pile up** — workers send disconnect requests that are never read
- **Credits never flow** — channel 1's data sits in the mux buffer, never forwarded

Even brief EDM congestion can cascade into a full deadlock because the unfair scheduling starves channels of their turns.

---

## 6. The Fix

### 6.1 Short Bounded Spin in `forward_data()` (both mux files)

Replace the unbounded `wait_for_empty_write_slot()` with a bounded spin of **256 iterations** (~5μs):

```cpp
constexpr uint32_t MAX_WAIT_ITERS = 256;
uint32_t wait_count = 0;
while (!fabric_connection.edm_has_space_for_packet()) {
    check_worker_connections(...);  // Still process teardowns during spin
    if (++wait_count >= MAX_WAIT_ITERS) {
        return;  // Give up, let main loop try other channels
    }
}
```

**Why 256?** The EDM processes packets in ~5-10μs. With `check_worker_connections` overhead, 256 iterations ≈ ~5μs. This is:
- **Short enough** that no channel monopolizes the EDM — both channels get roughly equal access
- **Long enough** to handle brief pipeline bubbles (~100 cycle stalls) without unnecessary retries

When the spin times out, `forward_data` returns without forwarding. The data stays in the buffer. The main loop rotates to the next channel, checks ITS teardowns, tries to forward ITS data. On the next pass, channel 0 gets another chance.

**Performance impact:** Under normal conditions (EDM has space within a few iterations), the bounded spin completes quickly — identical to the original. Only under congestion does the timeout trigger, adding ~1μs retry overhead per packet. For the reduce_scatter test (~8 packets), this adds ~8μs total — negligible.

### 6.2 Bounded Graceful Drain in Mux Main Loop

After GRACEFULLY_TERMINATE is received, the mux tries to drain remaining channel data. Originally this was unbounded — if any channel had data that couldn't be forwarded, the mux looped forever.

```cpp
constexpr uint32_t MAX_DRAIN_ITERS = 1000;
uint32_t drain_iter_count = 0;
// In the graceful termination check:
if (++drain_iter_count >= MAX_DRAIN_ITERS) {
    break;  // Give up draining, exit anyway
}
```

1000 iterations × ~65μs per iteration = ~65ms maximum drain time. If the EDM can process data at all, this is more than enough. If it can't (permanently stuck), we exit anyway so dispatch can proceed.

### 6.3 Files Modified

| File | Change |
|------|--------|
| `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp` | Bounded spin (256) in `forward_data`, bounded drain (1000) in main loop |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_mux_extension.cpp` | Bounded spin (256) in `forward_data` |

---

## 7. Previous Fix Attempts and Why They Failed

### Fix 1: Non-Blocking `forward_data` (zero spin)

Changed `forward_data` to return immediately when EDM is full:
```cpp
if (!fabric_connection.edm_has_space_for_packet()) {
    check_worker_connections(...);
    return;
}
```

**Result:** Eliminated device timeouts but caused vLLM 5-minute timeouts in later iterations. The zero-spin approach provided perfect fairness but added excessive main-loop overhead for every brief congestion event. Over hundreds of iterations, progressive performance degradation pushed inference time beyond vLLM's timeout.

### Fix 2: Interleaved Teardown Checks (unbounded spin)

Added `check_worker_connections` inside the spin loop, but kept the spin unbounded:
```cpp
while (!fabric_connection.edm_has_space_for_packet()) {
    check_worker_connections(...);  // Check THIS channel's teardown
}
```

**Result:** Didn't help at all — same device timeout rate. The interleaved check only processes the CURRENT channel's teardown. Other channels' teardowns are still blocked because the spin never exits.

### Fix 3: Large Bounded Spin (32768 iterations)

Same as current fix but with 32768 iterations (~130μs):
```cpp
constexpr uint32_t MAX_WAIT_ITERS = 32768;
```

**Result:** Didn't help — same device timeout rate. 130μs is LONGER than the EDM's per-packet processing time (~5-10μs). Channel 0's spin captured every freed EDM slot during its 130μs window, giving channel 1 zero effective access. Same starvation problem as the unbounded spin.

### Fix 4 (Current): Short Bounded Spin (256 iterations)

256 iterations ≈ ~5μs, which is SHORTER than one EDM packet processing time. This means channel 0 can only capture one (maybe zero) EDM slots during its spin. Then channel 1 gets its turn and has an equal chance.

---

## 8. Remaining Unbounded Spins (Not Yet Fixed)

These are potential future issues in the same code path:

| Location | Function | Risk |
|----------|----------|------|
| `edm_fabric_worker_adapters.hpp:270` | `wait_for_empty_write_slot()` | Worker waiting for mux credits. If mux is permanently stuck, worker hangs. |
| `edm_fabric_worker_adapters.hpp:454` | `close_finish()` | Worker waiting for teardown ack. If mux never processes teardown, worker hangs. |
| `fabric_erisc_router.cpp` | Various sender/receiver channel steps | EDM router has its own potential stall points. |
| `tt_fabric_mux.cpp:299` | `fabric_connection.close()` | Mux disconnecting from EDM router calls close_finish() — unbounded spin if EDM unresponsive. |

The current fix addresses the mux-level starvation. If the EDM router itself is permanently stuck (not just slow), the worker-level spins could still cause hangs. The bounded drain (Change 2) helps with the mux's own exit, but the worker's `wait_for_empty_write_slot` remains unbounded.

---

## 9. How the Host Detects the Timeout

The host monitors dispatch progress via `loop_and_wait_with_timeout()` in `system_memory_manager.cpp`:

1. Host reads completion queue for new events (from dispatch firmware)
2. If no events, reads `DISPATCH_PROGRESS` counter from device L1
3. If neither has changed for `TT_METAL_OPERATION_TIMEOUT_SECONDS` (5.0s in our test), throws TIMEOUT
4. The progress check runs every 100ms

The dispatch firmware (`cq_dispatch.cpp`) increments `DISPATCH_PROGRESS` as it processes commands. If dispatch is waiting for a kernel to complete (`process_wait`), it doesn't increment, and eventually the host detects the stall.

---

## 10. Key File Reference

| File | Purpose |
|------|---------|
| `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp` | Mux kernel — forwards worker data to EDM router |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_mux_extension.cpp` | Router-variant mux with same forwarding logic |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` | Worker-side fabric API (connect, send, disconnect) |
| `tt_metal/fabric/hw/inc/tt_fabric_utils.h` | Termination signal checks, teardown processing |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` | EDM channel internals (teardown_worker_connection, credits) |
| `tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp` | Host-side mux API (fabric_endpoint_terminate, fabric_client_disconnect) |
| `tt_metal/fabric/hw/inc/linear/api.h` | Fabric send functions (fabric_unicast_*_with_state) |
| `tt_metal/fabric/fabric_edm_packet_header.hpp` | TerminationSignal enum (KEEP_RUNNING, GRACEFULLY_TERMINATE, IMMEDIATELY_TERMINATE) |
| `tt_metal/impl/dispatch/system_memory_manager.cpp` | Host timeout detection (line 627) |
| `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp` | Dispatch firmware, progress counter |
| `ttnn/.../line_reduce_scatter_minimal_async_writer.cpp` | Reduce scatter writer kernel (disconnect + terminate sequence) |
