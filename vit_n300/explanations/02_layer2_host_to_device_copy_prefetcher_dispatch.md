# Layer 2: Host-to-Device Copy — Prefetcher and Dispatch Pipeline

This document zooms into what happens when the host copies input data to device DRAM during the VIT test. It explains the prefetcher and dispatcher roles and how the fetch queue coordinates them. **Read Layer 1 first** if you haven't already.

---

## The Path: Python Call to Device DRAM

When the test runs `ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)` or `tt_image_res = tt_inputs_host.to(device, ...)`:

1. **Python/ttnn** converts this into a mesh buffer write.
2. **`FDMeshCommandQueue::write_shard_to_device()`** is invoked (one CQ owns the mesh; the CQ ID comes from the `1` in the call — CQ1).
3. **`buffer_dispatch::write_to_device_buffer()`** builds the actual command sequence and enqueues it.

The rest of this document traces that enqueue path and how the device executes it.

---

## The Two Queues: Issue Queue and Fetch Queue

The host uses two main structures to send work to the device:

### 1. Issue Queue (in hugepages / host–device shared memory)

- **Location**: Host-visible memory (hugepages) shared with the device over PCIe.
- **Content**: Raw command bytes — prefetch commands, dispatch commands, and inline data.
- **Flow**: Host allocates space, fills it with commands, then tells the prefetcher where to read from via the fetch queue.
- **API**: `issue_queue_reserve()` → fill buffer → `issue_queue_push_back()`.

Commands here are high-level: e.g. “relay X bytes from host address A to device address B,” “stall until workers complete,” etc.

### 2. Fetch Queue (in prefetcher L1)

- **Location**: Prefetcher core L1 memory.
- **Content**: Small entries (e.g. 16-bit) encoding *sizes* of commands in the issue queue.
- **Role**: Acts as a producer–consumer queue between host and prefetcher.
  - **Host**: Producer — writes new entries (command sizes), advances write pointer.
  - **Prefetcher**: Consumer — reads entries, uses them to know how many bytes to fetch from the issue queue.
- **Size**: Finite (e.g. dozens of entries). When full, the host must wait before adding more.

The fetch queue does *not* hold the commands themselves. It holds metadata that tells the prefetcher how much to read from the issue queue for each command.

---

## Host-Side Flow for a Buffer Write

For a sharded buffer write (e.g. `copy_host_to_device_tensor`), the sequence is:

### Step 1: Build the command sequence

`buffer_dispatch::issue_buffer_dispatch_command_sequence()` assembles a command sequence that includes:

- **Dispatch wait + prefetch stall**: Wait for workers to finish prior work, then insert `CQ_PREFETCH_CMD_STALL` so the prefetcher will drain the pipe before continuing.
- **Prefetch relay**: Instructions for the prefetcher to read data from the host (PCIe) and write it to the dispatch core’s circular buffer.
- **Dispatch write**: Instruction for the dispatch core to copy from its buffer to the target device DRAM/L1.

The exact layout depends on whether the buffer is sharded, uses pinned memory, etc.

### Step 2: Reserve space and enqueue

```
issue_queue_reserve(cmd_size)     → get pointer into issue queue
<fill command_sequence into that region>
issue_queue_push_back(cmd_size)  → advance host’s issue queue write pointer
fetch_queue_reserve_back(cq_id)  → wait for prefetcher to free fetch queue space  ← CAN BLOCK
fetch_queue_write(cmd_size, cq_id) → write new fetch queue entry, advance host write ptr
```

### Step 3: The critical wait — `fetch_queue_reserve_back`

Before writing a new fetch queue entry, the host must ensure there is space. That logic lives in `SystemMemoryManager::fetch_queue_reserve_back()`:

- **Host state**:
  - `prefetch_q_dev_ptrs[cq_id]`: host write pointer (next slot to write).
  - `prefetch_q_dev_fences[cq_id]`: last known prefetcher read pointer.

- **When `prefetch_q_dev_ptrs == prefetch_q_dev_fences`**: the fetch queue is full. The host must wait.

- **Wait loop**:
  1. Read the prefetcher’s current read pointer from its L1 (`prefetch_q_rd_ptr`).
  2. Update `prefetch_q_dev_fences` with that value.
  3. If `prefetch_q_dev_ptrs != prefetch_q_dev_fences`, there is space → exit.
  4. Otherwise, sleep briefly and repeat.
  5. If no progress for `TT_METAL_OPERATION_TIMEOUT_SECONDS` (5 s in CI), call timeout handler and throw.

So the host blocks when the prefetcher stops advancing its read pointer. That is the exact path that times out in the CI failure.

---

## Prefetcher Overview

The prefetcher is firmware running on an ERISC core. For each CQ there is a prefetcher (e.g. CQ0 and CQ1 each have one).

**Responsibilities**:

1. Read fetch queue entries (command sizes).
2. For each entry, read that many bytes from the issue queue (host memory) into its **cmddat_q** (command-data queue in L1).
3. Process commands in cmddat_q and forward them to the dispatch core.

**Data flow**:

```
Fetch queue (L1) → [read entry] → Issue queue (hugepages) → [noc_async_read] → cmddat_q (L1)
                                                                                    ↓
                                                                              process_cmd() loop
                                                                                    ↓
                                               Dispatch core circular buffer ← write via NoC
```

---

## Prefetcher Command Processing

The prefetcher runs a loop over commands in cmddat_q. Each command has a type; the handler advances `cmd_ptr` and returns. Important types include:

| Command | Purpose |
|---------|---------|
| `CQ_PREFETCH_CMD_RELAY_LINEAR` | Copy a linear range from host/DRAM to the dispatch buffer |
| `CQ_PREFETCH_CMD_RELAY_PAGED` | Paged copy from host/DRAM to dispatch buffer |
| `CQ_PREFETCH_CMD_RELAY_INLINE` | Small inline payload + optional dispatch command |
| `CQ_PREFETCH_CMD_STALL` | **Pause prefetcher until dispatch signals** |
| `CQ_PREFETCH_CMD_TERMINATE` | Shutdown |

For a buffer write, the sequence typically includes relay commands (to move data) and a stall. The stall is what can cause the hang.

---

## Dispatch Core Overview

The **dispatch** core is another ERISC program. It:

- Reads from its own circular buffer (written by the prefetcher).
- Interprets dispatch commands (e.g. “write this data to DRAM”, “launch these kernels”).
- Writes data to device DRAM/L1 and launches worker kernels on Tensix cores.

Prefetcher and dispatch are producer–consumer: prefetcher produces into the dispatch buffer, dispatch consumes and executes.

---

## Highlighted Failure Zone: `process_stall` and Semaphore Wait

The most likely failure point in this pipeline is the prefetcher’s handling of **`CQ_PREFETCH_CMD_STALL`**.

### What `CQ_PREFETCH_CMD_STALL` Does

When the prefetcher encounters `CQ_PREFETCH_CMD_STALL`, it calls `process_stall()`:

```c
uint32_t process_stall(uint32_t cmd_ptr) {
    static uint32_t count = 0;
    count++;
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(my_downstream_sync_sem_id));
    uint32_t heartbeat = 0;
    do {
        invalidate_l1_cache();
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat, CQ_PREFETCH_CMD_BARE_MIN_SIZE);
    } while (*sem_addr != count);
    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}
```

So the prefetcher:

1. Increments a local `count`.
2. Spins on a **semaphore** in shared memory until `*sem_addr == count`.
3. The semaphore is the “downstream sync” semaphore, signaled by the **dispatch** core when it has finished the corresponding work (e.g. the dispatch wait and copy that preceded the stall in the command stream).

### Why the Stall Exists

The stall is used to **drain the pipe**: make sure all data written to the dispatch buffer has been consumed and all prior work (e.g. buffer writes) has completed before the prefetcher continues. That avoids races when the next commands reuse buffers or depend on prior results.

### The Failure Scenario

- Prefetcher reaches `CQ_PREFETCH_CMD_STALL` and enters `process_stall()`.
- It spins waiting for `*sem_addr == count`.
- Dispatch is supposed to signal that semaphore when it finishes the prior commands.
- If dispatch never signals (or signals with the wrong value, or there is a timing/race bug), the prefetcher stays in the loop forever.

**Effect on the host**:

- The prefetcher never returns from `process_stall()`.
- It never advances its fetch queue read pointer.
- `fetch_queue_reserve_back` never sees progress.
- After 5 seconds, the timeout fires and the test fails with “device timeout in fetch queue wait.”

### Why Non-Deterministic?

The stall is a synchronization point between prefetcher and dispatch. The bug likely involves:

- Exact ordering of commands and semaphore updates.
- Timing between CQ0 and CQ1 when both are active (e.g. CQ1 copying while CQ0 runs a trace).
- Possible races in semaphore use or in when the stall is inserted relative to dispatch completion.

A rare wrong timing can leave the prefetcher waiting forever on a semaphore that never reaches the expected value.

---

## Summary: The Full Copy Path

```
1. Python: copy_host_to_device_tensor(...)
2. FDMeshCommandQueue::write_shard_to_device()
3. buffer_dispatch::write_to_device_buffer()
4. issue_buffer_dispatch_command_sequence()
   - Builds commands (relay, stall, dispatch write)
   - issue_queue_reserve()  → get buffer
   - <fill commands>
   - issue_queue_push_back()
   - fetch_queue_reserve_back()  → WAIT FOR PREFETCHER (can timeout here)
   - fetch_queue_write()

5. Prefetcher (on device):
   - Reads fetch queue entry
   - Fetches command bytes from issue queue into cmddat_q
   - process_cmd() for each command
   - CQ_PREFETCH_CMD_RELAY_* → copy data to dispatch buffer
   - CQ_PREFETCH_CMD_STALL → process_stall() → SPIN ON SEMAPHORE (can hang here)
   - Advances fetch queue read ptr when done

6. Dispatch (on device):
   - Reads from its buffer
   - Executes write to DRAM
   - Signals semaphore when done
```

---

## Next Step: Layer 3

Layer 3 will focus on the **`process_stall` semaphore protocol**: who writes the semaphore, when, and how the prefetcher and dispatch coordinate. That is the sub-component most likely to explain the hang.
