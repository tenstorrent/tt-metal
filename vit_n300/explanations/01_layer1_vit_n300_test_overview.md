# Layer 1: VIT N300 Test — High-Level Overview

This document explains what happens from start to finish when the `vit-N300-func` test runs. It assumes you can program but haven’t worked much with TT-Metal. The goal is to orient you before drilling into the failure point.

---

## The Big Picture

The test runs a **Vision Transformer (ViT)** model on a single N300 accelerator. The N300 uses Wormhole silicon with two chips on one board. The test measures inference throughput (samples per second) by running batch-8 inference many times using a pre-captured *trace* and two *command queues* to overlap data movement with compute.

---

## Key Concepts You Need First

### Host vs. Device

- **Host**: Your CPU, running Python and tt-metal/ttnn host-side code. It builds programs, enqueues work, copies data to/from the accelerator, and waits for completion.
- **Device**: The N300 accelerator (Tensix cores, DRAM, NoC). It runs kernels (matmul, data movement, etc.) when the host tells it to.

The host and device communicate over PCIe. The host writes commands into shared memory (hugepages); the device reads them and executes.

### Command Queues (CQs)

A **command queue** is a FIFO of work for the device. The host enqueues commands (e.g. “run kernel X,” “copy data”). The device drains the queue.

This test uses **two command queues** (2CQ):

- **CQ0**: Inference (trace execution) — runs the ViT compute.
- **CQ1**: Data movement — copies input from host→device and output from device→host.

Using two queues lets CQ1 copy the *next* batch while CQ0 is still computing the *current* batch, so work overlaps and throughput goes up.

### The Prefetcher

The **prefetcher** is a small firmware program running on dedicated ERISC cores *on the device*. Its job:

1. Read commands from a host-visible **fetch queue** (in DRAM/hugepages).
2. Move those commands into faster on-device buffers (ring buffers).
3. Feed the **dispatch** cores, which then launch the actual compute kernels on Tensix cores.

So the flow is:

```
Host → fetch queue → Prefetcher → Dispatch → Tensix kernels (matmul, etc.)
```

The prefetcher acts as a buffer: the host can enqueue many commands quickly, and the prefetcher feeds them to dispatch at the rate the device can handle. If the prefetcher stalls, the fetch queue stops draining, and the host can block when trying to add new commands.

### Trace

A **trace** is a recorded sequence of device commands. Instead of re-submitting hundreds of ops each run, the test captures them once into a trace and then replays that trace many times. This is faster and more deterministic.

---

## Phases of the Test (Init → Teardown)

### Phase 1: Setup and Device Creation

**What happens:** Pytest creates a device, and the test builds the ViT model and its parameters.

1. `create_test_infra()`:
   - Loads the ViT model (e.g. `google/vit-base-patch16-224`).
   - Preprocesses weights for the device.
   - Allocates buffers, sets up memory configs, etc.

2. Device initialization:
   - Opens the N300 and its two chips.
   - Sets up hugepages, fabric, and command queues.
   - Allocates prefetcher and dispatch cores for CQ0 and CQ1.

**Outcome:** You have a ready device and a test harness (`test_infra`) that knows how to run one ViT forward pass.

---

### Phase 2: JIT (JIT Compile / First Run)

**What happens:** A single “normal” run to compile and validate.

1. Copy input from host to device (CQ1).
2. Reshard input into the layout the kernels expect.
3. Run the model (no trace yet).
4. Move output to DRAM and prepare it for readback.

**Outcome:** Kernels are compiled and cached. The device has run ViT once and is known to work.

---

### Phase 3: Trace Capture

**What happens:** Record all ViT compute commands into a trace.

1. `ttnn.begin_trace_capture(device, cq_id=0)` — start recording for CQ0.
2. `test_infra.run()` — run ViT; all commands for that run are captured.
3. `ttnn.end_trace_capture(device, trace_id, cq_id=0)` — stop and assign a trace ID.

The trace is stored on the device (trace region). Later, `ttnn.execute_trace()` replays it instead of re-submitting individual ops.

**Outcome:** A reusable trace that represents one ViT forward pass.

---

### Phase 4: Warmup Loop (100 iterations)

**What happens:** Run the trace 100 times to warm caches and stabilize timing.

For each iteration, the test does:

1. Wait for CQ0’s previous write event (input from last iteration is ready).
2. Reshard input into L1 for this run.
3. Record event on CQ0, then `execute_trace(device, trace_id, cq_id=0, blocking=False)`.
4. Wait for CQ0 read event (inference done).
5. Move output to DRAM.
6. Record event on CQ1, then start copying *next* batch from host to device (CQ1).
7. Wait for CQ1’s previous completion, then read output from device (CQ1).
8. Record CQ1 read event.

CQ0 and CQ1 overlap: while CQ0 runs the trace, CQ1 copies the next input and reads the previous output.

**Outcome:** Device and runtime are warmed; timing is more stable.

---

### Phase 5: Measurement Loop (1000 iterations)

**What happens:** Same loop as warmup, but with profiling.

1. Tracy signpost `"start"` (if available).
2. `profiler.start("run")`.
3. For each of 1000 iterations:
   - Same pattern as warmup: CQ0 runs trace, CQ1 does copies.
4. `profiler.end("run")`.
5. Tracy signpost `"stop"`.
6. Read device profiler.

**Outcome:** Average inference time and samples/sec are computed.

---

### Phase 6: Assert and Teardown

**What happens:**

1. Assert throughput is within the expected range (e.g. ~1323 samples/sec for N300).
2. Release the trace.
3. Close the device and free resources.

---

## Where the Error Likely Occurs

The CI failure shows:

- `TIMEOUT: device timeout in fetch queue wait, potential hang detected`
- Location: `SystemMemoryManager::fetch_queue_reserve_back`
- Triage: prefetcher for one CQ is stuck in `process_stall()` waiting on a downstream semaphore

So the failure is not in the ViT math or trace logic itself, but in the **command pipeline** between host and device.

---

## Highlighted Failure Zone: Host–Prefetcher Coordination

The most likely failure zone is the **host–prefetcher coordination** around the **fetch queue** during the measurement loop.

**Role of the fetch queue**

- The host writes commands into a shared **fetch queue** (in hugepages).
- The prefetcher reads from this queue and forwards commands to dispatch.
- Before writing, the host must **reserve** space by calling `fetch_queue_reserve_back()`.
- That function waits until the prefetcher has consumed enough entries to free space.

**Where it breaks**

If the prefetcher stalls (e.g. in `process_stall()` waiting for dispatch to signal a semaphore), it stops consuming the fetch queue. The queue fills, and the next time the host does something that needs fetch queue space—such as `copy_host_to_device_tensor` (which triggers a buffer write and thus `fetch_queue_reserve_back`)—the host blocks. After a timeout (5 seconds in CI), it reports “device timeout in fetch queue wait.”

**When it’s most likely**

This occurs during the **measurement loop**, specifically when:

- The host enqueues the **next batch copy** on CQ1 (`copy_host_to_device_tensor`).
- That enqueue needs space in the fetch queue.
- The CQ1 prefetcher is stuck in `process_stall()`, so the fetch queue for that CQ is full.
- The host waits in `fetch_queue_reserve_back`, hits the timeout, and the test fails.

The failure is non-deterministic because it depends on exact timing between:

- Host enqueue rate.
- Prefetcher consumption rate.
- When the prefetcher enters `process_stall()` and when dispatch signals it.

If you’re ready, we can go to **Layer 2** and zoom in on how the fetch queue, prefetcher, and `process_stall()` interact in that region.
