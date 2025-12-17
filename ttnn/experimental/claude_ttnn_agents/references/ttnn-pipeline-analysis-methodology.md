# Kernel Pipelining Analysis Methodology for TT-Metalium

A comprehensive reference for understanding and analyzing pipeline behavior in TT-Metalium kernels.

**Purpose**: This reference file provides detailed methodology, worked examples, and formulas for pipeline analysis. The `ttnn-pipeline-analyzer` agent should consult this file when encountering unfamiliar patterns or needing worked examples.

---

## Table of Contents

1. CB Semantics Deep Dive
2. CB State Model
3. Tools & Techniques
4. Case Studies
5. Common Patterns (Detailed)
6. Best Practices
7. Methodology Summary & Decision Tree
8. Appendix: Formulas & Quick Reference

---

## CB Semantics Deep Dive

### Producer Operations

```cpp
// Producer (e.g., Reader)
cb_reserve_back(cb_id, N);  // BLOCKS if < N free slots
// ... write data to CB ...
cb_push_back(cb_id, N);      // Mark N tiles as ready
```

### Consumer Operations

```cpp
// Consumer (e.g., Compute)
cb_wait_front(cb_id, N);     // BLOCKS if < N ready tiles
// ... read data from CB ...
cb_pop_front(cb_id, N);      // Free N slots for producer
```

### Key Insight: Shared CBs = Synchronization

```
Reader → CB0 → Compute
         ↓
         └──── Shared resource!

If CB0 capacity = block size:
  → Single buffering → NO overlap

If CB0 capacity = 2 × block size:
  → Double buffering → Possible overlap
```

---

## CB State Model

### State Variables

```
Capacity (C): Total tiles the CB can hold
Used (U): Tiles currently in CB
Ready (R): Tiles pushed but not popped (0 ≤ R ≤ U ≤ C)
Free (F): Slots available for writing (F = C - U)
```

### Blocking Conditions

```
cb_reserve_back(N) BLOCKS when: Free < N
  where Free = Capacity - Used

cb_wait_front(N) BLOCKS when: Ready < N
  where Ready = tiles pushed but not yet popped
```

### Single Buffer State Transitions

```
State 1: CB empty (0/N)
  Reader: reserve(N) → SUCCESS ✓
  Reader: ... works ...
  Reader: push(N)

State 2: CB full (N/N)
  Reader: reserve(N) → BLOCKS ✗ (needs N free, has 0)

  Compute: wait(N) → SUCCESS ✓
  Compute: ... works ...
  Compute: pop(N)

State 3: CB empty (0/N)
  Reader: UNBLOCKS ✓
  Compute: wait(N) for next → BLOCKS ✗ (needs N ready, has 0)
```

**Pattern: Strict alternation. NO OVERLAP.**

### Double Buffer State Transitions

```
State 1: CB empty (0/2N)
  Reader: reserve(N) → SUCCESS ✓
  Reader: push(N) → CB has N/2N

State 2: CB half full (N/2N)
  Reader: reserve(N) → SUCCESS ✓ (N free slots available)
  Reader: push(N) → CB has 2N/2N

  Compute: wait(N) → SUCCESS ✓
  Compute: pop(N) → CB has N/2N

State 3: CB half full (N/2N)
  Reader: reserve(N) → SUCCESS ✓ (N free slots available)
  Reader: push(N) → CB has 2N/2N

  Compute: wait(N) → SUCCESS ✓ (N ready from earlier)
  Compute: pop(N) → CB has N/2N
```

**Pattern: Steady-state overlap. Reader and Compute can work simultaneously!**

### Blocking Point Checklist

| Question | Single Buffer | Double Buffer |
|----------|---------------|---------------|
| Can Reader reserve while CB has data? | NO (full) | YES (half full) |
| Can Compute wait while CB has space? | NO (empty) | YES (half full) |
| Do they alternate strictly? | YES | NO |
| Is there overlap possible? | NO | YES |

---

## Tools & Techniques

### Tool 1: CB State Tracking Table

Create a table to track CB state over time:

| Time | Operation | CB State | Ready | Free | Blocked? |
|------|-----------|----------|-------|------|----------|
| 0 | Init | 0/4 | 0 | 4 | - |
| 0 | R: reserve(4) | 0/4 | 0 | 4 | No |
| 5 | R: push(4) | 4/4 | 4 | 0 | - |
| 5 | R: reserve(4) | 4/4 | 4 | 0 | **YES** |
| 5 | C: wait(4) | 4/4 | 4 | 0 | No |
| 13 | C: pop(4) | 0/4 | 0 | 4 | - |
| 13 | R: unblocks | 0/4 | 0 | 4 | - |

### Tool 2: Execution Trace Template

```
=== BLOCK N ===

Reader:
  [T=X] cb_reserve_back(CB0, 4)
    State before: CB0=[4, U, R]
    Need: 4 free slots
    Have: C - U = Y free slots
    Result: [BLOCKS / SUCCESS]

  [T=X+5] cb_push_back(CB0, 4)
    State after: CB0=[4, U+4, R+4]

Compute:
  [T=Y] cb_wait_front(CB0, 4)
    State before: CB0=[4, U, R]
    Need: 4 ready tiles
    Have: R ready tiles
    Result: [BLOCKS / SUCCESS]

  [T=Y+8] cb_pop_front(CB0, 4)
    State after: CB0=[4, U-4, R-4]

=== END BLOCK N ===
```

### Tool 3: Critical Path Analysis

Identify the slowest stage:

```
Per-block times:
  Reader:  5 units
  Compute: 8 units
  Writer:  5 units

Critical path: Compute (8 units)

Best possible throughput (with perfect overlap):
  8 units per block

Actual throughput (single buffer):
  13 units per block (no overlap)

Efficiency: 8/13 = 62%
```

### Tool 4: Buffer Occupancy Graph

Plot CB occupancy over time:

**Single-buffered (ping-pong pattern):**
```
CB Occupancy (tiles)
4 |  ████    ████    ████
3 |  ████    ████    ████
2 |  ████    ████    ████
1 |  ████    ████    ████
0 |██    ████    ████    ██
  +------------------------> Time
    0   5  13  18  26

Legend:
  ████ = CB occupied
       = CB empty

Pattern: Square wave = ping-pong (no overlap)
```

**Double-buffered (steady around 50%):**
```
CB Occupancy (tiles)
8 |      ████████████████
7 |      ████████████████
6 |      ████████████████
5 |      ████████████████
4 |  ████████████████████
3 |  ████
2 |  ████
1 |  ████
0 |██████
  +------------------------> Time
    0   5  10  15  20

Pattern: Stays around 50% = good overlap
```

---

## Case Studies

### Case Study 1: Softmax Backward (Single Buffered)

**Configuration:**

```cpp
constexpr uint32_t tiles_per_block = 4;

CircularBufferConfig(
    tiles_per_block * tile_size,  // 4 tiles
    {{src0_cb_index, data_format}})
```

**Analysis:**

```
CB capacity: 4 tiles
Block size: 4 tiles
Ratio: 1:1 → Single buffering

Expected: NO overlap
```

**Verification:**

```
T=0-5:   Reader: reserve(4), read, push(4) → CB: 4/4
T=5:     Reader: reserve(4) → BLOCKS (need 4, have 0)
T=5-13:  Compute: wait(4), process, pop(4) → CB: 0/4
T=13:    Reader: UNBLOCKS

Conclusion: Strict ping-pong, NO overlap ✓
```

**Performance:**

```
Single block: 5 (read) + 8 (compute) = 13 units
Efficiency: 8/13 = 62%
```

---

### Case Study 2: Hypothetical Double Buffered

**Configuration:**

```cpp
constexpr uint32_t tiles_per_block = 4;
constexpr uint32_t num_buffers = 2;

CircularBufferConfig(
    num_buffers * tiles_per_block * tile_size,  // 8 tiles
    {{src0_cb_index, data_format}})
```

**Analysis:**

```
CB capacity: 8 tiles
Block size: 4 tiles
Ratio: 2:1 → Double buffering

Expected: Overlap possible
```

**Verification:**

```
Block 0:
T=0-5:   Reader: reserve(4), read, push(4) → CB: 4/8
T=5:     Reader: reserve(4) → SUCCESS (need 4, have 4) ✓

Block 1:
T=5-10:  Reader: read, push(4) → CB: 8/8
T=5-13:  Compute: wait(4), process Block 0

T=10:    Reader: reserve(4) → BLOCKS (need 4, have 0)
T=13:    Compute: pop(4) → CB: 4/8
T=13:    Reader: UNBLOCKS

Conclusion: Overlap achieved T=5-10 ✓
```

**Performance:**

```
Steady state: max(5, 8) = 8 units per block
Speedup: 13/8 = 1.625× faster
Efficiency: 8/8 = 100%
```

---

### Case Study 3: Multi-Pass with Accumulator

**Configuration:**

```cpp
// Input CBs: Single buffered
CircularBufferConfig(4 * tile_size, {{src0_cb, format}})

// Accumulator CB: 1 tile, retained
CircularBufferConfig(1 * tile_size, {{accum_cb, format}})
```

**Analysis:**

```
Pass 1: Input CBs used normally
        Accumulator CB: written each block, never popped
        Lifetime: Entire row

Pass 2: Input CBs used normally (re-read)
        Accumulator CB: read (broadcast), not popped
        Lifetime: Still entire row

After Pass 2: Accumulator CB finally popped
```

**Key Insight:**

```cpp
// WRONG:
for (block in row) {
    update_accumulator();
    cb_pop_front(accum_cb, 1);  // ✗ Don't do this!
}

// RIGHT:
for (block in row) {
    cb_wait_front(accum_cb, 1);
    update_accumulator();
    cb_pop_front(accum_cb, 1);
    cb_push_back(accum_cb, 1);  // ✓ Push updated value
}
// After entire row:
cb_pop_front(accum_cb, 1);  // ✓ Finally free it
```

---

## Common Patterns (Detailed)

### Pattern 1: Single-Buffered Ping-Pong

**Configuration:** CB capacity = block_size

**Characteristic:**
- Strict alternation
- No overlap
- High blocking time
- Simple to reason about

**Timeline:**
```
R: ████···· ████···· ████····
C: ····████ ····████ ····████
   Block 0   Block 1   Block 2
```

**When Used:**
- Minimal L1 memory priority
- Simple debugging desired
- Bandwidth is bottleneck anyway

---

### Pattern 2: Double-Buffered Pipeline

**Configuration:** CB capacity = 2 × block_size

**Characteristic:**
- Steady-state overlap
- Lower blocking time
- Better throughput
- More complex

**Timeline:**
```
R: ████████████████
C: ·····████████████████
   |<->| Overlap!
```

**When Used:**
- Performance critical
- L1 memory available
- Compute is bottleneck

---

### Pattern 3: Triple-Stage Pipeline (Reader-Compute-Writer)

**Configuration:**
- Input CB: capacity = 2 × block_size
- Output CB: capacity = 2 × block_size

**Characteristic:**
- Three-way overlap possible
- Maximum throughput
- Most complex

**Timeline:**
```
R: ████████████████████
C: ·····████████████████████████
W: ··········████████████████████
   |<-->|<-->| Triple overlap!
```

**When Used:**
- Maximum performance needed
- Abundant L1 memory
- All stages have similar latency

---

### Pattern 4: Accumulator Pattern

**Configuration:**
- Main CBs: Normal size
- Accumulator CB: 1 tile, retained across blocks

**Characteristic:**
- One CB with extended lifetime
- Survives across loop iterations
- Critical for multi-pass algorithms

**When Used:**
- Reductions across multiple blocks
- Multi-pass algorithms
- Global state needed

---

## Best Practices

### Practice 1: Start with CB Inventory

Before analyzing anything:

1. List all CBs
2. Note capacity, purpose, producer, consumer, **lifetime**
3. Identify block size
4. Calculate capacity/block ratio

### Practice 2: Simulate Don't Assume

- Don't trust architecture diagrams alone
- Trace actual code
- Manually simulate 2-3 blocks
- Track CB state transitions

### Practice 3: Draw Before Documenting

- Create Gantt charts
- Mark blocking points explicitly
- Use different colors for active vs. blocked
- Verify chart matches code trace

### Practice 4: Document Assumptions

```
Assumptions:
  - Reader: 5 units per block (based on X)
  - Compute: 8 units per block (based on Y)
  - CB capacity: 4 tiles (from program factory)
  - Block size: 4 tiles (from kernel code)

Conclusion:
  Single buffering → no overlap

Verification:
  - Traced code: lines X-Y
  - Simulated blocks 0-2
  - Timeline shows blocking at T=5, T=18
```

### Practice 5: Sanity Check Against Performance

```
If claiming overlap:
  Expected speedup: X%
  Measured speedup: Y%

If Y << X: Overlap might not be happening!
```

### Practice 6: Peer Review

- Have someone else verify your analysis
- Walk through execution step-by-step
- Explain blocking points
- Show CB state transitions

---

## Methodology Summary & Decision Tree

### The Analysis Workflow

1. Read Kernel Code
2. Extract CB Config
3. Calculate Capacity/Block Ratio
4. Does Ratio ≥ 2?
   - YES → Possible Overlap
   - NO → No Overlap
5. Trace Execution
6. Identify Blocking Points
7. Build Timeline
8. Create Gantt Chart
9. Calculate Performance
10. Verify Against Code
    - Match? → Document
    - No Match? → Find Error → Re-trace

### Quick Decision Tree

```
Q1: What is CB capacity C and block size B?
    → If C = B: Single buffering → NO overlap, go to Q2
    → If C = 2B: Double buffering → Possible overlap, go to Q3
    → If C > 2B: Multi-buffering → Complex, deep analysis needed

Q2: Single buffering confirmed?
    → Expect ping-pong
    → Timeline: Read + Compute per block (serialized)
    → Document: "Strict alternation, no overlap"

Q3: Double buffering, verify overlap:
    → Simulate first 3 blocks
    → Check if Reader can reserve while Compute works
    → If yes: Document overlap
    → If no: Find why (maybe other constraint)

Q4: Are there multiple CB stages (Reader→Compute→Writer)?
    → Analyze each CB independently
    → Determine critical path
    → Calculate end-to-end latency
```

---

## Sanity Checks

Before finalizing analysis, verify:

| Check | Expected | Verify |
|-------|----------|--------|
| CB never exceeds capacity | Used ≤ Capacity | ✓/✗ |
| No negative tiles | Used ≥ 0 | ✓/✗ |
| Push/pop balance | Total pushes = Total pops (eventually) | ✓/✗ |
| Blocking is symmetric | If A blocks B, then B unblocks A | ✓/✗ |

---

## Appendix: Formulas & Quick Reference

### Buffer Math

```
Capacity (C): Total tiles the CB can hold
Used (U): Tiles currently in CB
Ready (R): Tiles pushed but not popped (0 ≤ R ≤ U ≤ C)
Free (F): Slots available for writing (F = C - U)

Reserve succeeds if: F ≥ requested_tiles
Wait succeeds if: R ≥ requested_tiles
```

### Overlap Conditions

```
For Reader-Compute overlap:
  Required: CB_capacity ≥ 2 × block_size
  AND: Read_time < Compute_time (otherwise Reader waits anyway)

For Compute-Writer overlap:
  Required: Output_CB_capacity ≥ 2 × block_size
  AND: Compute_time < Write_time
```

### Performance Formulas

```
Single buffer throughput:
  Time_per_block = Read_time + Compute_time + Write_time

Double buffer throughput (steady state):
  Time_per_block = max(Read_time, Compute_time, Write_time)

Speedup:
  Single_time / Double_time

Efficiency:
  (Sum of stage times) / (Actual wall clock time)
```

### Common Configurations

| Pattern | Input CB | Output CB | Overlap? | Use Case |
|---------|----------|-----------|----------|----------|
| Minimal | 1× block | 1× block | None | Memory critical |
| Single | 1× block | 1× block | None | Simple, robust |
| Double | 2× block | 2× block | R-C, C-W | Performance |
| Triple | 2× block | 2× block | R-C-W | Maximum perf |

---

**End of Methodology Reference**
