# Binary Operation CB Contract Reference

> **Purpose:** Complete documentation of circular buffer behavior for all input mode × broadcast dimension combinations in `binary_op_helpers.hpp`

---

## Quick Reference

| Shape | A Tiles Required | B Tiles Required | Output Tiles Produced |
|-------|------------------|------------------|----------------------|
| `grid(Ht, Wt)` | Ht × Wt | Depends on broadcast | Ht × Wt |
| `single()` | 1 | 1 (or depends on broadcast) | 1 |
| `row(Wt)` | Wt | Wt (or depends on broadcast) | Wt |
| `col(Ht)` | Ht | Ht (or depends on broadcast) | Ht |

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `wait(N)` | `cb_wait_front(cb, N)` |
| `pop(N)` | `cb_pop_front(cb, N)` |
| `reserve(N)` | `cb_reserve_back(cb, N)` |
| `push(N)` | `cb_push_back(cb, N)` |
| `—` | No operation performed |
| `caller` | Caller is responsible for this operation |
| `persist` | Tiles remain in CB after function returns |

---

## 1. BroadcastDim::NONE (Element-wise)

**Operation:** `C[h,w] = A[h,w] op B[h,w]`

| | Input A | Input B | Output |
|---|---------|---------|--------|
| **Tiles Required** | Ht × Wt | Ht × Wt | Ht × Wt |

### STREAMING Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Per tile (Ht×Wt times) | wait(1), pop(1) | wait(1), pop(1) | reserve(1), push(1) |
| **Total wait** | Ht × Wt | Ht × Wt | — |
| **Total pop** | Ht × Wt | Ht × Wt | — |
| **Total push** | — | — | Ht × Wt |

### STREAMING_BATCHED Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Per chunk (ceil(Wt/DEST) times per row) | wait(chunk), pop(chunk) | wait(chunk), pop(chunk) | reserve(chunk), push(chunk) |
| **Total wait** | Ht × Wt | Ht × Wt | — |
| **Total pop** | Ht × Wt | Ht × Wt | — |
| **Total push** | — | — | Ht × Wt |

### PRELOADED Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Before call | caller: wait(Ht×Wt) | caller: wait(Ht×Wt) | — |
| Start of function | — | — | reserve(Ht×Wt) |
| Per tile | indexed access | indexed access | indexed pack |
| End of function | — | — | push(Ht×Wt) |
| After call | caller: pop(Ht×Wt) | caller: pop(Ht×Wt) | — |
| **Total wait** | caller | caller | — |
| **Total pop** | caller | caller | — |
| **Total push** | — | — | Ht × Wt |

### PERSISTENT Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Start of function | wait(Ht×Wt) | wait(Ht×Wt) | — |
| Per tile | indexed access | indexed access | reserve(1), push(1) |
| End of function | **no pop** (persist) | **no pop** (persist) | — |
| **Total wait** | Ht × Wt | Ht × Wt | — |
| **Total pop** | 0 (persist) | 0 (persist) | — |
| **Total push** | — | — | Ht × Wt |

---

## 2. BroadcastDim::ROW (Row Broadcast)

**Operation:** `C[h,w] = A[h,w] op B[w]` — B has shape [1, Wt], broadcasts across all Ht rows

| | Input A | Input B | Output |
|---|---------|---------|--------|
| **Tiles Required** | Ht × Wt | Wt | Ht × Wt |

### STREAMING Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Start of function | — | wait(Wt) | — |
| Per tile (Ht×Wt times) | wait(1), pop(1) | indexed access | reserve(1), push(1) |
| End of function | — | **no pop** (persist) | — |
| **Total wait** | Ht × Wt | Wt | — |
| **Total pop** | Ht × Wt | **0 (persist)** | — |
| **Total push** | — | — | Ht × Wt |

### STREAMING_BATCHED Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Start of function | — | wait(Wt) | — |
| Per chunk | wait(chunk), pop(chunk) | indexed access | reserve(chunk), push(chunk) |
| End of function | — | **no pop** (persist) | — |
| **Total wait** | Ht × Wt | Wt | — |
| **Total pop** | Ht × Wt | **0 (persist)** | — |
| **Total push** | — | — | Ht × Wt |

### PRELOADED Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Before call | caller: wait(Ht×Wt) | caller: wait(Wt) | — |
| Start of function | — | — | reserve(Ht×Wt) |
| Per tile | indexed access | indexed access | indexed pack |
| End of function | — | — | push(Ht×Wt) |
| After call | caller: pop(Ht×Wt) | **no pop** (persist) | — |
| **Total wait** | caller | caller | — |
| **Total pop** | caller | **0 (persist)** | — |
| **Total push** | — | — | Ht × Wt |

### PERSISTENT Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Start of function | wait(Ht×Wt) | wait(Wt) | — |
| Per tile | indexed access | indexed access | reserve(1), push(1) |
| End of function | **no pop** (persist) | **no pop** (persist) | — |
| **Total wait** | Ht × Wt | Wt | — |
| **Total pop** | **0 (persist)** | **0 (persist)** | — |
| **Total push** | — | — | Ht × Wt |

---

## 3. BroadcastDim::COL (Column Broadcast)

**Operation:** `C[h,w] = A[h,w] op B[h]` — B has shape [Ht, 1], broadcasts across all Wt columns

| | Input A | Input B | Output |
|---|---------|---------|--------|
| **Tiles Required** | Ht × Wt | Ht | Ht × Wt |

### STREAMING Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Per row (Ht times): | | | |
| — Row start | — | wait(1) | — |
| — Per tile (Wt times) | wait(1), pop(1) | (reuse same tile) | reserve(1), push(1) |
| — Row end | — | pop(1) | — |
| **Total wait** | Ht × Wt | Ht | — |
| **Total pop** | Ht × Wt | Ht | — |
| **Total push** | — | — | Ht × Wt |

### STREAMING_BATCHED Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Per row (Ht times): | | | |
| — Row start | — | wait(1) | — |
| — Per chunk | wait(chunk), pop(chunk) | (reuse same tile) | reserve(chunk), push(chunk) |
| — Row end | — | pop(1) | — |
| **Total wait** | Ht × Wt | Ht | — |
| **Total pop** | Ht × Wt | Ht | — |
| **Total push** | — | — | Ht × Wt |

### PRELOADED Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Before call | caller: wait(Ht×Wt) | caller: wait(Ht) | — |
| Start of function | — | — | reserve(Ht×Wt) |
| Per tile | indexed access | indexed access | indexed pack |
| End of function | — | — | push(Ht×Wt) |
| After call | caller: pop(Ht×Wt) | caller: pop(Ht) | — |
| **Total wait** | caller | caller | — |
| **Total pop** | caller | caller | — |
| **Total push** | — | — | Ht × Wt |

### PERSISTENT Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Start of function | wait(Ht×Wt) | wait(Ht) | — |
| Per tile | indexed access | indexed access | reserve(1), push(1) |
| End of function | **no pop** (persist) | **no pop** (persist) | — |
| **Total wait** | Ht × Wt | Ht | — |
| **Total pop** | **0 (persist)** | **0 (persist)** | — |
| **Total push** | — | — | Ht × Wt |

---

## 4. BroadcastDim::SCALAR (Scalar Broadcast)

**Operation:** `C[h,w] = A[h,w] op B[0,0]` — B is single tile, broadcasts to all elements

| | Input A | Input B | Output |
|---|---------|---------|--------|
| **Tiles Required** | Ht × Wt | 1 | Ht × Wt |

### STREAMING Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Start of function | — | wait(1) | — |
| Per tile (Ht×Wt times) | wait(1), pop(1) | (reuse same tile) | reserve(1), push(1) |
| End of function | — | **no pop** (persist) | — |
| **Total wait** | Ht × Wt | 1 | — |
| **Total pop** | Ht × Wt | **0 (persist)** | — |
| **Total push** | — | — | Ht × Wt |

### STREAMING_BATCHED Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Start of function | — | wait(1) | — |
| Per chunk | wait(chunk), pop(chunk) | (reuse same tile) | reserve(chunk), push(chunk) |
| End of function | — | **no pop** (persist) | — |
| **Total wait** | Ht × Wt | 1 | — |
| **Total pop** | Ht × Wt | **0 (persist)** | — |
| **Total push** | — | — | Ht × Wt |

### PRELOADED Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Before call | caller: wait(Ht×Wt) | caller: wait(1) | — |
| Start of function | — | — | reserve(Ht×Wt) |
| Per tile | indexed access | index 0 always | indexed pack |
| End of function | — | — | push(Ht×Wt) |
| After call | caller: pop(Ht×Wt) | **no pop** (persist) | — |
| **Total wait** | caller | caller | — |
| **Total pop** | caller | **0 (persist)** | — |
| **Total push** | — | — | Ht × Wt |

### PERSISTENT Mode

| Phase | Input A | Input B | Output |
|-------|---------|---------|--------|
| Start of function | wait(Ht×Wt) | wait(1) | — |
| Per tile | indexed access | index 0 always | reserve(1), push(1) |
| End of function | **no pop** (persist) | **no pop** (persist) | — |
| **Total wait** | Ht × Wt | 1 | — |
| **Total pop** | **0 (persist)** | **0 (persist)** | — |
| **Total push** | — | — | Ht × Wt |

---

## 5. Summary Tables

### 5.1 B-Input Persistence Rules

| Broadcast Dim | B Tiles Needed | B Popped? | Reason |
|---------------|----------------|-----------|--------|
| NONE | Ht × Wt | Yes (follows A) | One B tile per A tile |
| ROW | Wt | **No (persist)** | Same Wt tiles reused for all Ht rows |
| COL | Ht | Yes (1 per row) | Each row needs different B tile |
| SCALAR | 1 | **No (persist)** | Same tile reused for all Ht×Wt tiles |

### 5.2 Total Tiles Waited/Popped by Mode

**For grid(Ht, Wt) shape:**

| Mode | A Wait | A Pop | B Wait (NONE) | B Pop (NONE) | B Wait (ROW) | B Pop (ROW) | B Wait (COL) | B Pop (COL) | B Wait (SCALAR) | B Pop (SCALAR) |
|------|--------|-------|---------------|--------------|--------------|-------------|--------------|-------------|-----------------|----------------|
| STREAMING | Ht×Wt | Ht×Wt | Ht×Wt | Ht×Wt | Wt | **0** | Ht | Ht | 1 | **0** |
| STREAMING_BATCHED | Ht×Wt | Ht×Wt | Ht×Wt | Ht×Wt | Wt | **0** | Ht | Ht | 1 | **0** |
| PRELOADED | caller | caller | caller | caller | caller | **0** | caller | caller | caller | **0** |
| PERSISTENT | Ht×Wt | **0** | Ht×Wt | **0** | Wt | **0** | Ht | **0** | 1 | **0** |

### 5.3 Output Behavior by Mode

| Mode | Output Reserve | Output Push | Pattern |
|------|----------------|-------------|---------|
| STREAMING | 1 per tile | 1 per tile | Immediate |
| STREAMING_BATCHED | chunk per chunk | chunk per chunk | Chunked |
| PRELOADED | Ht×Wt upfront | Ht×Wt at end | Bulk |
| PERSISTENT | 1 per tile | 1 per tile | Immediate |

---

## 6. CB Sizing Requirements

### Minimum CB Sizes

| CB | STREAMING | STREAMING_BATCHED | PRELOADED | PERSISTENT |
|----|-----------|-------------------|-----------|------------|
| A (input) | 1 tile | DEST_LIMIT tiles | Ht × Wt tiles | Ht × Wt tiles |
| B (NONE) | 1 tile | DEST_LIMIT tiles | Ht × Wt tiles | Ht × Wt tiles |
| B (ROW) | Wt tiles | Wt tiles | Wt tiles | Wt tiles |
| B (COL) | 1 tile | 1 tile | Ht tiles | Ht tiles |
| B (SCALAR) | 1 tile | 1 tile | 1 tile | 1 tile |
| Output | 1 tile | DEST_LIMIT tiles | Ht × Wt tiles | 1 tile |

### DEST_LIMIT Values

| Sync Mode | Accum Mode | DEST_LIMIT |
|-----------|------------|------------|
| SyncFull | 16-bit | 16 tiles |
| SyncFull | 32-bit | 8 tiles |
| SyncHalf | 16-bit | 8 tiles |
| SyncHalf | 32-bit | 4 tiles |

---

## 7. Reader/Writer Kernel Coordination

### Reader Kernel Requirements

| Mode | A Tiles | B Tiles (depends on broadcast) |
|------|---------|-------------------------------|
| STREAMING | Push 1 at a time, can be slower than compute | Push required amount before compute starts (or stream for COL) |
| STREAMING_BATCHED | Push chunks of DEST_LIMIT | Same as STREAMING |
| PRELOADED | Push all Ht×Wt before compute starts | Push all required before compute starts |
| PERSISTENT | Push all Ht×Wt before compute starts | Push all required before compute starts |

### Writer Kernel Requirements

| Mode | Output Tiles |
|------|--------------|
| STREAMING | Pop 1 at a time as they're produced |
| STREAMING_BATCHED | Pop chunks of DEST_LIMIT |
| PRELOADED | Pop all Ht×Wt after compute completes |
| PERSISTENT | Pop 1 at a time as they're produced |

---

## 8. Common Patterns

### Pattern 1: Simple Element-wise Add
```cpp
// Reader pushes A and B tiles one at a time
// Compute processes one at a time
// Writer pops output one at a time
compute_kernel_lib::add(cb_a, cb_b, cb_out,
    BinaryTileShape::grid(Ht, Wt));
```
- A: wait Ht×Wt (1 at a time), pop Ht×Wt
- B: wait Ht×Wt (1 at a time), pop Ht×Wt
- Out: push Ht×Wt

### Pattern 2: Bias Addition (Row Broadcast)
```cpp
// Reader pushes all Wt bias tiles once, then streams data
// Bias tiles persist for reuse across all rows
compute_kernel_lib::add<BroadcastDim::ROW>(cb_data, cb_bias, cb_out,
    BinaryTileShape::grid(Ht, Wt));
```
- A: wait Ht×Wt (1 at a time), pop Ht×Wt
- B: wait Wt once at start, **never popped**
- Out: push Ht×Wt

### Pattern 3: Scalar Multiply (e.g., scaling)
```cpp
// Reader pushes scalar tile once, then streams data
// Scalar persists for entire operation
compute_kernel_lib::mul<BroadcastDim::SCALAR>(cb_data, cb_scalar, cb_out,
    BinaryTileShape::grid(Ht, Wt));
```
- A: wait Ht×Wt (1 at a time), pop Ht×Wt
- B: wait 1 once at start, **never popped**
- Out: push Ht×Wt

### Pattern 4: Pre-loaded Data Processing
```cpp
// All tiles already in CBs before compute starts
// Caller manages wait/pop externally
cb_wait_front(cb_a, Ht * Wt);
cb_wait_front(cb_b, Ht * Wt);

compute_kernel_lib::add<BroadcastDim::NONE, BinaryInputMode::PRELOADED>(
    cb_a, cb_b, cb_out, BinaryTileShape::grid(Ht, Wt));

cb_pop_front(cb_a, Ht * Wt);
cb_pop_front(cb_b, Ht * Wt);
```
- A: caller waits, caller pops
- B: caller waits, caller pops
- Out: bulk reserve/push by library

### Pattern 5: Persistent for Reuse
```cpp
// Tiles kept for subsequent operations
compute_kernel_lib::mul<BroadcastDim::NONE, BinaryInputMode::PERSISTENT>(
    cb_data, cb_weights, cb_temp, BinaryTileShape::grid(Ht, Wt));

// Same data used again - still in CB!
compute_kernel_lib::add<BroadcastDim::SCALAR, BinaryInputMode::PERSISTENT>(
    cb_temp, cb_bias, cb_out, BinaryTileShape::grid(Ht, Wt));
```
- First op: wait all, no pop (persist)
- Second op: data still available

---

## 9. Important Notes

### B-Input After Function Returns

| Broadcast | After STREAMING/BATCHED | After PRELOADED | After PERSISTENT |
|-----------|-------------------------|-----------------|------------------|
| NONE | B fully consumed (popped) | Caller must pop | B persists |
| ROW | **B persists (Wt tiles)** | **B persists** | B persists |
| COL | B fully consumed (popped) | Caller must pop | B persists |
| SCALAR | **B persists (1 tile)** | **B persists** | B persists |

### Caller Responsibility Summary

| Mode | A Wait | A Pop | B Wait | B Pop | Out |
|------|--------|-------|--------|-------|-----|
| STREAMING | Library | Library | Library | Depends on bcast | Library |
| STREAMING_BATCHED | Library | Library | Library | Depends on bcast | Library |
| PRELOADED | **Caller** | **Caller** | **Caller** | **Caller** (except ROW/SCALAR) | Library |
| PERSISTENT | Library | **Never** | Library | **Never** | Library |
