# SDPA (Scaled Dot-Product Attention) Pipeline Analysis

## Overview
- **Operation**: Scaled Dot-Product Attention (SDPA)
- **Program Factory**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
- **Analysis Date**: 2026-01-08
- **Kernels Analyzed**:
  - Reader: `kernels/dataflow/reader_interleaved.cpp`
  - Compute: `kernels/compute/sdpa.cpp`
  - Writer: `kernels/dataflow/writer_interleaved.cpp`

## Algorithm Summary

SDPA implements the Flash Attention algorithm:
```
Output = softmax(Q @ K^T / sqrt(d_k)) @ V
```

The implementation processes data in chunks to fit within L1 memory:
- Q is chunked along sequence dimension (Sq_chunk_t tiles per chunk)
- K and V are chunked along sequence dimension (Sk_chunk_t tiles per chunk)
- For each Q chunk, iterate over all relevant K/V chunks (all for non-causal, lower triangle for causal)

---

## CB Configuration Summary

### Input CBs (Reader -> Compute)

| CB Index | Purpose | Capacity Formula | Double-Buffered | Producer | Consumer | Lifetime |
|----------|---------|------------------|-----------------|----------|----------|----------|
| c_0 | Q Input | `Sq_chunk_t * DHt * q_buffer_factor` | Conditional* | Reader | Compute | Per Q chunk |
| c_1 | K Input | `Sk_chunk_t * DHt * 2` | **YES** | Reader | Compute | Per K chunk |
| c_2 | V Input | `Sk_chunk_t * vDHt * 2` | **YES** | Reader | Compute | Per K chunk |
| c_3 | Mask | `Sq_chunk_t * Sk_chunk_t * 2` | **YES** | Reader/Writer | Compute | Per K chunk |
| c_4 | Attention Sink | `Sq_chunk_t` (if enabled) | NO | Reader | Compute | Per head |
| c_5 | Identity Scale | `1 tile` | NO | Writer | Compute | Program |
| c_6 | Page Table | `page_table_stick_size` (if chunked) | NO | Reader | Reader | Per batch |
| c_7 | Column Identity | `1 tile` | NO | Writer | Compute | Program |

*`q_buffer_factor = 2` if `q_per_core > 1`, otherwise `1`

### Intermediate CBs (Compute Internal)

| CB Index | Purpose | Capacity | Double-Buffered | Lifetime |
|----------|---------|----------|-----------------|----------|
| c_24 | QK Intermediate | `Sq_chunk_t * Sk_chunk_t` | NO | Per K chunk |
| c_25 | Output Intermediate A | `Sq_chunk_t * vDHt` | Ping-pong | Per K chunk |
| c_26 | Output Accumulator B | `Sq_chunk_t * vDHt` | Ping-pong | Per K chunk |
| c_27 | Current Max (A) | `Sq_chunk_t` | Ping-pong | Per K chunk |
| c_28 | Previous Max (B) | `Sq_chunk_t` | Ping-pong | Per K chunk |
| c_29 | Current Sum (A) | `Sq_chunk_t` | Ping-pong | Per K chunk |
| c_30 | Previous Sum (B) | `Sq_chunk_t` | Ping-pong | Per K chunk |
| c_31 | Exp Max Diff | `Sq_chunk_t` | NO | Per K chunk |

### Output CB (Compute -> Writer)

| CB Index | Purpose | Capacity | Double-Buffered | Producer | Consumer | Lifetime |
|----------|---------|----------|-----------------|----------|----------|----------|
| c_16 | Output | `Sq_chunk_t * vDHt` | NO | Compute | Writer | Per Q chunk |

---

## Blocking Point Analysis

### CB_0 (Q Input)

**Configuration**:
- Capacity = `Sq_chunk_t * DHt * q_buffer_factor` tiles
- Block size = `Sq_chunk_t * DHt` tiles (q_chunk_tiles)
- Ratio = `q_buffer_factor` (1 or 2)

**Producer (Reader)**:
```cpp
// reader_interleaved.cpp - read_chunk_with_padding()
cb_reserve_back(cb_id, num_tiles);  // num_tiles = Sq_chunk_t * DHt
// ... NOC reads ...
cb_push_back(cb_id, num_tiles);
```
- Blocks when: CB has >= `q_buffer_factor - 1` chunks (0 free slots when single-buffered)
- Unblocks when: Compute pops Q chunk

**Consumer (Compute)**:
```cpp
// sdpa.cpp - implicit in matmul_blocks()
cb_wait_front(in0_cb, ...);  // Waits for Q data
// ... after all K chunks processed ...
cb_pop_front(cb_q_in, q_chunk_tiles);  // Line 340
```
- Blocks when: No Q data available
- Unblocks when: Reader pushes Q chunk

**Pattern**:
- If `q_buffer_factor = 1`: Strict serialization between Q chunks
- If `q_buffer_factor = 2`: Can prefetch next Q while processing current

### CB_1 (K Input) - DOUBLE BUFFERED

**Configuration**:
- Capacity = `Sk_chunk_t * DHt * 2` tiles
- Block size = `Sk_chunk_t * DHt` tiles (k_chunk_tiles)
- Ratio = 2:1 (Double buffered)

**Producer (Reader)**:
```cpp
// reader_interleaved.cpp
read_chunk_with_padding<k_tile_bytes>(..., cb_k_in, ...);
// Inside read_chunk_with_padding:
cb_reserve_back(cb_id, num_tiles);
// ... NOC reads ...
cb_push_back(cb_id, num_tiles);
```

**Consumer (Compute)**:
```cpp
// sdpa.cpp - matmul_blocks()
cb_wait_front(in1_cb, K * N);  // Waits for K chunk
// ... matmul ...
cb_pop_front(in1_cb, K * N);
```

**Pattern**: Double buffering allows Reader to prefetch next K chunk while Compute processes current K chunk. **OVERLAP POSSIBLE**.

### CB_2 (V Input) - DOUBLE BUFFERED

**Configuration**:
- Capacity = `Sk_chunk_t * vDHt * 2` tiles
- Block size = `Sk_chunk_t * vDHt` tiles (v_chunk_tiles)
- Ratio = 2:1 (Double buffered)

**Pattern**: Same as CB_1 (K). Double buffering allows overlap between V read and compute.

### CB_3 (Mask) - DOUBLE BUFFERED

**Configuration**:
- Capacity = `Sq_chunk_t * Sk_chunk_t * 2` tiles
- Block size = `Sq_chunk_t * Sk_chunk_t` tiles
- Ratio = 2:1 (Double buffered)

**Producer (Reader - for provided mask)**:
```cpp
// reader_interleaved.cpp
cb_reserve_back(cb_mask_in, mask_chunk_tiles);
// ... NOC reads for mask ...
cb_push_back(cb_mask_in, mask_chunk_tiles);
```

**Producer (Writer - for generated causal/padded mask)**:
```cpp
// writer_interleaved.cpp - generate_mask()
cb_reserve_back(cb_mask_in, mask_size_tiles);
// ... generate mask in L1 ...
cb_push_back(cb_mask_in, mask_size_tiles);
```

**Consumer (Compute)**:
```cpp
// sdpa.cpp
add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
// add_block_inplace calls:
cb_wait_front(in1_cb, num_tiles);
// ...
cb_pop_front(in1_cb, num_tiles);
```

**Pattern**: Double buffered. Mask generation (Writer) or reading (Reader) can overlap with compute processing previous mask.

### CB_16 (Output) - SINGLE BUFFERED

**Configuration**:
- Capacity = `Sq_chunk_t * vDHt` tiles
- Block size = `Sq_chunk_t * vDHt` tiles (out_chunk_tiles)
- Ratio = 1:1 (Single buffered)

**Producer (Compute)**:
```cpp
// sdpa.cpp
mul_block_bcast_cols<Sq_chunk_t, vDHt>(alias_mm2_prev_out, alias_prev_sum, cb_out, false);
// Inside mul_block_bcast_cols:
cb_reserve_back(out_cb, num_tiles);
// ...
cb_push_back(out_cb, num_tiles);
```

**Consumer (Writer)**:
```cpp
// writer_interleaved.cpp
cb_wait_front(cb_out, out_chunk_tiles);
// ... NOC writes ...
cb_pop_front(cb_out, out_chunk_tiles);
```

**CRITICAL BLOCKING POINT**: Single-buffered output CB means:
- Compute BLOCKS on `cb_reserve_back(cb_out, ...)` until Writer finishes previous output
- Writer BLOCKS on `cb_wait_front(cb_out, ...)` until Compute finishes current output

**Pattern**: Strict ping-pong between Compute and Writer for output. **NO OVERLAP POSSIBLE**.

---

## Execution Flow Analysis

### High-Level Loop Structure

```
For each phase:
  For each batch (nb):
    [Reader: Load page table if chunked]
    For each head (nq):
      [Reader: Load attention sink if enabled]
      For each Q chunk (q_iter):
        [Reader: Load Q chunk] -> CB_0

        For each K chunk (k_chunk):  // Until k_low >= q_high (causal)
          [Reader: Load K chunk] -> CB_1
          [Reader: Load mask if provided] -> CB_3
          [Reader: Load V chunk] -> CB_2

          [Compute: QK = Q @ K^T]
          [Compute: Apply mask]
          [Compute: Softmax statistics update]
          [Compute: exp((QK - max) * scale)]
          [Compute: OUT_IM = softmax(QK) @ V]
          [Compute: Accumulate with rescaling]

        [Compute: Final normalization]
        [Compute: Push output] -> CB_16
        [Writer: Write output to DRAM]
```

### Detailed Synchronization Points

**Per Q chunk iteration:**

1. **Reader prepares Q**:
   - `cb_reserve_back(cb_q_in, q_chunk_tiles)` - May block if previous Q not consumed
   - NOC reads Q data
   - `cb_push_back(cb_q_in, q_chunk_tiles)` - Signals Q ready

2. **For each K/V chunk**:

   a. **Reader prepares K** (double-buffered):
      - `cb_reserve_back(cb_k_in, k_chunk_tiles)` - Blocks only if 2 K chunks pending
      - NOC reads K data (transposed)
      - `cb_push_back(cb_k_in, k_chunk_tiles)`

   b. **Reader prepares Mask** (if needed, double-buffered):
      - `cb_reserve_back(cb_mask_in, mask_chunk_tiles)`
      - NOC reads mask data
      - `cb_push_back(cb_mask_in, mask_chunk_tiles)`

   c. **Reader prepares V** (double-buffered):
      - `cb_reserve_back(cb_v_in, v_chunk_tiles)` - Blocks only if 2 V chunks pending
      - NOC reads V data
      - `cb_push_back(cb_v_in, v_chunk_tiles)`

   d. **Compute processes K/V chunk**:
      - `matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, ...)` - Waits for Q and K
      - `add_block_inplace(cb_qk_im, cb_mask_in, ...)` - Waits for mask
      - `reduce_c<MAX>(...)` - Compute statistics
      - `sub_exp_block_bcast_cols_inplace(...)` - Softmax
      - `matmul_blocks(cb_qk_im, cb_v_in, ...)` - S @ V
      - `cb_pop_front(cb_k_in, k_chunk_tiles)` - Free K slot
      - `cb_pop_front(cb_v_in, v_chunk_tiles)` - Free V slot (implicit in matmul)

3. **Compute finalizes output**:
   - `recip_block_inplace(alias_prev_sum, ...)` - 1/sum
   - `mul_block_bcast_cols<...>(alias_mm2_prev_out, alias_prev_sum, cb_out, ...)` - Normalize
   - `cb_push_back(cb_out, out_chunk_tiles)` - Implicit in mul_block_bcast_cols
   - `cb_pop_front(cb_q_in, q_chunk_tiles)` - Free Q slot

4. **Writer outputs result** (blocks Compute):
   - `cb_wait_front(cb_out, out_chunk_tiles)` - Wait for output ready
   - NOC writes to DRAM
   - `cb_pop_front(cb_out, out_chunk_tiles)` - Free output slot

---

## Execution Simulation

### Assumptions for Simulation
- Sq_chunk_t = 1 (1 tile per Q chunk row)
- Sk_chunk_t = 1 (1 tile per K chunk row)
- DHt = 4 (4 tiles for head dimension)
- vDHt = 4
- k_num_chunks = 3 (3 K/V chunks to process)
- q_buffer_factor = 1 (single Q buffer)

Tile counts:
- q_chunk_tiles = 1 * 4 = 4 tiles
- k_chunk_tiles = 1 * 4 = 4 tiles
- v_chunk_tiles = 1 * 4 = 4 tiles
- out_chunk_tiles = 1 * 4 = 4 tiles

CB Capacities:
- CB_0 (Q): 4 tiles (single buffer)
- CB_1 (K): 8 tiles (double buffer)
- CB_2 (V): 8 tiles (double buffer)
- CB_16 (Out): 4 tiles (single buffer)

### Simulation: First Q Chunk Processing

```
Legend:
  [C,U,R] = [Capacity, Used, Ready]
  R: Reader, C: Compute, W: Writer
  >>> = Active, ... = Blocked/Idle
```

**Initial State:**
```
T=0:   CB0=[4,0,0] CB1=[8,0,0] CB2=[8,0,0] CB16=[4,0,0]
       R: Active   C: Blocked  W: Blocked
```

**Phase 1: Reader loads Q chunk**
```
T=0:   R: cb_reserve_back(CB0, 4) -> SUCCESS (need 4, have 4 free)
T=0-5: R: NOC read Q data...
T=5:   R: cb_push_back(CB0, 4)
       CB0=[4,4,4]
```

**Phase 2: K-chunk 0 processing**
```
T=5:   R: cb_reserve_back(CB1, 4) -> SUCCESS (need 4, have 8 free)
T=5-8: R: NOC read K0 data...
T=8:   R: cb_push_back(CB1, 4)
       CB1=[8,4,4]

T=8:   R: cb_reserve_back(CB2, 4) -> SUCCESS (need 4, have 8 free)
T=8-11: R: NOC read V0 data...
T=11:  R: cb_push_back(CB2, 4)
       CB2=[8,4,4]

// Reader can now start K1 while Compute works on K0
T=11:  R: cb_reserve_back(CB1, 4) -> SUCCESS (need 4, have 4 free)
T=11-14: R: NOC read K1 data... (OVERLAPPED!)

T=5:   C: cb_wait_front(CB0, ...) -> SUCCESS in matmul_blocks
T=8:   C: cb_wait_front(CB1, 4) -> SUCCESS
T=8-16: C: matmul(Q, K0) -> QK...
       C: softmax updates...
T=11:  C: cb_wait_front(CB2, 4) -> SUCCESS
T=16-22: C: matmul(softmax, V0) -> OUT_IM...
T=22:  C: cb_pop_front(CB1, 4) -> CB1=[8,4,4] (K0 freed, K1 ready)
       C: cb_pop_front(CB2, 4) -> CB2=[8,0,0] (V0 freed)
```

**Phase 3: K-chunk 1 processing (OVERLAP VISIBLE)**
```
T=14:  R: cb_push_back(CB1, 4)
       CB1=[8,8,8]

T=14:  R: cb_reserve_back(CB2, 4) -> SUCCESS
T=14-17: R: NOC read V1 data...
T=17:  R: cb_push_back(CB2, 4)
       CB2=[8,4,4]

// Reader starts K2 while Compute processes K1
T=17:  R: cb_reserve_back(CB1, 4) -> BLOCKS! (need 4, have 0 after K1 not consumed)

T=22:  C: cb_wait_front(CB1, 4) -> SUCCESS (K1)
T=22-30: C: matmul(Q, K1)...
T=30:  C: cb_pop_front(CB1, 4) -> CB1=[8,4,4]
       R: UNBLOCKS!
```

**Phase 4: Final output (OUTPUT BOTTLENECK)**
```
// After all K chunks processed...
T=50:  C: Final normalization...
T=55:  C: cb_reserve_back(CB16, 4) -> SUCCESS
       C: mul_block_bcast_cols -> pack output
T=58:  C: cb_push_back(CB16, 4)
       CB16=[4,4,4]

T=58:  W: cb_wait_front(CB16, 4) -> SUCCESS
T=58-65: W: NOC write output...
T=65:  W: cb_pop_front(CB16, 4)
       CB16=[4,0,0]

// For NEXT Q chunk:
T=65:  C: Can now reserve CB16 for next output
       C: cb_reserve_back(CB16, 4) -> SUCCESS
```

**Key Observation**: The output CB16 is the serialization point between Compute and Writer.

---

## Timeline Visualization (Gantt Chart)

### Single Q Chunk with 3 K/V Chunks

```
Time:   0    5    10   15   20   25   30   35   40   45   50   55   60   65
        |    |    |    |    |    |    |    |    |    |    |    |    |    |
Reader: ████████████████████████████████·························
              Q    K0  V0  K1  V1  K2  V2
                      ▲▲▲▲overlap▲▲▲▲

Compute:     ·····██████████████████████████████████████████████████
                  Q@K0  S@V0 Q@K1  S@V1 Q@K2  S@V2 norm  out
                   ▲▲▲▲▲overlap▲▲▲▲▲

Writer:      ·······················································████████
                                                                    write

Legend:
  ████ = Active
  ···· = Blocked/Idle
  ▲▲▲▲ = Period where Reader/Compute overlap due to double buffering
```

### Multiple Q Chunks (showing output serialization)

```
Time:   0        20       40       60       80       100      120
        |        |        |        |        |        |        |
Reader: ████████████····████████████····████████████····
        Q0 K/V chunks   Q1 K/V chunks   Q2 K/V chunks

Compute:    ····████████████████████████████████████████████
                 process Q0  |wait|  process Q1  |wait|

Writer:         ············████████············████████····
                             out0                out1

        |<-- Q0 processing -->|<-- Q1 processing -->|
                              ▲
                              |
                    Output CB serialization point
                    (Writer must complete before
                     Compute can produce next)
```

---

## CB State Over Time (Detailed)

| Time | Event | CB0 (Q) | CB1 (K) | CB2 (V) | CB16 (Out) | Reader | Compute | Writer |
|------|-------|---------|---------|---------|------------|--------|---------|--------|
| 0 | Init | [4,0,0] | [8,0,0] | [8,0,0] | [4,0,0] | Active | Blocked | Blocked |
| 5 | Q pushed | [4,4,4] | [8,0,0] | [8,0,0] | [4,0,0] | Active | Blocked | Blocked |
| 8 | K0 pushed | [4,4,4] | [8,4,4] | [8,0,0] | [4,0,0] | Active | Active | Blocked |
| 11 | V0 pushed | [4,4,4] | [8,4,4] | [8,4,4] | [4,0,0] | Active | Active | Blocked |
| 14 | K1 pushed | [4,4,4] | [8,8,8] | [8,4,4] | [4,0,0] | Active | Active | Blocked |
| 17 | V1 pushed, R blocks | [4,4,4] | [8,8,8] | [8,8,8] | [4,0,0] | Blocked | Active | Blocked |
| 22 | C pops K0,V0 | [4,4,4] | [8,4,4] | [8,4,4] | [4,0,0] | Active | Active | Blocked |
| ... | Processing | ... | ... | ... | ... | ... | ... | ... |
| 55 | Out pushed | [4,0,0] | [8,0,0] | [8,0,0] | [4,4,4] | Blocked | Blocked | Active |
| 65 | Out popped | [4,0,0] | [8,0,0] | [8,0,0] | [4,0,0] | Active | Active | Blocked |

---

## Performance Analysis

### Component Timing Estimates

| Component | Estimated Time (per unit) | Notes |
|-----------|---------------------------|-------|
| Reader: Q chunk load | ~5 units | NOC read latency, q_chunk_tiles |
| Reader: K chunk load | ~3 units | NOC read latency, k_chunk_tiles |
| Reader: V chunk load | ~3 units | NOC read latency, v_chunk_tiles |
| Reader: Mask load | ~2 units | NOC read latency (if provided) |
| Compute: Q @ K^T matmul | ~8 units | Matrix multiplication |
| Compute: Softmax + stats | ~4 units | exp, max, sum operations |
| Compute: S @ V matmul | ~6 units | Matrix multiplication |
| Compute: Accumulate/rescale | ~2 units | Per K chunk after first |
| Compute: Final norm | ~3 units | recip, mul |
| Writer: Output write | ~7 units | NOC write latency |

### Per-Q-Chunk Analysis

**Total per Q chunk (with k_num_chunks K/V iterations)**:

```
Reader time per Q chunk:
  = Q_load + k_num_chunks * (K_load + V_load + Mask_load)
  = 5 + 3 * (3 + 3 + 2) = 5 + 24 = 29 units

Compute time per Q chunk:
  = k_num_chunks * (QK_matmul + softmax + SV_matmul + accumulate) + final_norm
  = 3 * (8 + 4 + 6 + 2) + 3 = 60 + 3 = 63 units

Writer time per Q chunk:
  = output_write = 7 units
```

### Throughput Analysis

**Without Double Buffering (Theoretical)**:
```
Sequential execution:
Time per Q chunk = Reader + Compute + Writer
                 = 29 + 63 + 7 = 99 units
Throughput = 1 Q chunk / 99 units
```

**With Double Buffering on K/V (Actual)**:
```
K/V Read-Compute Overlap:
- Reader can prefetch K(i+1) while Compute processes K(i)
- Overlap saves: k_num_chunks * K_load = 3 * 3 = 9 units

Effective time:
Reader time = 5 + k_num_chunks * (V_load + Mask_load) = 5 + 3 * 5 = 20 units
  (K loads hidden)

However, Compute is the bottleneck at 63 units, so:
Effective time = max(Reader, Compute) + Writer (serialized)
               = max(20, 63) + 7 = 63 + 7 = 70 units

Throughput = 1 Q chunk / 70 units
Speedup from K/V double buffering = 99/70 = 1.41x
```

**Output CB Bottleneck Impact**:
```
With double-buffered output CB:
- Compute could start next Q's processing while Writer writes current output
- Potential overlap: Writer time = 7 units hidden

Hypothetical time = max(Reader, Compute, Writer) = 63 units
Speedup potential = 70/63 = 1.11x additional
Total potential speedup = 99/63 = 1.57x
```

### Efficiency Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Current Efficiency | 63/99 = 64% | Time doing compute / total sequential time |
| With K/V DB | 63/70 = 90% | Compute utilization with double buffering |
| Reader Utilization | 20/70 = 29% | Reader active time / total cycle |
| Compute Utilization | 63/70 = 90% | Compute active time / total cycle |
| Writer Utilization | 7/70 = 10% | Writer active time / total cycle |

**Bottleneck**: Compute kernel (63 units per Q chunk)

**Limiting Factors**:
1. Compute-bound: Matrix multiplications dominate
2. Output CB serialization: Writer blocks next Compute output
3. Q single-buffering: Cannot prefetch next Q chunk

---

## Optimization Recommendations

### 1. Double-Buffer Output CB (cb_out / c_16)

**Current**: `out0_t = Sq_chunk_t * vDHt` (single buffer)

**Recommendation**:
```cpp
uint32_t out0_t = Sq_chunk_t * vDHt * 2;  // Double buffer
```

**Expected Impact**:
- Allows Compute to produce next output while Writer writes current
- Potential 11% improvement (7 units hidden per Q chunk)
- Requires: Additional L1 memory for output buffer

**Trade-off**:
- L1 memory increase: `Sq_chunk_t * vDHt * out_tile_size` bytes
- For typical config (Sq_chunk_t=1, vDHt=4, bfloat16): 4 * 2KB = 8KB additional

### 2. Increase Q Buffer Factor When Possible

**Current**: `q_buffer_factor = (q_per_core > 1) ? 2 : 1`

**Observation**: Q is single-buffered when `q_per_core == 1`

**Recommendation**: Consider always using q_buffer_factor = 2 when L1 permits

**Expected Impact**:
- Reader can prefetch next Q while Compute processes current
- Reduces Q load latency from critical path
- ~5 units savings when Reader faster than Compute

### 3. Granular Output Pushing

**Current**: Output is pushed after all K chunks processed

**Observation**: The compute kernel already does granular push in some operations:
```cpp
// In sub_exp_block_bcast_cols_inplace
cb_push_back(in0_cb, cols);  // Granular write output
```

**Recommendation**: Consider streaming partial outputs earlier if algorithm permits (Flash Attention v2 style)

### 4. Matmul Subblock Optimization

**Current**: Subblock sizes chosen based on DST register capacity

**Observation**:
```cpp
uint32_t qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
uint32_t qk_out_subblock_h = (qk_out_subblock_w == Sk_chunk_t)
    ? (std::min(Sq_chunk_t, dst_size / qk_out_subblock_w)) : 1;
```

**Recommendation**: Profile different subblock configurations for specific workload sizes

### 5. Memory Layout Optimization

**Current**: K is transposed during read for matmul efficiency

**Observation**: Transpose adds overhead to read path

**Alternative**: Pre-transpose K in DRAM or use non-transposed matmul variant

---

## Double-Buffering Effectiveness Summary

| Buffer | Double-Buffered? | Overlap Achieved? | Impact |
|--------|------------------|-------------------|--------|
| CB_0 (Q) | Conditional | Partial | When q_per_core > 1 only |
| CB_1 (K) | **YES** | **YES** | Hides K load latency |
| CB_2 (V) | **YES** | **YES** | Hides V load latency |
| CB_3 (Mask) | **YES** | **YES** | Hides mask load latency |
| CB_16 (Out) | **NO** | **NO** | **SERIALIZATION POINT** |

**Overall Assessment**:
- K/V/Mask double buffering is effective
- Output CB is the primary serialization bottleneck
- Q buffering is conditional and could be improved

---

## Verification Checklist

- [x] All CBs identified with correct capacities
- [x] Block sizes extracted from kernel code
- [x] Capacity/block ratio calculated for each CB
- [x] Lifetime identified for each CB (Block/Row/Program)
- [x] Simulation traced for multiple blocks
- [x] Blocking points explicitly identified
- [x] Timeline matches code trace
- [x] Performance calculations verified

---

## Code References

### CB Creation (Program Factory)
- Q CB: `sdpa_program_factory.cpp:525-528`
- K CB: `sdpa_program_factory.cpp:530-532`
- V CB: `sdpa_program_factory.cpp:534-536`
- Mask CB: `sdpa_program_factory.cpp:541-544`
- Output CB: `sdpa_program_factory.cpp:615-618`

### Reader Kernel Blocking Points
- Q reserve/push: `reader_interleaved.cpp` via `read_chunk_with_padding()` in `dataflow_common.hpp:102,135`
- K reserve/push: `reader_interleaved.cpp:227-237` via helper
- V reserve/push: `reader_interleaved.cpp:291-301` via helper
- Mask reserve/push: `reader_interleaved.cpp:247,270`

### Compute Kernel Blocking Points
- Q wait: `sdpa.cpp:149` via `matmul_blocks()` -> `compute_common.hpp:659`
- K wait/pop: `compute_common.hpp:655,694`
- V wait/pop: `compute_common.hpp:655,694` (same in second matmul)
- Mask wait/pop: `compute_common.hpp:374-384` in `add_block_inplace()`
- Output reserve/push: `compute_common.hpp:275,296` in `mul_block_bcast_cols()`
- Q pop: `sdpa.cpp:340`

### Writer Kernel Blocking Points
- Output wait: `writer_interleaved.cpp:143`
- Output pop: `writer_interleaved.cpp:159`
- Mask generation: `dataflow_common.hpp:452,575` in `generate_mask()`

---

## Appendix: Typical Configurations

### Example 1: Llama-style Attention (GQA)
```
B = 1, NQH = 32, NKH = 8 (GQA ratio 4:1)
Sq = 2048, Sk = 2048
DH = 128, q_chunk_size = 128, k_chunk_size = 128

Derived:
Sq_chunk_t = 4, Sk_chunk_t = 4, DHt = 4
q_num_chunks = 16, k_num_chunks = 16
q_chunk_tiles = 16, k_chunk_tiles = 16, v_chunk_tiles = 16

CB Sizes:
CB_0 (Q): 16-32 tiles (512KB-1MB with bfloat16)
CB_1 (K): 32 tiles (1MB)
CB_2 (V): 32 tiles (1MB)
CB_16 (Out): 16 tiles (512KB)
```

### Example 2: Short Sequence Decoding
```
B = 32, NQH = 32, NKH = 32
Sq = 1, Sk = 512
DH = 128, q_chunk_size = 32, k_chunk_size = 32

Derived:
Sq_chunk_t = 1, Sk_chunk_t = 1, DHt = 4
q_num_chunks = 1, k_num_chunks = 16

CB Sizes (minimal):
CB_0 (Q): 4 tiles
CB_1 (K): 8 tiles
CB_2 (V): 8 tiles
CB_16 (Out): 4 tiles
```
