# moe_gpt Op – Handover Document

**Last updated:** 2026-03-10
**Author:** handover from bring-up session
**Repo:** `/data/handrews/tt-metal` (main branch)

---

## 1. What is `moe_gpt`?

`moe_gpt` is a **fused sparse MoE compute kernel** designed for GPT-OSS on Galaxy hardware (4×8 = 32 chips). It replaces the dense all_to_all flow with a sparse fused pipeline:

```
OLD (dense):
  all_to_all_dispatch → repeat input × E experts → dense batched matmul (W1/W3/W2)
  → all_to_all_combine → weighted sum → all_reduce

NEW (fused/sparse):
  all_to_all_dispatch_metadata → moe_gpt → selective_reduce_combine → all_reduce
```

Within a single kernel invocation, `moe_gpt` does:
1. **Tilize** – NCRISC and BRISC read the sparse token buffer, identify which tokens map to local experts, and build sorted activation metadata
2. **W0/W1 matmul** – ring-pipelined matmul of tokens against fused gate+up weights (produces pre-SwiGLU output, width = 2×N)
3. **SwiGLU** – element-wise activation on the compute core
4. **A2A ring** – all-to-all ring exchange of SwiGLU outputs across the 12 matmul cores
5. **W2 matmul** – down-projection matmul
6. **Combine** – writes untilized outputs to a 4×3 core BLOCK_SHARDED combine buffer

`ttnn::experimental::moe_gpt` is registered at:
- C++ op: `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/`
- Python: `ttnn.experimental.moe_gpt(...)`

---

## 2. Relationship to `moe_compute` (DeepSeek op)

`moe_gpt` is a **fork and adaptation** of `moe_compute` (at `ccl/moe_compute/`). The two share the same overall kernel topology and many implementation patterns.

### Similarities
| Aspect | Both ops |
|---|---|
| Kernel files | `dm0.cpp`, `dm1.cpp`, `tilize_reader.cpp` (NCRISC), `tilize_writer.cpp` (BRISC), `tilize_compute.cpp`, `compute.cpp`, `combine_dm1.cpp` |
| Data flow | Dispatch drain → tilize → matmul ring → combine output |
| CB aliasing | HEIGHT_SHARDED indices/scores CBs aliased to dispatch drain's L1 buffer |
| Expert mapping format | `[total_devices, num_experts]` uint16 DRAM; `mapping[d, e] = device_id_owning_expert_e` |
| Outputs | [0] per-expert token counts, [1] activation metadata, [2] e_t flat token indices, [3/4] combine output |
| Combine grid | 4×3 BLOCK_SHARDED core rectangle avoiding matmul (DRAM-bank) cores |
| Semaphore protocol | `tilize_partial_metadata_ready`, `tilize_chunk_ready`, `previous_chunk_sent`, `initial_gather` |
| Sentinels | `token_id = 0xFFFFFFFF` marks unwritten activation rows; `k_idx = selected_k+1` marks unactivated experts |

### Differences (moe_gpt vs moe_compute)
| Aspect | `moe_gpt` (GPT-OSS) | `moe_compute` (DeepSeek) |
|---|---|---|
| Model | GPT-OSS 20b / 120b | DeepSeek |
| K = N = hidden/intermediate | 2880 | Different dims |
| Experts per device (E) | 4 (20b: 16 experts in ring of 4) | Different |
| Matmul cores | 12 (DRAM-bank aligned) | Different count |
| W0/W1 layout | Fused `[K, 2N]` (gate+up together for SwiGLU) | May differ |
| A2A ring step | `NUM_CORES=12`, `TOKENS_PER_CHUNK=32` | Different |
| Common header | `moe_gpt_ring_common.h` (namespace `moe_gpt_ring`) | `moe_ring_common.h` (namespace `moe_ring`) |
| Weight tiles/core | 7 or 8 (90 tiles ÷ 12 cores = 7.5) with two layout variants (A=boundary-opt, B=even) | Different |
| Operation name | `ttnn::experimental::moe_gpt` | `ttnn::experimental::prim::moe_compute` |

---

## 3. Hardware Configuration (Galaxy 4×8)

```
Total devices:        32  (4 rows × 8 columns)
cluster_axis=0:       Column rings (4 devices each, 8 independent rings)
cluster_axis=1:       Row rings (8 devices each, 4 independent rings)
Dispatch ring:        cluster_axis=0, num_links=4
A2A ring (in kernel): 12 matmul cores per device
TP axis:              cluster_axis=1 (used for final all_reduce, num_links=1)
```

For GPT-OSS 20b:
- `experts_total = 128`, `experts_per_ring = 16`, `experts_per_device = 4`
- `tokens_per_device (M) = 32`, `total_ring_tokens = 128`
- `K = N = 2880`, `selected_k = 4`

---

## 4. Op Signature and I/O

### Python
```python
moe_gpt_outputs = ttnn.experimental.moe_gpt(
    input_tensor,       # [total_tokens, K] bfloat16 L1 (sparse buffer from dispatch)
    expert_indices=..., # [total_tokens, selected_k] uint16 HEIGHT_SHARDED L1
    expert_scores=...,  # [total_tokens, selected_k] bfloat16 HEIGHT_SHARDED L1
    expert_mapping=..., # [total_devices, num_experts] uint16 DRAM (or L1)
    w0_w1_tensor=...,   # [num_cores, L, E, groups_per_core, K, 4*TILE] bfloat16 DRAM
    w2_tensor=...,      # [num_cores, L, E, 2, N, 4*TILE] bfloat16 DRAM
    cluster_axis=0,     # optional; None = single-device mode
)
```

### Outputs (5 tensors, index 3 and 4 alias the same buffer)
| Index | Shape | Format | Content |
|---|---|---|---|
| `[0]` | `[1, padded_E_elems]` uint32 L1 | Interleaved | Per-expert token counts |
| `[1]` | `[1, flat]` uint32 L1 | Interleaved | Expert activation metadata (see below) |
| `[2]` | `[1, E*total_tokens]` uint32 L1 | Interleaved, stride-1 flat | Token IDs per expert (zero-padded) |
| `[3]` | `[E*total_tokens, K]` bfloat16 L1 | BLOCK_SHARDED 4×3 combine grid | Expert output (W2 result) |
| `[4]` | Same buffer as `[3]` | Same | Alias for moe_compute compatibility |

#### Output[1] – Expert Activation Metadata
Each row is `align((2E+1)*4, 16)` bytes (= 48 bytes = 12 uint32 for E=4).
- `row[0]`: `token_id` (global, 0..total_tokens-1); `0xFFFFFFFF` = sentinel (no more rows)
- `row[1..E]`: k-index for each local expert (which position in the token's top-k selected it); `selected_k+1` = not activated for this token
- `row[E+1..2E]`: placeholder scores (not currently filled with useful data)
- Rows are written in NCRISC+BRISC order, consolidated by drain core

#### Output[2] – e_t Flat Token Indices
Stride-1 flat: `flat[e * total_tokens + k]` = k-th token routed to local expert `e` (zero-padded beyond count).

---

## 5. Key File Locations

### Kernel (C++)
```
ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/
├── moe_gpt.hpp                        # Op registration
├── moe_gpt.cpp                        # Python binding / invoke
├── device/
│   ├── moe_gpt_device_operation.hpp   # Op struct
│   ├── moe_gpt_device_operation.cpp   # validate / compute_output_specs / create_output_tensors
│   ├── moe_gpt_device_operation_types.hpp  # operation_attributes_t, tensor_args_t
│   ├── moe_gpt_program_factory.cpp    # Main program setup (CB creation, kernel launch, RT args)
│   ├── moe_gpt_program_factory.hpp
│   └── kernels/
│       ├── moe_gpt_ring_common.h      # Compile-time constants (tile counts, layout variants)
│       ├── tilize_reader.cpp          # NCRISC: reads indices/scores, builds metadata, does e_t repack
│       ├── tilize_writer.cpp          # BRISC: reads indices/scores in parallel with NCRISC
│       ├── tilize_compute.cpp         # TRISC on tilize cores (CB management for matmul pipeline)
│       ├── dm0.cpp                    # NCRISC on matmul cores: DRAM weight reads + A2A ring
│       ├── dm1.cpp                    # BRISC on matmul cores: untilize + write to combine buffer
│       ├── compute.cpp                # TRISC on matmul cores: W0/W1 matmul → SwiGLU → W2 matmul
│       ├── combine_dm1.cpp            # BRISC on combine cores: waits for matmul cores to finish
│       └── swiglu_sfpu.h              # SwiGLU implementation
```

### Model Integration (Python)
```
models/demos/gpt_oss/tt/experts_throughput/
├── config.py          # FusedMoeGptConfig, ThroughputExpertConfig
├── weights.py         # prepare_w0_w1_tensor, prepare_w2_tensor
├── fused_decode.py    # fused_decode_forward() — the full pipeline function
└── decode.py          # Dense (old) decode path (unchanged)
```

### Tests
```
tests/ttnn/nightly/unit_tests/operations/experimental/
├── test_moe_gpt_e2e.py              # Main E2E test: dispatch → moe_gpt (Galaxy 4×8)
├── test_moe_gpt_single_device.py    # Single-device reference and verify_device_output
├── test_moe_gpt.py                  # Basic multi-device test
├── test_moe_gpt_galaxy.py           # Galaxy-specific test
└── test_moe_gpt_fused.py            # Fused pipeline test
```

---

## 6. Expert Mapping Format

Both `all_to_all_dispatch_metadata` and `moe_gpt` use the **same** expert mapping tensor:

- **Shape**: `[total_devices, num_experts]` = `[32, 128]` for GPT-OSS-120b
  (The `[32, 16]` ring-local variant used in some tests is for simulation only)
- **Dtype**: uint16, DRAM
- **Formula**: `mapping[d, e] = e // experts_per_device` (same value for ALL `d`)
- **Meaning**: value at `[d, e]` is the **global linearized device ID** that owns expert `e`

The dispatch kernel silently skips experts where the target device is not in the same column. The moe_gpt kernel scans `mapping[my_device_id, :]` to find which global experts map to the current device.

**Routing indices** passed to dispatch and moe_gpt must be **ring-local** (0..15 for the 16-expert ring) on the test simulation, but **global** (0..127) in production with the full mapping.

---

## 7. CB Aliasing for HEIGHT_SHARDED Indices/Scores

This is a critical pattern shared with `moe_compute`. The dispatch output (`expert_indices`, `expert_scores`) is HEIGHT_SHARDED L1, all 128 tokens concentrated on a single drain core (CoreCoord 6,9 by default). moe_gpt reads these via CB aliasing:

**Drain core**: CB backed directly by the HEIGHT_SHARDED buffer:
```cpp
create_cb(indices_cb, program, drain_range, shard_bytes, 1, format, buffer);
// reader skips NOC read — data is already in the CB
```

**Non-drain cores**: CB allocated normally; reader does a single bulk NOC read from the drain core's buffer address into the local CB.

If CB aliasing is broken (e.g., old ShardSpec API that produces `buffer_distribution_spec=null`), the InterleavedAddrGen is used instead and reads from 128 wrong core addresses → PCC≈0. This was the root cause of a major correctness bug fixed early in bring-up.

---

## 8. What Has Been Verified

### Tests Passing (as of 2026-03-10)

| Test | Status | Notes |
|---|---|---|
| `test_dispatch` | **PASSING** | Dispatch metadata only |
| `test_dispatch_compute` | **PASSING** | Dispatch + moe_gpt compute (PCC ~0.990) |
| `test_dispatch_compute_combine` | **PASSING** | Dispatch + moe_gpt + selective_reduce_combine (PCC ~0.987) |
| `test_moe_gpt_e2e` | **PASSING** | Full pipeline with accuracy verification on all 32 devices |
| `test_moe_gpt_e2e_perf` | **PASSING** | Full pipeline with Tracy-compatible perf measurement |
| Output[0] – per-expert token counts | **PASSING** | Verified against torch routing |
| Output[2] – e_t flat token indices | **PASSING** | Stride-1 format, zero-padded |
| Output[4] – combine output (matmul accuracy) | **PASSING** | PCC > threshold vs torch W0/W1/SwiGLU/W2 reference |
| `selective_reduce_combine` after moe_gpt | **PASSING** | With correct batch_size=128 (total ring tokens) |
| Output[1] – activation metadata | **PASSING** | Verified on all 32 devices |

### Verification Approach

The E2E test (`test_moe_gpt_e2e.py`) runs:
1. On a (4,8) mesh – 8 independent column rings run in parallel
2. Randomly generates routing: 128 tokens × top-4 from 16 ring-local experts
3. Runs `all_to_all_dispatch_metadata` and verifies sparse buffer correctness
4. Runs `moe_gpt` and verifies:
   - **Matmul output**: PCC vs torch reference (W0/W1 → SwiGLU → W2) per device
   - **Routing metadata**: output[0], output[1], output[2] against python routing oracle
5. Compares only the `last_chunk_tokens` valid rows (combine shard has uninitialized L1 beyond valid rows)

---

## 9. Bugs Fixed During Bring-Up

### (a) DispatchAlgorithm: SPARSE_UNICAST not SPARSE_MCAST
- For `cluster_axis=0`, dispatch must use `SPARSE_UNICAST`; MCAST hangs

### (b) CB aliasing for HEIGHT_SHARDED dispatch outputs (critical)
- Old ShardSpec API produced `buffer_distribution_spec=null` → InterleavedAddrGen → reads wrong addresses → PCC≈0
- Fix: Explicit `create_cb(..., buffer)` on drain core, `create_cb(...)` without buffer on non-drain; non-drain reader does single bulk NOC read from drain core

### (c) `selective_reduce_combine` batch_size bug
- Must be `total_ring_tokens=128` (not `M=32`); using M caused source_device indices to go out of range → kernel hang

### (d) BRISC activation buffer sentinel bug (`tilize_writer.cpp`)
- **`row_ptr[0] = 0`** initialized token_id to 0 (a valid token!) instead of `0xFFFFFFFF`
- Unwritten BRISC slots appeared as valid rows with token_id=0
- **Fixed**: `row_ptr[0] = static_cast<uint32_t>(-1)` (2026-03-10)

### (e) e_t output page size mismatch in program factory
- Old formula: `(tokens+1) * align(4, 16) = 129*16 = 2064 bytes`
- New e_t spec shape `{1, E*total_tokens}` has page_size = `4*128*4 = 2048 bytes`
- 16-byte overflow → L1 corruption
- **Fixed**: `tilize_e_t_output_page_size = experts_per_device * tokens * sizeof(uint32_t)` (2026-03-10)

### (f) BRISC "not activated" sentinel used wrong value
- `tilize_writer.cpp` used `selected_experts_k` (=4) as sentinel; NCRISC uses `selected_experts_k+1` (=5)
- Fixed earlier: `selected_experts_k + 1`

### (g) Zero-initialized non-dispatched token slots masked as valid routing
- Non-dispatched tokens have all-zero indices; device 0's expert 0 maps to index 0 → falsely matched
- Fixed: skip tokens where all K indices are identical (valid routing always has K distinct experts)

### (h) Test output[2] stride bug
- e_t entries are 64 bytes apart internally; test was reading stride-1 and picking up padding zeros
- Fixed: read with `e_t_stride = l1_align_bytes // 4 = 16`

### (i) Galaxy `num_links` conventions
- `cluster_axis=0` (column, 4 devices): `num_links=4`
- `cluster_axis=1` (row, 8 devices): `num_links=1`
- Using `num_links=4` for `cluster_axis=1` hangs (requesting more links than exist)

### (j) Expert mapping formula
- Initial confusion between ring-local (0..15) and global (0..127) expert IDs
- moe_gpt kernel works with global mapping `[32, 128]` directly; ring-local `[32, 16]` only works in the 4-device simulation test
- Production code: use global `[32, 128]` mapping

### (k) `selective_reduce_combine` worker core ordering mismatch (2026-03-10)
- `corerange_to_cores()` defaults to `row_wise=False` (column-major: x-outer, y-inner)
- BLOCK_SHARDED with `ShardOrientation::ROW_MAJOR` assigns shards in row-major order (y-outer, x-inner)
- Each data-parallel worker read from the wrong L1 shard, producing PCC ~0.02
- **Fixed**: pass `row_wise=True` to `corerange_to_cores()` so worker cores match shard assignment

### (l) `test_moe_gpt_e2e` experts_total semantics (2026-03-10)
- `experts_total` param was 16 (per-ring) but should be 128 (global) for 4x8 Galaxy
- `create_expert_mapping_tensors` called with wrong keyword args
- Output reshape used `[:E*M, :]` slice (only first height shard) instead of full tensor
- **Fixed**: corrected all three issues

---

## 10. What Still Needs to Be Done for Model Integration

### 10.1 ~~Re-verify output[1]~~ DONE
All 5 tests pass on all 32 devices (2026-03-10). Output[1] verified.

### 10.2 Remove debug logging from `fused_decode.py`
`fused_decode.py` currently has extensive `logger.info` and tensor read-back calls (lines ~128–218) for debugging. These do device→host transfers every step and will significantly impact performance. They must be removed or gated behind a debug flag before production use.

### 10.3 Verify `batch_size` parameter in `selective_reduce_combine` call
In `fused_decode.py` line 244 the `batch_size` argument is currently `tokens_per_device` (=32). Per the analysis in §8 of the E2E test: the E2E test passes with `batch_size=total_tokens=128`. Confirm which is correct for the production selective_reduce_combine kernel, and update accordingly.

### 10.4 Wire `fused_decode_forward` into the model
`fused_decode_forward` in `models/demos/gpt_oss/tt/experts_throughput/fused_decode.py` is implemented but not yet connected to the main `ThroughputExperts` class in `decode.py`. The intended integration point is:

```python
# In ThroughputExperts.forward() or similar:
if self.fused_config is not None:
    return fused_decode_forward(hidden_states, indices, scores, self.config, self.fused_config, mesh_device)
else:
    return decode_forward(...)  # existing dense path
```

### 10.5 Pre-allocation / FusedMoeGptConfig setup in model init
`FusedMoeGptConfig` requires pre-allocated dispatch output tensors, semaphores, and combine preallocated tensor. These need to be created in the model's `__init__` / weight loading step alongside `tt_w0_w1` and `tt_w2`. Follow the pattern in `test_moe_gpt_e2e.py` for creating:
- `dispatch_sparse` (DRAM, `[total_tokens, K]`)
- `dispatch_indices` / `dispatch_scores` (HEIGHT_SHARDED L1, `[total_tokens, selected_k]`)
- `dispatch_semaphore` (global semaphore)
- `combine_preallocated` (DRAM, `[experts_per_ring, M, K]`)
- `combine_semaphore` (global semaphore)
- `combine_worker_cores`, `combine_mux_cores` — copied from the combine core selection logic in `moe_gpt_device_operation.cpp`

### 10.6 Routing weight map for accurate output
`fused_decode_forward` accepts an optional `routing_weight_map` `[1, E_ring, M, 1]` for scaling expert outputs by routing scores before summing. Without it, the output is an unweighted sum and PCC vs reference will be degraded. This map must be built from the router output each step:
```python
weight_map[e, m] = scores[m, k] where indices[m, k] == local_expert_e
```

### 10.7 End-to-end model accuracy test
Once wired in, run a text generation test with `gpt-oss-20b` on Galaxy and compare:
- Perplexity / output quality vs reference (dense path)
- Throughput (tokens/s) — the fused path should be significantly faster at batch ≥ 32

---

## 11. Performance Reference (from Tracy profiling, 2026-03-10)

On a (4,8) Galaxy mesh, single decode step (`python3 -m tracy -m -r -p pytest ...`):
| Stage | Kernel Avg (μs) | Kernel Max (μs) | FW Avg (μs) | FW Max (μs) |
|---|---|---|---|---|
| `all_to_all_dispatch_metadata` | 25.4 | 58.1 | 26.5 | 59.2 |
| `moe_gpt` | 259.5 | 262.3 | 263.4 | 266.2 |
| `selective_reduce_combine` | 37.0 | 79.4 | 38.0 | 80.4 |
| `CopyDeviceOperation` (DRAM→L1) | 3.0 | 3.4 | 4.1 | 4.4 |
| **Total** | **~325 μs** | | **~332 μs** | |

Note: moe_gpt dominates at ~260 μs (80% of total). The dispatch and combine steps
are fast (~25-37 μs each). Max times are higher due to fabric synchronization variance.

---

## 12. Key Gotchas for a New Developer

1. **CB aliasing is mandatory** for HEIGHT_SHARDED inputs. Never change the `create_cb` calls for indices/scores without understanding the aliasing pattern (see §7).

2. **num_links is axis-dependent**: always `4` for `cluster_axis=0`, always `1` for `cluster_axis=1` on Galaxy.

3. **Expert mapping is always global** `[32, 128]` in production; the ring-local `[32, 16]` variant only makes sense in the 4-device simulation test.

4. **output[3] and output[4] alias the same L1 buffer**. Do not deallocate both; deallocate only `output[3]` (or just output[0..3], since 4 is an alias).

5. **Combine shard has uninitialized L1** beyond the valid token rows. Always slice to `[:E*M, :]` when comparing, never compare the full shard.

6. **The tilize phase is split** between NCRISC (first half of tokens) and BRISC (second half) running in parallel on the same tilize core. They merge results on the drain core before matmul starts.

7. **moe_gpt weight tensors** have a non-standard shape:
   `w0_w1: [num_cores, L, E, groups_per_core, K, 4*TILE_SIZE]`
   `w2: [num_cores, L, E, 2, N, 4*TILE_SIZE]`
   These are prepared by `weights.py:_prepare_w0_w1_tensor` / `_prepare_w2_tensor`, not standard ttnn weight tensors.
