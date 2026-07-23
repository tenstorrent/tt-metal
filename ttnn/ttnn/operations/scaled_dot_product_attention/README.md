# Fused tiled flash attention

`ttnn.flash_attention(Q, K, V)` computes non-causal
`softmax(scale * Q @ Kᵀ) @ V` in one multi-core device program. Q, K, V, and
the result are BF16 tile tensors in interleaved DRAM with shape `[B,H,S,D]`.
Sequence and head dimensions must be multiples of 32.

The program never writes the score matrix to DRAM. Each core retains scaled Q
and BF16 online-softmax `(m, l, O)` state in L1 while it streams K/V blocks.
Alpha is BF16 and the final reciprocal intermediate is FP32. K arrives in the
tile-grid order needed by transposed matmul, so there is no separate transpose
pass.

For the primary 8+ head, 4096+ token prefill regime, the schedule forms one
horizontal core group per flattened batch/head when the grid permits. One core
per group reads K/V from DRAM and multicasts it to the query-parallel peers. The
sender column rotates by head by default to distribute DRAM traffic. K/V and
output streams are double buffered and reads and writes use separate NoCs.

The output recurrence is fused into PV. The compute kernel first packs
`alpha * O_old` into a caller-owned O-state region, then PV matmul starts packer
L1 accumulation on K-block zero and adds `P @ V` directly to that region. This
removes the former temporary O block and separate output-add phase. The generic
`kernel_lib::matmul_block` helper now exposes this preinitialized-target mode and
restores the packer's L1-accumulation state before returning.

For update blocks whose query rows and old maxima fit together in DEST, block
maximum and online rescaling share one register window. Q4 with BF16 DEST keeps
the four K16 row maxima in D0--D3 and loads the four old maxima into D4--D7,
then directly produces `m_new` and `alpha`. This removes the intermediate
`BLOCK_M` pack/unpack round trip. Other geometries retain the general fallback.

## Tuning

`FlashAttentionProgramConfig` exposes:

- `query_block_tiles`, `key_block_tiles`: online-softmax geometry and L1 use;
  defaults resolve up to 4 query tiles and 16 key/value tiles;
- `qk_output_subblock`, `pv_output_subblock`: independent matmul subblocks;
  automatic selection favors tall QK reuse and wide PV reuse within DEST;
- `softmax_block_tiles`: elementwise helper batch size;
- `num_cores`, `q_parallel_group_size`, `use_kv_multicast`, and
  `spread_kv_readers`: parallelism and K/V-sharing topology;
- `kv_buffer_depth`, `output_buffer_depth`, `read_barrier_tiles`, and
  `write_barrier_tiles`: stream depth and NoC issue batching;
- `reader_noc` and `writer_noc`: independent NoC placement;
- `math_fidelity` and `fp32_dest_acc_en`: matmul fidelity and DEST format;
- `exp_approx_mode`: probability exponent (`fast`, `accurate_fast`, `exact`);
- `rescale_exp_approx_mode`: separate online-rescale exponent. With BF16 DEST,
  `None` selects `accurate_fast`;
- `profile_phase`: an opt-in named device zone for per-phase profiling.

The tuned default is HiFi2 with BF16 DEST accumulation. LoFi was not used for
the reported results. BF16 DEST provides an eight-tile output-subblock budget;
the default `[1,8,4096,128]` geometry resolves to Q4/K16, QK 4x2, and PV 2x4.

`fast` uses the rough approximate exponent. `accurate_fast` selects the
hand-tuned `exp_21f` path, omits its lower input clamp, and uses packer ReLU to
repair extreme-negative results. `exact` retains clamping and selects the
architecture's precise path. The default spends `accurate_fast` only on the
small online-rescale vector and uses `fast` for the much larger probability
block.

## Measured 8-head/4K result

On the available Wormhole B0 device, ten trials after three warmups produced:

| Shape | Fidelity / DEST | Device time | Effective QK+PV | Sampled PCC |
|---|---|---:|---:|---:|
| `[1,8,4096,128]` | HiFi2 / BF16 | 1.401 ms | 49.06 TFLOP/s | 0.999840486763 |

PCC samples six query rows from every head while referencing all 4096 K/V rows.
Effective throughput is `4*B*H*Sq*Sk*D / device_time`, counting the QK and PV
matmuls. It is an operation-level throughput number, not hardware math-unit
utilization.

The kernel optimization progression on the same device and shape was:

| Kernel checkpoint | Device time | Effective QK+PV | Sampled PCC |
|---|---:|---:|---:|
| Initial implementation | 2.362 ms | 29.09 TFLOP/s | 0.999846232042 |
| Q4/K16 plus fused softmax state work | 1.637 ms | 41.97 TFLOP/s | 0.999834272123 |
| Fused O-state seed and PV accumulation | 1.515 ms | 45.35 TFLOP/s | 0.999834193009 |
| Direct K16 max, two-tile softmax, BF16 alpha | 1.418 ms | 48.48 TFLOP/s | 0.999840486763 |
| Fused block max and online rescale | 1.401 ms | 49.06 TFLOP/s | 0.999840486763 |

The final kernel is 40.7% lower latency and 68.6% higher effective throughput
than the initial implementation, with essentially unchanged PCC.

## Device-zone findings

`DeviceZoneScopedN` profiling of the current default gave representative
per-core medians:

| Compute phase | Median wall/core | Limiting engine |
|---|---:|---|
| Q scale | 25.5 us | math |
| Block max portion of fused update | 42.7 us | unpack |
| Online-rescale portion of fused update | 198.6 us | pack |
| QK matmul | 349.8 us | math |
| Probability subtract/exp/pack | 351.1 us | unpack |
| Block sum and denominator recurrence | 163.8 us | unpack |
| PV matmul into preseeded O state | 342.5 us | pack |
| O-state seed (`alpha * O_old`) | 58.9 us | pack |
| Final normalize | 37.6 us | unpack |

Because maximum and rescaling now share a DEST window, some waits move between
their individual scopes; their combined time is the useful comparison. Summed
compute-zone medians were 1.348 ms unpack, 1.391 ms math, and 1.390 ms pack. QK
plus PV occupy 692.3 us, or 49.8% of the 1.391 ms compute critical path. K/V
receive measured 1.310 ms and the output DRAM write only 42.8 us, so the kernel
remains narrowly compute-pipeline bound rather than output- or DRAM-bound.

A block sweep confirmed Q4/K16 at 1.516 ms as the best tested configuration.
Q4/K8 measured 1.800 ms, Q2/K8 1.995 ms, and Q8/K8 with single-buffered output
measured 1.747 ms. Double-buffered Q8/K16 exceeds usable L1; a fitting
single-buffered variant measured 1.791 ms. Larger blocks therefore did not beat
the tuned default.

## Tests

Run correctness and opt-in measurements through the safe wrapper:

```sh
./scripts/run_safe_pytest.sh \
    tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_flash_attention.py -q
TTNN_RUN_LONG_FLASH_ATTN=1 ./scripts/run_safe_pytest.sh \
    tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_flash_attention.py::test_flash_attention_prefill_8h_4k -q
TTNN_RUN_FLASH_ATTN_PRECISION=1 ./scripts/run_safe_pytest.sh \
    tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_flash_attention.py::test_flash_attention_prefill_8h_4k_precision_sweep -s
TTNN_RUN_FLASH_ATTN_DEFAULT_PERF=1 ./scripts/run_safe_pytest.sh \
    tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_flash_attention_perf.py::test_flash_attention_prefill_default_device_perf -s
TTNN_RUN_FLASH_ATTN_PHASE_PROFILE=1 ./scripts/run_safe_pytest.sh \
    tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_flash_attention_perf.py::test_flash_attention_prefill_phase_profile -s
TTNN_RUN_FLASH_ATTN_SWEEP=1 ./scripts/run_safe_pytest.sh \
    tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_flash_attention_perf.py::test_flash_attention_prefill_block_sweep -s
```
