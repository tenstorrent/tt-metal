# High-bandwidth all-gather

`ttnn.experimental.deepseek_prefill.high_bw_all_gather` is a dedicated
large-message collective for the sparse-MLA prefill path. It is independent of
`ttnn.all_gather`: there is no composite fallback and no environment-variable
selection.

## Contract

- row-major or tile-layout DRAM input;
- preallocated, persistent, interleaved-DRAM output;
- required `cluster_axis` (0 for N/S or 1 for E/W) selecting one mesh axis
  with at least two devices and at least one Fabric link;
- a one-dimensional all-gather only: devices on the orthogonal mesh axis run
  independent collectives, and gathering across both axes is unsupported;
- Fabric1D line/ring, or Fabric2D on a direct physical line/ring;
- Fabric2D Torus can wrap `cluster_axis` when every ring edge is a direct
  physical neighbor, including a size-two physical torus dimension;
- semaphores use L1-small when the device reserves it, otherwise L1 with a
  warning.

Fabric2D eligibility is intentionally strict. The host resolves each directed
line or ring neighbor, verifies every requested link index, and hashes the
complete physical route plan into the program-cache key. A topology that would
require Fabric forwarding is rejected rather than silently taking a slower or
unsafe path. Fabric handles a size-two torus dimension specially: because the
ordinary and wrap neighbors collapse onto the same physical connection, bubble
flow control and first-level ACK are disabled only for that dimension. Genuine
torus dimensions retain deadlock avoidance.

`cluster_axis` is the device-mesh dimension, while `dim` is the tensor
dimension concatenated by the collective. On an `(R, C)` mesh, axis 0 produces
independent `R`-device gathers in each column; axis 1 produces independent
`C`-device gathers in each row. The output extent at `dim` must be multiplied
by the selected axis size, not by `R * C`.

## Data flow

```text
local DRAM pages
      |
      | bank-owned readers (2 links x 8 workers/direction on Blackhole)
      v
  Tensix CBs -- full-packet coalescing --> Fabric mux --> adjacent ERISC
                                                        |
                                                        | terminal one-hop packet
                                                        v
                                                neighbor output DRAM
                                                        |
                                                        | next relay iteration
                                                        v
                                                neighbor Tensix readers
```

Forward and backward halves run concurrently. A shard reaches a distant
logical rank through store-and-forward iterations: each transfer is terminal
at the adjacent physical device, and the next device rereads the received rows
from its persistent output before sending the next hop. This avoids multi-hop
Fabric2D packets, corner forwarding, dateline handling, and router-side
terminal delivery changes.

Small sparse-MLA rows and tiles are assigned by destination DRAM bank. Each
writer can therefore combine consecutive bank-local pages into large Fabric
packets rather than issuing small scatter packets. One L1-small `data_valid`
semaphore per worker grid gates relay reads and records completion; persistent
output removes the separate allocation-readiness handshake.

## Minimal Fabric requirements

### One-hop validation

The op proves that each logical line/ring edge is one physical hop by combining the
existing `pipeline_get_forwarding_direction` and `pipeline_get_chip_neighbors`
control-plane queries. This is a host-side safety check with no firmware or
performance effect and requires no Fabric change.

The operation follows the configured topology rather than opportunistically
upgrading a plain `FABRIC_2D` mesh to a ring. `FABRIC_2D_TORUS_X` wraps axis 1,
`FABRIC_2D_TORUS_Y` wraps axis 0, and `FABRIC_2D_TORUS_XY` can wrap either axis.
If the configured Fabric does not wrap `cluster_axis`, the operation uses its
direct-line schedule.

### Worker-injection spare slots

The Fabric2D static allocator selects a uniform packet-slot depth per virtual
channel. On this topology that discrete choice consumes 20 of 25 available
slots and leaves five whole slots unused. The change assigns only those five
stranded slots to the live VC0 local-worker injection channel:

- local-worker sender depth: 2 -> 7;
- forwarding and receive depths: unchanged;
- total allocation: still exactly the existing L1 capacity;
- Fabric1D Ring allocation: unchanged and covered by a host unit test.

This cannot be recovered by more op-level packet coalescing. The op already
emits large packets through the mux; the two-slot sender credit window is owned
by the ERISC Fabric allocator and limits how many packets the workers can have
in flight. An op-only workaround would require smaller packets or topology- and
format-specific throttling, reducing link efficiency without exposing the
unused buffer capacity.

No other Fabric changes are required. In particular, this implementation does
not need router firmware changes, corner forwarding, dateline escape VCs,
terminal offload, a channel-trimming profile, or runtime tuning variables.

### Channel-trimming compatibility

Channel trimming is configured when Fabric is initialized and applies to the
entire workload, not to an individual CCL. A captured profile must therefore
cover every CCL and shape that the process will run.

Until the Fabric trim-derived speedy-VC0 eligibility distinguishes this
high-rate, multi-worker all-gather from a simple source/terminal flow, use the
VC0 override below with any channel-trimming profile for a workload that
contains this op. The op remains correct without the override, but can suffer a
substantial performance regression.

```bash
TT_METAL_FABRIC_TRIMMING_PROFILE=/path/to/channel_trimming_capture.yaml \
TT_METAL_FABRIC_TRIMMING_OVERRIDE=/path/to/enable_vc0_all_channels.yaml \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/experimental/deepseek_prefill/test_high_bw_all_gather.py
```

Create `enable_vc0_all_channels.yaml` with:

```yaml
channel_trimming_overrides:
  vc0:
    force_enable_all_sender_channels: true
    force_enable_all_receiver_channels: true
```

The override force-enables all VC0 sender and receiver channels for the Fabric
instance. It is a workload-level compatibility setting: it prevents the
problematic trim-derived VC0 fast path, but also gives up VC0 channel-service
trimming for every op in that Fabric instance. Use a separate process/Fabric
instance if another workload needs a different trimming policy.

## Measured contribution

Measurements use firmware `19.10.99`, an 8x1 mesh, two links, a 14,336-byte
Fabric payload, 65,536 rows per device (512K rows globally), persistent output,
and seven cached device-profiler samples. Effective receive bandwidth is
`local_bytes * (axis_size - 1) / device_program_time`.

| Format | Fabric | Schedule | Median | Effective receive BW |
| --- | --- | ---: | ---: | ---: |
| BF16, 1,152-B row | Fabric1D | line | 10.901 ms | 48.480 GB/s |
| BF16, 1,152-B row | Fabric1D Ring | ring | 5.565 ms | 94.969 GB/s |
| BF16, 1,152-B row | Fabric2D | direct physical line | 10.973 ms | 48.162 GB/s |

The Fabric2D line is therefore expected to provide line bandwidth, not the
two-direction ring bandwidth. Active-axis Fabric2D Torus uses the ring schedule,
including on the size-two Y dimension of the 2x4 LoudBox.

## Qualification

Run the host allocator invariants:

```bash
build_Release/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter='FabricStaticSizedChannelsAllocatorTest.*'
```

Run exact correctness and performance for all supported Fabric configurations,
both layouts, and all four format/layout combinations:

```bash
TT_METAL_DISABLE_PRECOMPILED_FW=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/experimental/deepseek_prefill/test_high_bw_all_gather.py \
  -x -s -v
```

Do not add `--dev`: watcher and LLK assertions increase firmware code size for
this test. The test requires the standalone unicast kernel in profiler records,
compares every gathered replica exactly (including opaque FP8 payload bytes and
decoded BFP8_B tiles), and enforces a 90 GB/s regression floor for Fabric1D
Ring and active-axis Fabric2D Torus, and a 45 GB/s floor for line schedules.
