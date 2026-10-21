# High-bandwidth all-gather

`ttnn.experimental.high_bw_all_gather` is a dedicated
large-message collective for the sparse-MLA prefill path. It is independent of
`ttnn.all_gather`: there is no composite fallback and no environment-variable
selection.

The program is `device/high_bw_all_gather_fabric_factory.cpp`, with its kernels in `device/kernels/fabric/`.

## Program
Terms (logical vs physical pages, outer slices, fabric chunks, CB batches, outgoing shards, the counters):
`device/kernels/fabric/fabric_all_gather_chunk_walk.hpp`.

Per chip, per ring, per ring direction and per link, one **fabric link worker** core: a reader (NCRISC, NoC0) reads
this chip's own shard from the input, then the shards it forwards from its own output; a sender (BRISC, NoC1) sends
each fabric chunk one fabric hop into the same pages of the neighbour's output. **Local copy cores** write the chip's
own shard: one per link, or two per link for a non-interleaved input (the ND-sharded KV cache), which they convert
block by block into the output for the link workers to read from there. Link workers sit next to the Ethernet core of
their link (`tt_fabric::get_forwarding_eth_core`).

- **Fabric chunks**: up to `payload / page` consecutive pages of one page lane (pages k, k + B, k + 2B, ... for B
  DRAM banks), physically contiguous in an interleaved tensor, one packet each (a page larger than the payload is split
  over several packets). Chunks are numbered row block by row block and dealt round robin to the link workers (and to
  the local copy cores), so the cores stay balanced for any bank count: 8 on Blackhole, 7 with a harvested bank.
- **Forwarding**: outgoing shard k is what upstream sent as its outgoing shard k - 1. Upstream's last packet of every
  outgoing shard (or a bare increment, if none of the shard's chunks is that worker's) increments this worker's
  shards-arrived counter, so the reader forwards shard k once the counter reaches k.
- **Fence**: before sending, each sender waits until its downstream has signalled that it started this call (the
  downstream-started counter), so a reused output is never overwritten early.
- **Rings**: axis line / ring; for `cluster_axis=None` a snake, or on a torus with both sides >= 3 two edge-disjoint
  Hamiltonian cycles (every chip uses all four neighbours). Even rings are balanced: the opposite shard goes half each
  way.

## Contract

- row-major or tile-layout DRAM input. Pages move whole, so a row-major input
  cannot be gathered along its innermost dim (every output row would mix ranks);
- preallocated, persistent, interleaved-DRAM output;
- required `cluster_axis`: 0 selects N/S, 1 selects E/W, and explicit `None`
  linearizes the complete 2D mesh into one snake ring;
- tensor distribution axes aligned with the physical device-mesh axes;
- optional `num_links`; by default the op uses the number of links discovered
  usable across the complete selected axis, or the minimum across both axes in
  full-mesh mode. An override must be greater than zero and no larger than
  every participating axis supports;
- optional `subdevice_id` and `sub_core_grids` worker restriction. An explicit
  grid must be fully contained in the selected TENSIX subdevice; an accidental
  partial intersection is rejected;
- optional caller-owned `ready_semaphore` and `data_valid_semaphore`. Supply
  both or neither. The external pair enables the allocation-free,
  no-internal-synchronization path required for concurrent subdevice dispatch;
- a one-dimensional all-gather schedule: an integer `cluster_axis` runs
  independent collectives on the orthogonal axis, while `cluster_axis=None`
  makes every mesh device participate in one logical ring;
- full-mesh mode requires Fabric2D and a 2D mesh with at least one even
  dimension, row-major tensor mesh coordinates, and the gather dimension
  sharded across every device. Every edge in the selected row or column snake,
  including its closing edge, must have the requested direct physical links;
- Fabric1D line/ring, or Fabric2D on a direct physical line/ring;
- Fabric2D Torus can wrap `cluster_axis` when the axis has a distinct wrap
  neighbor and every ring edge is a direct physical hop; a size-two torus axis
  uses the line schedule because its ordinary and wrap neighbors coincide;
- semaphores use L1-small when the device reserves it, otherwise L1 with a
  warning.
- worker cores must be exclusive to this program for its duration: one fabric
  link worker per ring, direction and link, plus one local copy core per link
  (two for a non-interleaved input). With two links that is 6 cores for an axis
  gather, 8 for the ND-sharded KV cache and 10 for the full-mesh torus path. Use
  `sub_core_grids` to keep unrelated L1-resident tensors and concurrent programs
  off those cores.

Fabric2D eligibility is intentionally strict. The host resolves each directed
line or ring neighbor, verifies every requested link index, and hashes the
complete physical route plan into the program-cache key. A topology that would
require Fabric forwarding is rejected rather than silently taking a slower or
unsafe path. Fabric handles a size-two torus dimension specially: because the
ordinary and wrap neighbors collapse onto the same physical connection, the
axis retains mesh directionality and the operation uses its line schedule.
Genuine torus dimensions retain wrap routing and deadlock avoidance.

`cluster_axis` is the device-mesh dimension, while `dim` is the tensor
dimension concatenated by the collective. On an `(R, C)` mesh, axis 0 produces
independent `R`-device gathers in each column; axis 1 produces independent
`C`-device gathers in each row. The output extent at `dim` must be multiplied
by the selected axis size, not by `R * C`.

With `cluster_axis=None`, the host selects a direct Hamiltonian cycle over the
complete `(R, C)` mesh. It prefers a row snake when `R` is even:
`(0,0)..(0,C-1),(1,C-1)..(1,0),...`. If that route is not direct, it tries a
column snake when `C` is even:
`(0,0)..(R-1,0),(R-1,1)..(0,1),...`. A row snake closes along Y; a column
snake closes along X. A lane count of two closes through ordinary adjacency,
while a larger lane count requires the corresponding direct torus wrap. The
host route proof selects the first orientation for which every edge and link
is direct. Fabric uses the selected snake index for communication, while the
kernels translate each stripe back to the tensor's normal row-major rank
before reading or writing the persistent output. In this mode the output
extent at `dim` is multiplied by `R * C` and every shard placement of `dim`
becomes replicated.

Link discovery chooses a uniform count supported by every participating
device and direction. Full-mesh mode conservatively uses the minimum discovered
across both complete mesh axes, including configured wrap directions. Set
`num_links=2` when results should be comparable across QuietBox (four available
links), LoudBox, and Galaxy (two available links). This is a cap on discovered
hardware capacity: requesting more links than Fabric reports fails before
program construction.

The snake schedule uses one forward and one backward port per device. For an
8x4 mesh its idealized link-volume bound matches a two-phase axis-1 then axis-0
all-gather: the advantage is a single program, one synchronization sequence,
and a directly produced 32-way row-major layout, not an inherent claim of
higher fabric bandwidth. On a torus with both sides >= 3 the snake is replaced by
the two Hamiltonian cycles described above, which use all four links per chip.

## External semaphore interface

The overlap-capable call form is:

```python
ttnn.experimental.high_bw_all_gather(
    input_tensor,
    dim,
    output_tensor,
    cluster_axis=cluster_axis,
    subdevice_id=gather_subdevice_id,
    sub_core_grids=gather_core_grid,
    num_links=2,
    ready_semaphore=ready_semaphore,
    data_valid_semaphore=data_valid_semaphore,
)
```

The two semaphores are persistent runtime resources, not program structure.
They must be distinct, belong to the input tensor's mesh device, use the same
buffer type the op would select itself (`L1_SMALL` when configured,
otherwise `L1`), and cover every selected reader/writer core. Create them at
zero over the complete gather strip, then perform one mesh-wide readiness
barrier across every participating device before the first dispatch. The op
does not allocate semaphores or call `Synchronize` when the pair is supplied.

Successful kernels consume their ready and data-valid counts back to zero, so
the same pair can be reused for adjacent invocations. Program-cache hits may
also bind a different valid pair: semaphore addresses are excluded from the
program hash and patched into runtime arguments on every call. If an invocation
is aborted or fails after dispatch, the owner must join the affected subdevice,
drain any in-flight work, and reset both semaphore values to zero before reuse.

Omitting both handles preserves the existing behavior: the op allocates its
own semaphores on a cache miss and performs one readiness `Synchronize`.

## Data flow

```text
own shard (input DRAM)                forwarded shards (this chip's output DRAM)
      |                                          |
      +--------------------+---------------------+
                           |
                           | link worker reader (NCRISC, NoC0): one fabric chunk =
                           | consecutive pages of one DRAM bank, one DRAM read
                           v
                 chunk CB (two CB batches)
                           |
                           | link worker sender (BRISC, NoC1): one chunk per packet,
                           | direct fabric connection (no mux)
                           v
                     adjacent ERISC
                           |
                           | terminal one-hop packet; a shard's last packet
                           | increments the neighbour's shards-arrived counter
                           v
                  neighbour output DRAM
                           |
                           | next hop: forwarded once the whole shard has arrived
                           v
                neighbour link worker reader

own shard (input DRAM) --> local copy cores --> this chip's output DRAM
```

Forward and backward directions run concurrently, and on a torus with both sides >= 3 so do the two Hamiltonian
cycles. A shard reaches a distant logical rank through store-and-forward hops: each transfer is terminal at the
adjacent physical device, and the next device rereads the received pages from its persistent output before sending
the next hop. This avoids multi-hop Fabric2D packets, corner forwarding, dateline handling, and router-side terminal
delivery changes.

Pages are dealt to the link workers as fabric chunks: up to one packet of consecutive pages of one DRAM page lane
(pages k, k + B, k + 2B, ... for B banks), numbered by row block and lane and dealt round robin, so the workers stay
balanced for any bank count. Each chunk is one DRAM read and one full Fabric packet rather than many small scatter
packets. Two L1-small counters per link worker order the protocol: the downstream-started counter fences each call
(no chip writes into a neighbour's output before the neighbour has started the same call), and the shards-arrived
counter gates forwarding and records completion. Both return to zero at the end of every call, and the persistent
output removes any separate allocation-readiness handshake.

## Minimal Fabric requirements

### One-hop validation

The op proves that each logical line/ring edge is one physical hop by combining the
existing `pipeline_get_forwarding_direction` and `pipeline_get_chip_neighbors`
control-plane queries. This is a host-side safety check with no firmware or
performance effect and requires no Fabric change.

For an integer `cluster_axis`, the operation follows the configured topology
rather than opportunistically upgrading a plain `FABRIC_2D` mesh to a ring.
`FABRIC_2D_TORUS_X` wraps axis 1, `FABRIC_2D_TORUS_Y` wraps axis 0, and
`FABRIC_2D_TORUS_XY` can wrap either axis. If the configured Fabric does not
wrap the selected axis, the operation uses its direct-line schedule. Full-mesh
mode always requests a ring and is admitted only when the complete host route
proof shows that every snake edge is one physical hop.

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
emits full-payload packets; the two-slot sender credit window is owned
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
  tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py
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

Measurements use a Blackhole LoudBox (8x P150b, firmware bundle `19.12.0` with 200G Ethernet links and a 12x10
worker grid), opened as an 8x1 mesh, two links, a 14,336-byte Fabric payload, 65,536 rows per device (512K rows
globally), persistent output, and seven cached device-profiler samples. Effective receive bandwidth is
`local_bytes * (axis_size - 1) / device_program_time`.

| Format | Fabric | Schedule | Median | Effective receive BW |
| --- | --- | ---: | ---: | ---: |
| BF16, 1,152-B row | Fabric1D | line | 10.941 ms | 48.304 GB/s |
| BF16, 1,152-B row | Fabric1D Ring | ring | 5.533 ms | 95.519 GB/s |
| BF16, 1,152-B row | Fabric2D | direct physical line | 11.005 ms | 48.024 GB/s |

The Fabric2D line is therefore expected to provide line bandwidth, not the
two-direction ring bandwidth. Fabric2D Torus uses the ring schedule only when
the active axis has at least three devices and a direct physical wrap edge; a
size-two dimension such as the Y axis of the 2x4 LoudBox remains a line.

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
  tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py \
  -x -s -v
```

Do not add `--dev`: watcher and LLK assertions increase firmware code size for
this test. The test requires the fabric sender kernel in profiler records,
compares every gathered replica exactly (including opaque FP8 payload bytes and
decoded BFP8_B tiles), and enforces a 90 GB/s regression floor for Fabric1D
Ring and active-axis Fabric2D Torus, and a 45 GB/s floor for line schedules.
