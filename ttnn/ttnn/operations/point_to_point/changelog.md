# point_to_point — changelog

## 2026-08-19 — Phase 0: initial implementation from `op_design.md`

Self-contained Python CCL op on `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`.
Newly authored from scratch — it does **not** re-export, import, call, wrap, or
dispatch to `ttnn.point_to_point` / `ttnn._ttnn.operations.point_to_point`.

### Shipped

| Path | Purpose |
|---|---|
| `__init__.py` | re-exports `point_to_point`, `validate`, `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS` |
| `point_to_point.py` | four registry declarations, `validate()`, entry point, cached `GlobalSemaphore` |
| `point_to_point_program_descriptor.py` | `MeshProgramDescriptor` (two per-coordinate programs) |
| `kernels/point_to_point_sender_reader.cpp` | sender NCRISC: input DRAM → `cb_shard_pages` |
| `kernels/point_to_point_sender_writer.cpp` | sender BRISC: handshake, framing, fabric egress |
| `kernels/point_to_point_receiver_reader.cpp` | receiver NCRISC: ack, wait, read-back, de-frame |
| `kernels/point_to_point_receiver_writer.cpp` | receiver BRISC: `cb_output_pages` → output DRAM |

No compute kernel — the op is dataflow-only, on logical core `(0, 0)` of the two
participating devices. Every other mesh coordinate gets no program entry, so relay
hops are pure fabric routing.

### Accuracy

Pure byte copy, so the oracle is identity and the comparison is **exact**, not
tolerance-based. `tests/.../test_point_to_point_debug.py` asserts
`max|actual - expected| == 0.0` on the receiver shard **and** on every untouched
shard, for all-ones, monotonic, and tile-position-encoded payloads across
`ROW_MAJOR`/`TILE` and shapes `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,48)`,
`(1,1,24,24)`. The acceptance suite's PCC thresholds (0.995–0.999) are met with
margin everywhere.

### Test results

- **Acceptance: 90 / 90 PASS, 0 failures** (`tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point.py`).
  Run one case per pytest process with a device reset — see *Platform limitation*
  below for why that is required. Per-case verdicts:
  `agent_logs/acceptance_per_case_results.csv`; driver:
  `agent_logs/acceptance_per_case_driver.sh`.
- **Golden suite**: 432 cells collect cleanly; `SUPPORTED == TARGET` and
  `EXCLUSIONS == []`, so no cell is xfailed.

### `SUPPORTED`

All of `TARGET`. The op is format-agnostic by construction: all four CBs carry
opaque bytes (declared `uint32`), the only dtype-dependent rule lives in the bound
`ccl_packet_dims` host helper (the `bfloat16` `bit_floor` on the channel buffer
size), and both framing regimes (A coalesce / B segment) are implemented. So every
dtype × layout × topology × alignment cell rides one code path.
`EXCLUSIONS` is empty.

### Issues encountered

**1. NoC DRAM-read alignment — found and fixed.**
The watcher's NoC sanitizer, on a `(1,1,32,48)` bfloat16 `ROW_MAJOR` shard:

```
NCRISC point_to_point_sender_reader.cpp tried to unicast read 96 bytes to local
L1[0x01b360] from DRAM[0x00593e80] (invalid address alignment in NOC transaction)
```

`l1 % 64 == 32`, `dram % 64 == 0`. Blackhole's `NOC_DRAM_READ_ALIGNMENT_BYTES` is
**64**, and `sanitize.h:558` requires `(l1_addr & mask) == (noc_addr & mask)`. A CB's
page size *is* its per-slot address stride, so the design's
`round_up(page_size, l1_alignment)` = 96 put slot 1 at `base + 96` ≡ 32 (mod 64).
CB base addresses are already DRAM-aligned (`program.cpp:1352`), so only the stride
was wrong.

Fix: `_dram_slot_stride()` sizes `cb_shard_pages` and `cb_output_pages` — the two CBs
that are the local side of a DRAM transfer — at
`round_up(page_size, max(dram_align, l1_align))`. The **intra-packet** stride stays
`round_up(page_size, l1_alignment)` because `ccl_packet_dims` derives `packet_size`
from it. No kernel change was needed: the kernels already address CB slots via
`get_read_ptr`/`get_write_ptr` and compute intra-packet offsets from the `alignment`
compile-time arg, so the two strides were never conflated. This is a CB-sizing
choice, which `op_design.md` marks advisory.

Because the sanitizer *halts* the offending core, this bug also presented as a
device wedge, which is what made it look like a synchronization problem at first.

**2. Multi-packet / multi-hop ethernet wedge — pre-existing, NOT this op.**
After a fabric transfer of more than one packet, or over more than one hop, the next
`open_mesh_device` fails inside
`RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset`:

```
Device 0: Timed out while waiting for active ethernet core 24-25 to become active
again. Try resetting the board.
```

It is strictly **post-test**: pytest reports `PASSED`, the shard comparison is exact,
and only then does the process `SIGABRT` while closing the mesh device. Nothing in
the data path fails. Bisect (all on the graded Blackhole 1×4 QuietBox):

| case | this op | bound C++ `ttnn.point_to_point` |
|---|---|---|
| 16× op-free fabric mesh open/close | PASS | — |
| `GlobalSemaphore` per cycle, no traffic, 4× | PASS | — |
| 1 hop, 1-packet payload, 20 cycles | 20 PASS | PASS |
| 1 hop, 4-packet payload | 1 then wedge | 1 then wedge |
| 2 hops, 1-packet payload | 1 then wedge | 1 then wedge |

The pre-existing C++ op — an **independent** implementation this op deliberately does
not wrap — wedges identically: same core, same count, same duration. So this is a
fabric-teardown limitation of the board, at full parity between the two
implementations. Practical consequence: a pytest process that performs a
multi-packet or multi-hop transfer must be the last to use the board before a reset,
which is why the acceptance suite is driven one case per process.

Hypotheses raised and **falsified** along the way (kept as regression guards in the
debug file so they are not re-litigated):

- *`id(mesh_device)` reuse in the semaphore cache* — 4 unique ids across 4 open/close
  cycles, and the semaphore address is stable at `0x17FFC0` every cycle
  (`test_mesh_device_id_reuse_probe`).
- *`packet_size` overrunning the EDM channel slot* — measured `max_payload = 4352`,
  `header = 48`. No acceptance or golden case oversteps
  (`test_packet_geometry_within_edm_slot` asserts it). This is a **real latent trap**
  worth knowing: `ccl_packet_dims` caps a packet at `header + max_payload` while the
  worker writes its payload at `slot_base + header`, and the only guard is an
  `ASSERT` that omits the header and compiles out in Release. Pages of 16/32/48 B
  with ≥ 274/137/91 pages would overrun an ethernet core's L1. Shared with the C++ op.
- *Half-open EDM connection* — the static analyzer confirmed every constructed
  `FabricStreamSender` is unconditionally opened and closed on every path, matching
  the reference kernels including the construct-before-blocking-wait ordering.

### Tests added

- `tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_debug.py`
  (committed, do not delete):
  - **A** `test_fabric_mesh_open_close_only` — op-free fixture-cycle control
  - **A2** `test_mesh_device_id_reuse_probe` — semaphore-cache key soundness
  - **A3** `test_cpp_reference_op_fixture_cycle` — the C++ op under the identical
    fixture, for diagnosis only (the op itself never references it)
  - **A4** `test_packet_geometry_within_edm_slot` — asserts no case frames a packet
    larger than the EDM payload slot
  - **A5** `test_hop_count_stress` — hop-count × payload-size bisect
  - **B** `test_all_ones_single_tile`, `test_monotonic_exact`,
    `test_tile_position_encoding_row_major` — exact-value payload checks

### Advisory deviations from `op_design.md`

- **CB slot stride.** `cb_shard_pages` / `cb_output_pages` use a DRAM-aligned page
  size instead of the design's `round_up(page_size, l1_alignment)`. Required for
  correctness — see issue 1. The intra-packet stride is unchanged, and all other CB
  sizes/indices follow the design.
- **Stale note in the design.** `op_design.md` states the bound C++ op's coordinate
  order is `(receiver, sender)`. Measured, it is
  `(input, sender_coord, receiver_coord, *, output_tensor, intermediate_tensor, topology)`
  — the same order as this op. No code impact (this op's signature follows the
  design's own mandated order), but the warning is unnecessary.

Everything binding — the algorithm, the dataflow topology and RISC ownership, the
single-core/single-link work distribution, the semaphore handshake contract and its
re-arm ordering, the helper mapping (`FabricStreamSender` → `open(route)` →
`arm_unicast_write`/`arm_inc` → `write_page`/`inc` → `close`, plus `signal()` for the
receiver's one-shot ack), the mandatory 2-argument `TensorAccessor`, and the public
signature — is implemented as specified.
