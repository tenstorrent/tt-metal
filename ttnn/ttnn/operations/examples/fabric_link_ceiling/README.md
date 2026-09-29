# fabric_link_ceiling — how fast can the simplest op push bytes over a fabric link?

**Difficulty:** ⭐ T1  ·  **Concept(s):** fabric packet issue (flush every packet vs a ring of headers) · link / direction count
**First profiled on:** `bh-qb-11-special-mstaletovic-for-reservation-93463` · BH (4× p150a) · 2026-09-29 · `13f181e3779`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
Every multi-chip op (all-gather, all-reduce, all-to-all, point-to-point) ends up streaming packets from
a worker core to the fabric router, over an Ethernet link, into a neighbour's L1 or DRAM. Before
optimizing such an op you need a number to compare against: what does a link deliver when the kernel
does nothing but send? This example is that baseline, built as an ordinary single-dispatch
`ttnn.generic_op` so its number is achievable by any op.

## What this isolates — and how
- **Concepts:** (1) how the sender issues packets; (2) how the rate holds up in both directions and
  across links.
- **Isolation setup:** Tensix → fabric → Tensix, no DRAM, no compute. Chips are paired along mesh
  axis 0 (row 0 ↔ row 1 of a 2×2 mesh). On each link, one sender core streams an L1 ring of 8
  full-size packets into the same-address landing ring on its peer. Nothing consumes the landing
  ring, so there is no flow control; the last packet is a fused write + atomic increment, and the
  receiving core waits for it, so the program ends only after every byte has landed.
- **Why it's kernel-level:** how packets are issued, and how many links and directions are driven,
  are decisions made by the kernel and program author. Fabric config and router max payload are
  process-wide settings; they are swept as *parameters* here because they move the ceiling, not as
  kernel choices.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `flush_per_packet` *(baseline)* | One header reused for every packet: write the payload, then write the header with a flush-blocking send. | Each packet waits until it has left the core's L1 before the next one starts. |
| `header_ring` | A ring of 8 pre-routed headers; each packet is issued non-blocking (`send_current_slot_non_blocking`), and the core flushes only when it wraps to a header it is about to rewrite. | Packet issue overlaps with the previous packets' NoC writes. |

Directions: `uni` (row 0 sends, row 1 receives) and `bi` (both rows send and receive at once).

## CLI — measure your own params
Needs a 2×2 mesh (4 chips).

```bash
python -m ttnn.operations.examples.fabric_link_ceiling [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `all` or comma list | `all` | `flush_per_packet`, `header_ring` |
| `--direction` | `all` or comma list | `all` | `uni`, `bi` |
| `--trials` | int | `3` | measured launches per case (median) |
| `--fabric` | comma list | `1d,2d` | fabric config: `1d` = FABRIC_1D, `2d` = FABRIC_2D |
| `--payload` | comma list (bytes) | `4352,8704,14336,15232` | router max payload; every packet is exactly one payload |
| `--links` | comma list | `1,2` | links driven per chip pair (one sender core per link) |
| `--mb` | float | `64` | MiB streamed per link per direction per launch |

```bash
# the headline point only
python -m ttnn.operations.examples.fabric_link_ceiling --fabric 1d --payload 14336 --links 1 --variant header_ring
```

There is no `--kernel-iters`: one launch already streams 64 MiB per link (≥ 1.3 ms), so launch overhead
is negligible.

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box. Full matrix in
[`report.md`](report.md).* GB/s per link per direction, `header_ring`, median of 3:

| Fabric | Router max payload | 1 link, uni | 1 link, bi | 2 links, uni | 2 links, bi |
|---|---|---|---|---|---|
| FABRIC_1D | 4,352 B | 31.1 | 33.6 | 31.1 | 33.6 |
| FABRIC_1D | 8,704 B | 42.7 | 39.4 | 40.0 | 32.2 |
| FABRIC_1D | **14,336 B** | **48.5** | **48.3** | 40.7 | 30.5 |
| FABRIC_1D | 15,232 B | 33.6 | 30.8 | 30.6 | 26.8 |
| FABRIC_2D | 4,352 B | 24.8 | 24.7 | 24.8 | 24.7 |
| FABRIC_2D | 8,704 B | 43.6 | 42.2 | 40.2 | 33.1 |
| FABRIC_2D | 14,336 B | 38.5 | 35.5 | 36.1 | 30.7 |
| FABRIC_2D | 15,232 B | 41.9 | 38.1 | 40.7 | 30.9 |

`flush_per_packet` vs `header_ring` (FABRIC_1D, 1 link): 27.7 vs 31.1 GB/s at 4,352 B (uni) and 27.6 vs
33.6 (bi); 41.8 vs 42.7 at 8,704 B; identical (48.5) at 14,336 B.

**Reading of the result:**
- **The ceiling is 48.5 GB/s per link per direction** on this box: FABRIC_1D, 14,336 B packets, one link.
  Both directions of the link reach it at the same time (48.3 bi), so a link is fully duplex.
- **Packet issue only matters for small packets.** At 4,352 B the ring of headers is 12–22% faster; at
  14,336 B the router, not the sender, sets the pace and both variants are identical.
- **The best router payload is not the largest, and it depends on the fabric.** FABRIC_1D peaks at
  14,336 B, FABRIC_2D at 8,704 B, and 15,232 B (the Blackhole maximum) is 30% below the 1D peak. The
  router splits a fixed amount of buffer space into slots of this size, so the payload also sets how
  many packets can be in flight. Measure your own config; do not assume bigger is better.
- **A second link does not double the rate here.** With two links each link drops to about 40 GB/s
  (uni) and 30–37 GB/s (bi). The two sender cores sit side by side next to the Ethernet row, so their
  traffic into the routers, and the routers' writes into the receiving cores, share NoC paths; core
  placement is a candidate explanation, not yet measured.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_link_ceiling.py -s
```

## Code
`program_descriptor_with_inline_kernels.py` holds both kernels (sender and receiver) and the
`MeshProgramDescriptor` wiring: one program per chip, the fabric connection args appended last to the
sender's runtime args, and a one-hop route that works on both 1D fabrics (route by hop count) and 2D
fabrics (route by destination chip).
