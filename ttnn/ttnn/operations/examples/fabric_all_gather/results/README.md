# fabric_all_gather send modes on a Blackhole Galaxy — overview

Device-time sweeps of the example's send modes on `bh-glx-120-d06u14` (Blackhole Galaxy, 8 × 4 = 32 chips),
`FABRIC_1D_RING`, 14,336 B router payload, bf16, 4 to 8,192 tiles per chip, 1 and 2 links, 2026-10-22.

| ring | groups run at once | results |
|---|---|---|
| 4 chips (mesh rows, `FAG_AXIS=1`) | 8 | [`4ring_galaxy_bh/`](4ring_galaxy_bh/README.md) |
| 8 chips (mesh columns, `FAG_AXIS=0`) | 4 | [`8ring_galaxy_bh/`](8ring_galaxy_bh/README.md) (answers [`HANDOFF_8RING.md`](../HANDOFF_8RING.md)) |
| 32 chips (snake vs dual cycles) | 1 | not run yet |

Every case was bit-exact on all 32 chips; nothing was refused.

## What was added to the op for these runs

- **`mode="mcast_split"`** (`ms`): the shard as two halves, each multicast both ways: toward p+1 half A over G/2 chips
  and half B over G/2 − 1, toward p−1 the reverse. Every chip gets both halves, and every hop carries G/2 − ½ shards,
  the relay balance's load, with no relays.
- **`mode="mcast_halves"`** (`mh`): half A multicast toward p+1 over all G − 1 other chips, half B toward p−1 over all
  of them. Same hop load as `ms`; one multicast per direction.
- **Multicast with per-chip placement (`@auto`)**: a multicast packet bumps one arrival-counter address on every chip
  it writes, so multicast used to need the same port core on every chip (`uniform`). With any other placement the
  multicast modes now keep those counters on **sink cores** (one per ring, direction and link, the same core on every
  chip); a sink waits for its port's arrivals and re-arms the counter, and the port cores sit next to their Ethernet
  cores.

## Findings

1. **At size, only the busiest hop's bytes matter.** Relay, split multicast and whole-shard multicast all run their
   busiest link at ~24 GB/s per direction; time follows shards on that hop (8-ring: 1228 / 1409 µs = 0.872 against
   3.5 / 4 = 0.875). A whole-shard multicast puts G/2 shards on its busiest hop, the relay balance and the split
   multicasts G/2 − ½.
2. **Splitting the shard, not the hop range, fixes multicast.** Breaking a 4-chip multicast into 2 + 2 sends the same
   bytes over the same links. `ms` / `mh` halve the shard instead and match the relay at size (within 1–2%).
3. **`mh@auto` is the best middle-size choice**: on the 4-ring fastest from 16–64 tiles to the largest; on the
   8-ring fastest at 64–256 tiles (36% ahead of `b@auto` at 64 tiles / 1 link; 26.7 vs 43.5 µs at 256 tiles / 2
   links) and level at size.
4. **Whole-shard multicast with `@auto` wins the smallest shards** (≤ 16 tiles): fewest serial steps.
5. **Placement**: worth 20–30% to multicast at ≤ 64 tiles, and to the relay at 2 links (uniform `b` reaches
   16.6–19.3 GB/s per link at 2 links, `b@auto` 24). Plain `m` / `mb` get up to 16% slower with `@auto` at 1 link and
   large sizes (`m@auto` on the 4-ring); not profiled.
6. **Link count**: every neighbour pair has exactly 2 links. Two links give ~2× at ≥ 1,024 tiles and only 1.1–1.2× at
   ≤ 64 tiles. One link is worth it only to save cores (2 port cores + 1 copy core per link, + 2 sink cores per link
   for multicast `@auto`) or to leave a link to concurrent traffic.

**Best variant per size**

| tiles/chip | 4-ring, 1 link | 4-ring, 2 links | 8-ring, 1 link | 8-ring, 2 links |
|---|---|---|---|---|
| 4 | `m@auto` | `mb@auto` | `mb@auto` | `mb@auto` |
| 16 | `mh@auto` | `m@auto` | `m@auto` | `m@auto` |
| 64 | `mh@auto` | `mh@auto` | `mh@auto` | `mh@auto` |
| 256 | `mh@auto` | `mh@auto` | `mh@auto` ≈ `b@auto` | `mh@auto` |
| ≥ 1,024 | `mh@auto` (1–5% ahead) | `mh@auto` (2–4% ahead) | `b@auto` ≈ `mh@auto` ≈ `ms@auto` | `b@auto` ≈ `ms@auto` ≈ `mh@auto` |

## Link rate on this Galaxy — read it before using a roofline here

Every variant without a placement handicap levels off at **~24 GB/s per link per direction** at large sizes, at 1
and 2 links, on 4- and 8-chip rings alike. On the QB2 box of the handoff, `mb` ran the same 4-ring at ~44 GB/s per
link. The cause (the Galaxy links or their fabric setup, the 4 / 8 groups running at once, packet sizing) has not
been isolated; a single ring alone and a bare one-hop stream would separate them. `mh@auto` exceeds it at 64–256 tiles
(24.7–30.1 GB/s on the 4-ring, 34.4 GB/s at 256 tiles / 2 links on the 8-ring); those re-checked bit-exact over
alternating calls but are not profiled.

The CCL op prompts in `tt_metal/third_party/tt_ops_code_gen/eval/prompts/` carry rooflines with QB2 constants only:
`matmul_reduce_scatter` (`roofline.py`, `link_gbps=48.5`) and `distributed_rms_norm` (DRAM + a measured QB2 exchange
cost). `high_bw_all_reduce` has no roofline. A Galaxy `BOXES` row would need a link rate of ~24 GB/s, not 48.5, and a
re-measured exchange cost.

## Open

- The 32-chip ring sweep: single snake vs `scheme="dual_cycles"`. On a 1D fabric only relay variants follow a ring
  that turns corners (a multicast travels straight along one axis), and the example refuses multicast on 2D fabrics.
- Per-port profiles of `ms`, `mh` and the `@auto` multicast variants (the `m@auto` slowdown, the >24 GB/s points).
- `fag_modes_analyze.py` now prints every variant and a GB/s table; `fag_zones_analyze.py` does not show sink cores
  (their zone is `K_tail`).
