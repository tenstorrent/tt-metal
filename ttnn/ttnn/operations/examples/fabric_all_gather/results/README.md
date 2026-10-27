# fabric_all_gather send modes on a Blackhole Galaxy — overview

Device-time sweeps of the example's send modes on Blackhole Galaxies (8 × 4 = 32 chips), 14,336 B router payload,
bf16, 4 to 8,192 tiles per chip, 1 and 2 links, the same ten variants on two fabrics:
`FABRIC_1D_RING` on `bh-glx-120-d06u14` (2026-10-22) and `FABRIC_2D_TORUS_XY` on `bh-glx-110-a09u02` (2026-10-23).

| ring | groups run at once | `FABRIC_1D_RING` | `FABRIC_2D_TORUS_XY` |
|---|---|---|---|
| 4 chips (mesh rows, `FAG_AXIS=1`) | 8 | [`4ring_galaxy_bh/`](4ring_galaxy_bh/README.md) | [`4ring_galaxy_bh_2d_torus/`](4ring_galaxy_bh_2d_torus/README.md) |
| 8 chips (mesh columns, `FAG_AXIS=0`) | 4 | [`8ring_galaxy_bh/`](8ring_galaxy_bh/README.md) (answers [`HANDOFF_8RING.md`](../HANDOFF_8RING.md)) | [`8ring_galaxy_bh_2d_torus/`](8ring_galaxy_bh_2d_torus/README.md) |
| 32 chips (snake vs dual cycles) | 1 | not run yet | not run yet |

Every case was bit-exact on all 32 chips on both fabrics (880 cases); nothing was refused.

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
- **Multicast and direct modes on a 2D fabric** (for the 2D runs): every route, multicast or unicast, now runs by hop
  count along its connection's physical direction (`fabric_set_mcast_route` / `fabric_set_route` with the E/W/N/S
  direction from `ttnn.get_eth_forwarding_direction`) instead of by destination. A destination route lets the fabric
  pick the path: a 2-hop destination on a 4-ring is a tie, a 3-hop one goes the short way round, the other way from
  the router it was injected on, and the fence's *ready* packets sent that way never arrived (every variant with such
  a ready hung). On 2D the non-relay modes need a straight ring (`cluster_axis` 0 / 1); snake and dual cycles stay
  relay-only.

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

## FABRIC_2D_TORUS_XY against FABRIC_1D_RING

1. **The relay does not care which fabric it runs on.** `b@auto` is within 1–4% of its 1D time from 256 tiles up, on
   both rings and at both link counts, and runs its busiest hop at 23.2–23.7 GB/s per link direction at size.
2. **Long multicasts lose a quarter of the link rate on 2D.** Multicast over at most 2 chips (4-ring `m`, `mb`, `ms`)
   matches 1D at size. Multicast over 3 or more chips (4-ring `mh`; 8-ring `m`, `ms`, `mh`) levels off at 16–18 GB/s
   per link direction, 26–49% slower than on 1D at 8,192 tiles and 1 link, and worse at 2 links. The exception is the
   8-ring's `mb` (22.8 GB/s), whose long legs alternate direction. Per-port profile: on 2D the links are just as full
   (most of each leg waits for a fabric slot) but carry less; the router-side cause is not isolated.
3. **So on 2D the relay is the large-size choice**, not the split multicasts: `b@auto` is fastest from 1,024 tiles
   everywhere, and on the 8-ring from 64 tiles (1 link) / 256 tiles (2 links). `mh@auto` keeps its 64-tile win only on
   the 4-ring.
4. **Small sizes**: whole-shard multicast `@auto` still wins at ≤ 16 tiles. At ≤ 16 tiles the relay is 2–15% slower
   than on 1D and the multicasts up to 41% (8-ring `m@auto`, 16 tiles, 1 link). The two fabrics ran on different
   Galaxies, so this part is not attributed to the fabric.

**Best variant per size on FABRIC_2D_TORUS_XY**

| tiles/chip | 4-ring, 1 link | 4-ring, 2 links | 8-ring, 1 link | 8-ring, 2 links |
|---|---|---|---|---|
| 4 | `mb@auto` | `mb@auto` | `m@auto` | `m@auto` |
| 16 | `mh@auto` | `mb@auto` | `m@auto` | `m@auto` |
| 64 | `mh@auto` | `mh@auto` | `b@auto` | `b@auto` ≈ `m@auto` |
| 256 | `ms@auto` ≈ `b@auto` | `mh@auto` ≈ `b@auto` | `b@auto` | `b@auto` |
| ≥ 1,024 | `b@auto` ≈ `ms@auto` | `b@auto` | `b@auto` | `b@auto` |

## Link rate on this Galaxy — read it before using a roofline here

Every variant without a placement handicap levels off at **~24 GB/s per link per direction** at large sizes, at 1
and 2 links, on 4- and 8-chip rings alike (on `FABRIC_2D_TORUS_XY` too, on a second Galaxy, except the long
multicasts above). On the QB2 box of the handoff, `mb` ran the same 4-ring at ~44 GB/s per
link. The cause (the Galaxy links or their fabric setup, the 4 / 8 groups running at once, packet sizing) has not
been isolated; a single ring alone and a bare one-hop stream would separate them. `mh@auto` exceeds it at 64–256 tiles
(24.7–30.1 GB/s on the 4-ring, 34.4 GB/s at 256 tiles / 2 links on the 8-ring); those re-checked bit-exact over
alternating calls but are not profiled.

The CCL op prompts in `tt_metal/third_party/tt_ops_code_gen/eval/prompts/` carry rooflines with QB2 constants only:
`matmul_reduce_scatter` (`roofline.py`, `link_gbps=48.5`) and `distributed_rms_norm` (DRAM + a measured QB2 exchange
cost). `high_bw_all_reduce` has no roofline. A Galaxy `BOXES` row would need a link rate of ~24 GB/s, not 48.5, and a
re-measured exchange cost.

## Open

- The 32-chip ring sweep: single snake vs `scheme="dual_cycles"`. Only relay variants follow a ring that turns
  corners, on either fabric (a multicast travels straight along one axis).
- Why long multicasts run at 16–18 GB/s on the 2D router (and why the 8-ring's `mb` does not): a bare multi-hop
  multicast stream on 2D vs 1D, and per-port profiles of `ms` / `mh` on 2D.
- The 2D runs on the 1D results' Galaxy (or the 1D ones on `bh-glx-110-a09u02`), to separate fabric from box at small
  sizes.
- Per-port profiles of `ms`, `mh` and the `@auto` multicast variants (the `m@auto` slowdown, the >24 GB/s points).
- `fag_modes_analyze.py` now prints every variant and a GB/s table; `fag_zones_analyze.py` does not show sink cores
  (their zone is `K_tail`).
