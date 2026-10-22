# Handoff: fabric_all_gather send modes on an 8-chip ring

> **Status (2026-10-22): done on a Blackhole Galaxy** (`bh-glx-120-d06u14`, `FAG_MESH=8,4 FAG_AXIS=0`). Results and
> the answer to the question below: [`results/8ring_galaxy_bh/`](results/8ring_galaxy_bh/README.md); the 4-ring rerun
> on the same box and the new `mcast_split` / `mcast_halves` / multicast `@auto` variants:
> [`results/README.md`](results/README.md). Not run: the optional `base,d,db` variants.

Branch: `malimpic/ccl_code_gen` (everything CCL-code-gen related goes here). Owner: Milan Alimpic.

## What you are asked to do

Measure, on **device time**, how two multicast send patterns of the all-gather example compare on an **8-chip 1D
ring**, over a wide range of sizes, and profile them per port so the result can be explained. The numbers feed a
future perf example for CCL code generation. The 4-chip results below were taken on a 4x Blackhole box (QB2,
`FABRIC_1D_RING`); that box has no 8-chip ring, which is why this runs elsewhere.

The two patterns (every chip sends its own shard to the other 7 with exactly two multicasts, no relays):

| variant | chips at even ring position | chips at odd ring position |
|---|---|---|
| `m`  (same direction)  | 4-chip mcast toward p+1, 3-chip mcast toward p-1 | same |
| `mb` (alternating)     | 4-chip mcast toward p-1, 3-chip mcast toward p+1 | 4-chip toward p+1, 3-chip toward p-1 |

No code change is needed for this: `mode="mcast"` already sends one multicast over G//2 chips one way and one over
G-1-G//2 the other (`_schedule_routes` in `program_descriptor_with_inline_kernels.py`); `balance=True` swaps the two
legs on even positions. Expected link loads (shards per link direction): `m` 4 on every p->p+1 hop, 3 on every
p->p-1 hop; `mb` alternating 4 / 3 in both directions. Busiest hop = 4 in both, so any difference is not link bytes.

Run the relay baselines alongside: `b` (relay, far shard split between the directions) and `b@auto` (same, with the
per-chip placement next to the Ethernet cores — the fastest thing on the 4-ring at large sizes).

## Where to run

Any machine with an 8-chip ring on which `FABRIC_1D_RING` comes up:

| machine | settings |
|---|---|
| T3K (8x Wormhole, a physical ring of 8) | `FAG_MESH=1,8 FAG_AXIS=1`, and set `FAG_PAYLOAD` to a Wormhole-legal fabric payload (see below) |
| Galaxy (8x4) | `FAG_MESH=8,4 FAG_AXIS=0`: four 8-chip rings (the columns) run at once; needs the wrap-around links on axis 0 |
| two QB2 Blackhole boxes cabled as one 8-ring | `FAG_MESH=1,8 FAG_AXIS=1` (default payload 14336 is the Blackhole value used on the 4-ring) |

The mcast variants are **1D fabric only** (the host raises `ValueError` on a 2D fabric: 2D multicast follows one
fixed path per direction that need not be the ring). Always pass `MODES_FABRICS=1d_ring` unless you also want relay
numbers on a 2D fabric.

`FAG_PAYLOAD` (bytes, default 14336) goes into `FabricRouterConfig.max_packet_payload_size_bytes`. 14336 is right for
Blackhole. On Wormhole, if device open fails or the router refuses it, use a smaller value (try 4096 or the default
that `ttnn.get_tt_fabric_max_payload_size_bytes()` reports when the config is left alone) and record which you used.
The op itself sizes its chunks from `ttnn.get_tt_fabric_max_payload_size_bytes()`, so nothing else changes.

## Files

| file | what |
|---|---|
| `ttnn/ttnn/operations/examples/fabric_all_gather/program_descriptor_with_inline_kernels.py` | the example. `mode` = `relay` / `direct` / `mcast`, `balance`, `placement` (`auto` = probe-based per-chip, `uniform` = chip 0's port cores on every chip, which mcast needs). Device zones `S_*` / `R_*` + wait counters compile in only under the device profiler. |
| `tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_modes.py` | the sweep: bit-exact check on every chip, then traced device time (see Method). Prints `FAG_MODES ... device_us=` |
| `tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_zones.py` | per-port profiling run (under `--profile`). Prints `FAG_PLAN` (per chip: device id, ring position, port cores, routes) and `FAG_ZONES` (op ids per case) |
| `tests/ttnn/unit_tests/operations/examples/fag_modes_analyze.py` | tables from the sweep log |
| `tests/ttnn/unit_tests/operations/examples/fag_zones_analyze.py` | per-port tables from the profiling log + `profile_log_device.csv` |

Environment knobs (both tests): `FAG_MESH` (default `1,4`), `FAG_AXIS` (default `1`), `FAG_PAYLOAD` (default
`14336`). Sweep: `MODES_FABRICS` (`1d_ring,2d_torus_xy`), `MODES_LINKS` (`1,2`), `MODES_VARIANTS`
(`base,b,d,db,m,mb,base@auto,b@auto`), `MODES_SHAPES` (`H,W;H,W;...`, per-chip shard), `MODES_ROUNDS` (8),
`MODES_REPLAYS` (5). Profiling: `ZONES_VARIANTS` (`b,m,mb`), `ZONES_LINKS` (`1,2`), `ZONES_SHAPES`.

## Method (keep it — the 4-ring numbers were taken this way)

Device time as a traced model sees it: `[filler, all-gather] x ROUNDS` captured in one trace, replayed `REPLAYS`
times. The filler is a ~113 us `ttnn.add` on (1,1,2048,4096) bf16; it hides dispatch (the dispatcher prefetches up to 7
programs ahead) and lines the chips up. Per call, the **slowest chip's** program duration from the real-time profiler,
picked out by runtime ID; median over calls. Compare device time only with device time.

## Steps

1. Build the branch (`./build_metal.sh`), `source python_env/bin/activate`. Kernels compile at runtime.
2. Smoke test (one small shape, must print `bitexact=1` for all three):
   ```bash
   FAG_MESH=1,8 FAG_AXIS=1 MODES_FABRICS=1d_ring MODES_LINKS=1 MODES_VARIANTS=b,m,mb MODES_SHAPES="128,128" \
   scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_modes.py -s
   ```
3. Full sweep (default shape ladder: 64² .. 2048x4096 plus tall / wide shards; 1 and 2 links):
   ```bash
   FAG_MESH=1,8 FAG_AXIS=1 MODES_FABRICS=1d_ring MODES_VARIANTS=b,m,mb,b@auto \
   scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_modes.py -s \
     2>&1 | tee fag_modes_8ring.log
   python3 tests/ttnn/unit_tests/operations/examples/fag_modes_analyze.py fag_modes_8ring.log
   ```
   Add `base,d,db` to `MODES_VARIANTS` if time allows (on the 4-ring `d`/`db` never won). On a Galaxy, add
   `MODES_LINKS=1,2,...` as far as the axis has links.
4. Per-port profiling (four sizes x 1/2 links x `b,m,mb`, ~a minute):
   ```bash
   FAG_MESH=1,8 FAG_AXIS=1 scripts/run_safe_pytest.sh --profile \
     tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather_zones.py -s 2>&1 | tee fag_zones_8ring.log
   python3 tests/ttnn/unit_tests/operations/examples/fag_zones_analyze.py fag_zones_8ring.log \
     generated/profiler/.logs/profile_log_device.csv > fag_zones_8ring_tables.txt
   ```
   The `--profile` run ends with `SAFE_PYTEST_RESULT: FAIL` because tracy's post-processing chokes on the CSV
   (`pandas.errors.ParserError`). That is harmless: check for `1 passed` and that `profile_log_device.csv` was written.

## Reading the per-port tables

Per case: kernel span per chip, and `first send->end` per chip. Per port (`fwd<link>` = toward p+1, `bwd<link>` =
toward p-1), medians over calls, in us:
- `fence`: readies exchanged with the destination chips (cross-chip skew lands here).
- `S_e<k>`: sending entry k; `s` = waiting for a free fabric slot (the link / router is the limit), `c` = waiting for
  the reader. Duration minus both = the port core's own NoC writes into the Ethernet core: when that dominates with
  little slot wait, the port core (its placement) is the bottleneck, not the fabric.
- `tail`: after sending, waiting for everything from upstream to land (idle time on that port).
- `R_e<k>`: the reader; `a` = waiting for upstream arrivals (relays), `c` = blocked on CB space (sender is slower).

## 4-ring results to compare against (Blackhole QB2, `FABRIC_1D_RING`, uniform placement, device us)

| tiles/chip | 1 link: b | m | mb | 2 links: b | m | mb |
|---|---|---|---|---|---|---|
| 16    | 8.6   | 8.0   | **7.2** | — | — | — |
| 64    | **12.8** | 15.0 | 12.9 | 11.3 | 10.5 | **10.4** |
| 256   | 30.4  | 34.8  | **28.9** | — | — | — |
| 1024  | 110.7 | 122.1 | **100.8** | 99.7 | **86.2** | 90.1 |
| 8192  | 866   | 924   | **756** | 773 | **661** | 694 |

`b@auto` (per-chip placement) beats every uniform variant from 64 tiles up: 1024 tiles at 2 links = 46 us vs 86.

Profiling explained it (1 link, 4096 tiles: `m` 465 us, `mb` 382 us):
- `m`: every p->p+1 link carries 2 shards **and** every one of them carries a multicast's first hop (the receiving
  router writes locally and forwards). Those links ran at ~36 GB/s; the even chips' 2-hop mcast waited 280 us on
  fabric slots and set the time, while the p->p-1 ports finished at 184-296 us and idled.
- `mb`: the 2-shard link directions carry only traffic that terminates there (the unicast + another mcast's last
  hop); every write-and-forward first hop rides a 1-shard link. Those links ran at ~45-47 GB/s (≈ an uncontended
  unicast) and every port finished within 340-376 us.
- Uniform placement makes some ports slow with almost no slot wait (port core far from its Ethernet core); at
  2 links the slowest port in both `m` and `mb` was the same unicast on one chip (12 GB/s), so `m`'s 2-link win is
  that port, not the pattern.

**The question for the 8-ring:** with 4- and 3-hop multicasts, the write-and-forward hops (3 + 2 per chip) can no
longer all avoid the heavily loaded links. Does alternating still win, by how much, at which sizes and link counts —
and does the per-port profile show the same mechanism (slot waits on the 4-shard links, idle tails on the others)?

## Gotchas

- Run `scripts/run_safe_pytest.sh` (or `scripts/tt-probe.sh`) **in the foreground only**; never plain pytest / python
  on device. **Never run `tt-smi -r`** — the script resets the board after a run, including Ethernet cores left stuck by a hang.
- `ttnn.generic_op` caches programs by compile-time args, not runtime-arg counts. The example appends a variant
  descriptor (`variant_ct`) to the reader / sender compile-time args so that a variant with longer runtime-arg lists
  doesn't reuse a stale program (that hung). If you add a variant, make sure it changes `variant_ct`.
- mcast needs the receiving port on the same core of every chip in its range → `placement="uniform"` for `m`/`mb`;
  the `@auto` suffix only works with relay variants. This handicaps mcast at large sizes; say so in the results.
- Profiling zones: `ZONE_K` names entries `_e0`, `_e1`, `_e2`, with every entry >= 2 sharing `_e2`. mcast ports have
  1 entry, fine; relay `b` on G = 8 has up to 4 entries per port, so its `S_e2`/`R_e2` columns are only the last
  one. Widen `ZONE_K` (kernel source, `_CHUNK_WALK`) if you need relay per-entry detail.
- `fag_zones_analyze.py` maps profiler cores to ports with the virtual coordinates printed in `FAG_PLAN` (from one
  translation for the whole mesh); a chip that translates differently is matched by elimination among the cores
  that ran `S_` zones. Check that every chip shows all its ports.
- Profiler run host ID = op id << 10 | chip; `FAG_ZONES rids=` are op ids. The pre-collect pass prints a second set
  of `FAG_ZONES` lines; the analysis takes whichever op ids it finds in the CSV.
- On a Galaxy with `FAG_AXIS=0`, the four column rings share nothing but run concurrently; the device time is the
  slowest chip of the whole mesh.

## What to bring back

Commit to `malimpic/ccl_code_gen`, under `ttnn/ttnn/operations/examples/fabric_all_gather/results/8ring_<machine>/`:
`fag_modes_8ring.log` (the `FAG_MODES` lines are enough), the `fag_modes_analyze.py` output, and
`fag_zones_8ring_tables.txt` (not the profiler CSV — it is hundreds of MB). In the commit message / a short
`README.md` there: machine, arch, fabric, payload, mesh / axis, and per link count which variant wins at which sizes,
with the profiling explanation in the terms above. Note anything that refused to run (and the error).
