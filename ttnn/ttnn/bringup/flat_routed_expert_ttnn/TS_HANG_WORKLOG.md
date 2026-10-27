# Token-split experts hang hunt (GLM-5.3-Flash on the K3 block, LoudBox bh-lb-09, 12x10, 200G)

Raw data: `/localdev/mstaletovic/hang_campaign/<day>/runs.jsonl` (one row per run: spec, result, MoE calls, hang
site, triage summary) + `runs/<id>/` (log, triage, spec). Driver / report:
`models/demos/glm53_flash_d_p_lb/tests/hang/campaign.py` (`run` / `report`). Policy: once a condition reproduces,
narrow it (repeat, then one variable at a time toward the smallest repro), triage on every hang.

## The one hang so far (2026-10-18 17:42 UTC)

- Workload: `zz_scratch_babilong_think.py`, GLM_EXPERTS_MODE=ts, decode via prefill (chunk 2048 = S 256 per chip,
  max_seq 8192), qa1 4k sample 1 (3100-token prompt), eager, the host sampling each token.
- Where: op 111396 = **layer 20's fused flat+combine op, decode token 27** (the process's 29th forward pass,
  ~1150th fused call; every program warm since pass 1). Mapped by rerunning the same command with GLM_TS_OPLOG=1:
  a decode pass is exactly 3853 device ops; offset 1450 into pass 26 falls in layer 20's routed() ops 1438..1451,
  its second-to-last op (the fused op; tilize after) - matches the triage's op window.
- Triage (re-read 2026-10-18 20:00): the root site is the **x multicast on device 2**: check_cb_inactive "Device 2
  (1,4): NOC0 CB0 active - NoC is likely hung"; (1,4) ncrisc se11_xmc is stuck in ncrisc_noc_fast_write_any_len <
  noc_async_write_multicast (se11_xmc.cpp:245, a piece of a LINKED chain on NOC0, its command buffer never frees). The
  other primary (8,3) waits for freed slots (:191), the helpers (8,4) (1,5) in noc_semaphore_wait_min (se13_xhelp.cpp:62),
  the x readers in cb_reserve_back. The y writers (se6_dw, 17 cores, all device 2) mostly sit at se_yrm.hpp:103-107
  waiting for compute output - downstream of the missing x, not a grant problem; every chip's combine waits for rows.
  Config: CMB_DRISC=1, DRISC_DUAL=0, ND=32, NH=1, X1=0: x reads NOC1, x multicasts NOC0 (linked chains), DRISC weight
  writes NOC0. Fabric routers all just polling.
- Audits: codex (combine signalling): no plain-vs-atomic word mixing, no cross-link ordering assumption (a ring stays
  on one link; landed is local NoC after tagged-write completion; credit snapshots only after consumption); its
  top lead was the fused port core blocking in fabric sends - but the triage puts the stall upstream, in the x
  multicast. Follow-up questions to codex + a subagent: linked-chain deadlock with other NOC0 traffic.
- The same command rerun at 18:04 (identical host op sequence up to the hang, the same binaries: nothing compiled)
  passed -> not a deterministic wrong program.

## Prior art: the MiMo NoC0 wedge (2026-10-01/02, mimo-v2-dp branch)

- 1cabb2985c7 "gate/up data movement (se5_recv) on NOC1 - fixes the rare NoC0 wedge (relay multicast + gate/up writes
  + DRAM reads)"; the factory comment (gate/up cores block): "a relay stuck in noc_cmd_buf_ready or in its write
  barrier; ~1 in 1e4..2e5 launches, any chip, a hardware hang the NoC probe test reproduces from the traffic alone
  (tests/ttnn/unit_tests/operations/noc_hang_probe, op_skeleton - in the tt-metal-mimo worktree only); with these
  writes on NOC1 it never did." Our triage has the same signature (relay se11_xmc spinning in noc_cmd_buf_ready,
  the only active command buffer on the mesh).
- Marko (2026-10-18): MiMo also hung with the linked multicast OFF - the wedge is the NOC0 traffic mix (multicast into
  the gate/up rectangle + other writes into those cores + DRAM reads), not the linked path reservation; the fix was to
  move the competing writes to NOC1. So MIMO_FL_XMC_UNLINKED is not a fix candidate (dropped from the perf A/B); the
  candidates move today's competing NOC0 writes (DRISC job 1: MIMO_FL_DRISC_JOB1_NOC=1) or the multicast itself.
- What is on NOC0 today (GLM-Flash plan, 2026-10-12 table + DRISC since 10-14/15): the x relays' LINKED multicasts
  into the gate/up rectangle, the Tensix readers' DRAM weight reads (fallback calls, max rows/expert > 96), the down
  cores' se6_dw down-weight DRAM reads AND the y writes (GLM-Flash: no reader tails), the helpers' sends to the
  primary; pure DRISC: each bank's DRISC serves one slot on NOC0 and one on NOC1 - i.e. DRISC weight WRITES into
  gate/up cores on NOC0, the kind of write stream into the multicast rectangle the 10-02 fix took off NOC0.
- Open question: which mode layer 20's hung call ran (DRISC <= 96 rows/expert, or Tensix fallback): the routing capture
  (campaign capture_dvp_routing) answers it; decode via prefill pads the chunk (one pad token -> 8 hot experts).

## Audits (2026-10-18 evening: subagent + codex, read-only)

- Both: the combine / hand-off / planner / dispatch signalling is sound (every signal after its data is acked, no
  plain write to an atomically bumped word, a credit block has one writer on one fabric connection - nothing relies
  on order across links / routing planes; per-call zeroing ordered by the dispatch barrier). Minor: sparse_combine_rf
  `pend = rtot - rdone` can underflow briefly (duplicate credit snapshots after the drain, harmless).
- Both: the hang is a NOC0 stall of device 2's left relay (physical (2,6)) mid linked chain. XMC_WHOLE_SB is on, so
  the free-slot wait is BEFORE a chain; inside it the only wait is noc_cmd_buf_ready -> a NoC-level stall. Suspect: the
  DRISC job-1 weight writes on NOC0 (flat_drisc_noc: one tile per bank, job 0 NOC1, job 1 NOC0; DRISC (0,6) writes
  along row 6 = the chain's first hop (2,6)->(3,6); others share the column-3 wrap to the chain's start corner) plus the
  down-weight DRAM read responses / y writes (se6_dw, NOC0) crossing the rectangle rows. Rectangles exclude the
  non-Tensix columns (logical 0-6 -> physical 1-7, 7-11 -> 11-15).
- Discriminators (once a rate exists; ~5300 clean calls for 99% at 1/1150): MIMO_FL_XMC_UNLINKED=1, MIMO_FL_XMC_POSTED=1,
  MIMO_FL_DRISC_JOB1_NOC=1 (the narrow fix candidate), MIMO_FL_DRISC_XNOC=1, MIMO_FL_CMB_DRISC=0. Amplifier:
  MIMO_FL_DRISC_NOC1=0 (all DRISC jobs on NOC0) - check every amplified hang has the same triage signature.

## Side finding: the noc_async_writes_flushed source-reuse race (NoC team note, 2026-10)

`noc_async_writes_flushed` (= the NIU's nonposted-writes-SENT count) does not mean the last packet's data has left L1:
a store to the source within ~16 cycles can go out instead (seen 6/10 on WH, never in 50K runs on BH; fix:
`noc_async_write_flushed_with_trid(0)` for default-trid writes, or the new async_writes_departed). Our instance: the
hung relay itself, se11_xmc.cpp:200-212 - XARR multicast from `word`, wait ncrisc_noc_nonposted_writes_sent, then
`*word = sent + n` for the next chain. Worst case the receivers see the next count early and consume x blocks not yet
sent (silent corruption, counts stay consistent - not this NoC-level stall). Fix as a switchable variant later; do not
change the default kernels mid-campaign. Other flushed-then-reuse sites to review: combine_rf dispatch_prep.py:117,
fabric_all_gather_py.py:263, sparse_all_gather_2d.py:410/480, sparse_combine_rf.py:413/439, se11_xmc.cpp:273.

## Narrowing to the op

Queued test_flat_stress.py (the standalone flat op on all 8 chips, bit-exact determinism per pattern, old capacity
8192 call) with the ts env (CMB_DRISC=1, DRISC_DUAL=0, ND=32, NH=1), plain and amplified (MIMO_FL_DRISC_NOC1=0).
MiMo's wedge (per Marko): routing did not matter - one configuration hung non-deterministically.

## Ruled out / checked

- Wrong program from a cache (S 320/640 build served for S 256): the JIT kernel hash covers the source text, compile
  args, defines and CB formats (the S 256 and S 640 se6_drecv builds are separate binaries: CB 9 only at S 640); the
  hung process had one block (S 256, built once at layer 3); the fused op's program key covers every baked address
  and the tensor specs; the flat weight cache key is the plan's packing fields. And the identical rerun passed.
- Cold start: the hang was warm (above).

## Stress without a hang (MoE calls)

| workload | S | calls | result |
|---|---|---|---|
| block traced, uniform / hot / golden L4, fixed routing (fake weights) | 256 | 60K | pass |
| block eager | 256 | 60K | pass |
| block, 120 rebuilds (first-call misses) | 256 | 240 | pass |
| block traced, on-device routing rotation | 256 | 60K | pass |
| whole model traced chunk, 2000 replays (real weights) | 640 | 90K | pass |
| whole model traced chunk, start 2048, 4000 replays (real weights) | 256 | 168K | pass (244 ms traced vs 625 ms eager: eager S 256 is host bound - the device idles between ops, the hung run's regime) |

Note: the whole-model traced output differs from eager (not bitwise deterministic) - to fold into the determinism check.

## Campaign log

(one entry per wave: hypothesis, queue, run ids, result, conclusion)

### Wave 1 (19:17-20:12 UTC): reproduced in the flat op alone

- dvp_ts_s256 (24.6K MoE calls), dvp_ts_s256_x20 (115K): PASS. capture_dvp_routing: decode routing has 215-1558 max
  rows / expert (29-53 experts > 96) - in our DRISC-only config (MIMO_FL_DRISC_DUAL=0, no Tensix readers) every call
  is DRISC anyway; counts > 224 (ALT_MAXCNT) pick one tile per bank, job 0 NOC1 / job 1 NOC0.
- **op_stress_ts_amp_drisc_noc0** (test_flat_stress.py, the standalone flat op on all 8 chips, ts env +
  MIMO_FL_DRISC_NOC1=0 = every DRISC job on NOC0): **HANG at 80-90K mesh calls** (run 1018-200502). Triage = the
  original signature: device 1, the LEFT primary x relay logical (1,4) se11_xmc at :245 spinning in noc_cmd_buf_ready
  (noc_nonblocking_api.h:922), the only active command buffer on the mesh (NOC0 CB0); right primary (8,3) at :191,
  helpers at se13_xhelp:62; the hung chip's DRISCs alive, polling grants (se_drisc_gu.cpp:288-301). Same relay core as
  the model hang (device 2 there). -> no block / combine / model needed; NOC0 write traffic raises the rate.
- Rate under the amplifier: 3 runs, 3 hangs (80K, 60K, 10K mesh calls: ~1 / 50K). Every hang - the model one and
  these three - has exactly ONE active command buffer on the mesh: NOC0 CB0 of logical (1,4), the LEFT primary relay,
  on devices 2 / 1 / 5 / 7: any chip, always that core.
- Perf baseline (scratch_flat_combine_rf, model layout aop NH 1, S 256, uniform): flat alone 1209 us, fused 1231 us
  (weight-streaming bound: the DRISC weight path is what a fix can cost).
- Fix candidate MIMO_FL_DRISC_JOB1_NOC=1 (every DRISC weight job on NOC1): flat 1208 / 1316 us vs 1209 / 1315
  (uniform / zipf spread), fused 1234 / 1344 vs 1231 / 1338 - perf-neutral (within ~0.5%). The amplifier cannot test it
  (it forces those writes back to NOC0): validation = interleaved long runs default vs JOB1_NOC=1 (6 x 200K each).
- Perf A/B, S 256, fused flat+combine us (uniform / zipf spread 1.2), model layout (aop, NH 1):
  base 1231 / 1338; DRISC job 1 on NOC1 1234 / 1344 (+0.2 / +0.4%: neutral); posted x multicast 1232 / 1341
  (neutral); unlinked 1235 / 1664 (+24% spread); DN_NOC0 swap 1255 / 1536 (+2 / +15%); x multicast on NOC1 (DRISC_XNOC)
  1362 / 1820 (+11 / +36%). S 640 base (reference): flat 1305 / 1616, fused 2250 / 2155. Only JOB1_NOC=1 and
  XMC_POSTED are perf-neutral fix candidates.
- MIMO_FL_Y_NC=0 (BRISC y writes) is incompatible with the fused L1 hand-off: perf_s256_y_nc0 hung on its first
  fused call (op 90, combine kernels waiting) - a config hang, not the wedge; its standalone-op amp test stays queued.
- Marko: the MiMo wedge had three traffic classes on the multicasting NoC; moving any ONE to the other NoC stopped
  the hangs (maybe only made them rarer). On NOC0 today: the x multicast, DRISC job-1 weight writes (+ data-sem
  atomics), se6_dw down-weight DRAM reads AND y writes, helper -> primary sends. Each candidate removes one class:
  JOB1_NOC=1, Y_NC=0, DN_NOC0=1 (swap), or the multicast's own form (unlinked / posted / NOC1). Idea (Marko): move the
  relay core instead of the multicast's NoC - modelled with tt-npe (subagent).
- **amp + MIMO_FL_DRISC_XNOC=1 (x multicast on NOC1): PASS, 200K mesh calls** (amp alone: 3/3 hung within 80K) ->
  the wedge needs the x multicast and the DRISC writes on the same NoC (NOC0). Too slow as a fix (+11 / +36%).
- Traffic-only probe (hang/test_noc_wedge_probe.py, roles from the live plan, DRISC senders emulated from the worker
  cells next to their DRAM tiles, nothing that waits): the first smoke run deadlocked every chip on its FIRST launch with
  every role parked in its final read / write barrier and no command buffer active - y writes to DRAM on VC 2 next to
  full-rate DRAM reads on the default VC = BH-76 ("different VCs to the DRAM endpoint deadlock", noted in MiMo's probe),
  a different erratum. The probe's y VC is now NOC_PROBE_Y_VC (default 1).
- Next (queued, continue_on_hang): the amp config x2 more (rate), amp + MIMO_FL_DRISC_XNOC=1 (x multicast on NOC1),
  amp + MIMO_FL_XMC_POSTED=1, the plain ts config x3 (base rate); then the perf A/B of the fix candidates.

### Wave 2 (20:20-21:45 UTC): the fix candidates, the traffic-only probe

- tt-npe subagent (scratchpad npe_flat/README.md, glm_rolemap.json): in DRISC-only mode each bank's ALT tile (its
  NOC1 endpoint, NOC1 NIU in NOC2AXI mode) serves slots 8-15 on EVERY call, hard-wired to NOC0
  (SE_DRISC_ALT_MAXCNT 0xFFFFFFFF when !dual) - half the gate/up weights always reach the gate/up cores on NOC0;
  MIMO_FL_DRISC_JOB1_NOC=1 moves nothing (the free tile's job 1 is never posted; it only reshuffles receivers ->
  identical perf). The left relay's first hop (2,6)->(3,6) carries 180 KB / 33.6 us window (770 under the amplifier,
  which also puts DRISC NOC0 writes into all 40 rectangle cores); the right relay's 59 - fits "always the left relay".
  tt-npe sees no saturation (0.2%): it is a hardware deadlock, not bandwidth. Candidate relay cell: (1,0) (first hop
  carries no DRISC writes).
- Real op, amplified (op_stress, MIMO_FL_DRISC_NOC1=0): + x multicast on NOC1 (DRISC_XNOC): 2 x 200K PASS;
  + posted x multicast (MIMO_FL_XMC_POSTED=1): 200K PASS (amp alone ~1 / 50K: P(0 in 200K) ~ 2%), bit-exact
  determinism markers 0.
- Perf (S 256 fused, uniform / spread): one DRISC tile per bank on NOC1 (ALT_MAXCNT=0 + JOB1_NOC=1) 1572 / 1710 vs
  1231 / 1338 = +28% -> rejected. Posted +0.1 / +0.2%.
- Traffic-only probe (fixed: y VC 1 to avoid BH-76; dynamic-NoC mode where a core's two RISCs share a NoC; DRISC slots
  from the role map): **default traffic (probe_base2) wedged with the exact signature - device 4 (1,4) NOC0 CB0
  active** - nothing in the probe can logically hang -> a hardware NoC wedge of the traffic mix. DRISC writes all on
  NOC1 (probe_one_tile_noc1): 2 x 20K launches clean. amp + unlinked: (1,4) wedged again (as MiMo: unlinked does not
  fix it). amp + posted and amp + relay (1,0): wedged, but at the DRISC EMULATOR cells (several DRAM tiles mapped onto
  one worker cell, unpaced 9 KB writes - much more than a DRAM tile's ~28 GB/s) - inconclusive; rerunning paced
  (NOC_PROBE_DRISC_GAP=450 cycles / block).
- Leading fix: MIMO_FL_XMC_POSTED=1 (perf-neutral; real op amp 200K clean). Open: the paced probe, more amp runs, the
  default config + posted, perf on S 640 / packed / K3, codex review (posted reuses its source after
  posted_writes_sent - the NoC team's flushed race), then millions of iterations fused + single chip.

### Wave 3 (21:45-00:00 UTC): posted is not a fix; y off NOC0 only lowers the rate

- Real op amp + posted: 5 of 6 runs hung (530K calls, ~9.4 / M vs ~20 / M amp alone), all with the (1,4) signature ->
  posted halves the rate at best. Default + posted 3 x 200K clean (meaningless: default alone 1 hang / 450K).
- Default config, op alone, no amp: op_stress_ts_r0 HANG at 50K ((1,4) signature, device 5); r1, r2 200K clean.
- Real op amp + y on NOC1 (MIMO_FL_Y_NC=0, BRISC writes y; standalone op): 2 x 200K clean. amp + unlinked: 2/2 hung;
  amp + DN_NOC0: 2/2 hung (no op mapped). Perf posted: packed 2216 vs 2218, S 640 +0.4 / +0.6%, K3 2437 / 2688 vs
  2442 / 2688 (all neutral - moot).
- Paced traffic-only probe, launches to the wedge: base 1000 / 4500 ((1,4)); posted 5050 / 1650 ((6,0), (8,3));
  relay at (1,0) 2850 (the moved relay); **y writes on NOC1 14200 ((8,3)) - ~5x rarer, not cured**; DRISC writes all
  on NOC1: 2 x 20K clean. -> only taking the DRISC writes into the multicast rectangle off NOC0 cures it in the probe
  (+28% in the op as one tile per bank). Next: the DRISC write VC (fwd_vc 3; MIMO_FL_FWD_VC in the op) as a free lever,
  the drop-one-class sweep, and MIMO_FL_Y_NOC1 (new: the NCRISC's y writes on NOC1 in dynamic-NoC mode, kernel
  SE_YRM_NOC + factory switch; built, installed at a queue pause).

### Wave 4 (00:00-00:45 UTC): the minimal traffic-only recipe

Paced probe, every class on its default NoC unless named (2 x 20K launches each):
- drop y / drop down-weight reads / drop helpers / drop every NOC1 class: all still wedge (each run, early).
- drop the DRISC writes: clean (also DRISC writes all on NOC1: clean).
- only x multicast + DRISC writes: clean. only x multicast + y + down reads: clean.
- x multicast + DRISC + y writes (to DRAM): wedged - run 1 with the exact (1,4) NOC0 CB0 signature.
- x multicast + DRISC + down-weight DRAM reads: wedged 2 / 2.
- x multicast + DRISC + helper sends (Tensix -> Tensix): clean 2 / 2.
=> the NOC0 wedge needs THREE things on NOC0: the x relay's multicast into the gate/up rectangle, the DRISC (DRAM-core)
weight writes into that same rectangle, and Tensix <-> DRAM traffic (the down cores' reads or y writes). Removing a
"third" class alone cannot fix it (the other one remains); a fix must separate the multicast from the DRISC writes.
The DRISC alt tiles can only write on NOC0 (their NOC1 NIU is the bank's NOC2AXI endpoint), so: the multicast on NOC1
(new MIMO_FL_XMC_NOC1: only the relays move, x reads to NOC0) with every DRISC write on NOC0 (MIMO_FL_DRISC_NOC1=0) -
'split'. The full-swap DRISC_XNOC cost +11 / +36% (it also moved the gate/up receivers); split perf queued.

### Wave 5 (00:45-01:40 UTC): separating the multicast from the DRISC writes; relay placement

- Probe: split (multicast + helpers on NOC1, x reads NOC0, every DRISC write NOC0) 3 x 20K clean; multicast on NOC1 with
  the default DRISC NoCs 2 x 20K clean; the DRISC write VC (0 / 1 / 2 instead of 3): every one wedges -> no VC fix.
- Real op, split (MIMO_FL_XMC_NOC1=1 + MIMO_FL_DRISC_NOC1=0): HANG at 190K - BOTH relays stuck on NOC1 CB0
  (se11_xmc:245): with the multicast on NOC1 the gate/up receivers' h writes + freed atomics (se5_recv, NOC1) are the
  writes OUT of the rectangle beside it (the MiMo trio). -> the multicast needs a NoC with no heavy traffic into or out
  of its rectangle: the full separation DRISC_XNOC=1 (multicast + helpers NOC1; receivers, x reads, all DRAM traffic on
  NOC0) + MIMO_FL_DRISC_NOC1=0 (every DRISC write NOC0) - the config that ran 400K amplified calls clean (wave 2).
- amp + DN_NOC0: both hangs had an EMPTY triage (NOC1 wedged): the alt DRISC tile (the bank's NOC1 endpoint) vs heavy
  NOC1 DRAM reads (the known DRISC-endpoint conflict) - not a fix.
- Perf, S 256 fused us (uniform / spread): split 1297 / 1822 (+5 / +36), multicast-only NOC1 1406 / 1940 (+14 / +45),
  every DRISC write NOC0 1233 / 1339 (neutral). The cost is the NOC1 multicast's lead-in: NOC1 goes north then west and
  starts at the rectangle's far corner, ~20 hops from the relay at (1,4). Marko's idea - move the relay: new
  MIMO_FL_RELAY_PRIM="6,9;11,8" (the free cells next to each rectangle's far corner): split + relay 1234 / 1400
  (+0.2 / +4.6), full separation + relay 1242 / 1404 (+0.9 / +4.9) vs full separation 1362 / 1820. Packed split + relay
  2409 vs 2218 (+8.6). Next: helpers next to the moved primaries (MIMO_FL_RELAY_HELP="7,9;11,9"), packed / S 640, then
  the hang validation of full separation + relay.
- Build note: tests load build/lib/_ttnncpp.so = the INSTALLED copy - a plain ninja build is not live; installs happen
  at campaign 'pause' entries (device idle a minute).

### Wave 6 (01:40-01:50 UTC): the candidate, codex review

Candidate (env): MIMO_FL_DRISC_XNOC=1 + MIMO_FL_DRISC_NOC1=0 + MIMO_FL_RELAY_PRIM="6,9;11,8" - the x multicast and
helper sends alone on NOC1 (relays next to each rectangle's far corner), every DRISC write, the gate/up receivers, x
reads and all Tensix <-> DRAM traffic on NOC0. Fused us vs default: S 256 uniform 1242 / 1231 (+0.9%), spread 1404 /
1338 (+4.9%), packed 2447-2466 / 2218 (+10-11%, flat alone +9%); S 640 uniform 2261 / 2250 (+0.5%), spread 2295 / 2155
(+6.5%). Helpers moved next to the primaries (RELAY_HELP): spread worse (1473). x reads back on NOC1 (XRD_NOC=1 +
XRD_DYN=1): hangs on the first fused call (config incompatibility).
Codex review: the candidate separates the classes (receivers + every DRISC sub-mode on NOC0; x forwarding / helper
writes target relays); open: the fused combine's NOC1 DRAM output writes next to the NOC1 multicast (needs the fused
validation, queued: block traced / eager, dvp, traced chunk); bugs in the new plan overrides (both cells can be the same,
strtoul parser can spin on bad input) - fix before committing; perf ideas: XHELP_ALL, x read batching (queued).

### Wave 7 (01:45-02:05 UTC): candidate probe + perf decomposition; validation launched

- Traffic-only probe of the candidate (multicast + helpers NOC1; x reads, receivers, every DRISC write NOC0):
  3 x 20K launches clean (paced base wedges within 1-5K).
- Perf decomposition, S 256 fused us (spread / packed), same-run default 1343 / 2215:
  every DRISC write NOC0 only 1339 / 2207 (neutral); DRISC_XNOC + relay with default DRISC 1403 / 2476; candidate (+
  DRISC NOC0) 1404 / 2447; split + relay (receivers stay NOC1 - but wedges on NOC1) 1400 / 2409 -> the cost is the
  multicast swap (multicast NOC1, x reads + receivers NOC0), not the DRISC NoC. x read batch 4: no change; XHELP_ALL
  (+ helpers moved): worse (1536-1606 / 2766-2880). Relays moved with the multicast still on NOC0: wedged at op 18
  (relay (11,8) NOC0 CB0) - the move only makes sense with the NOC1 multicast.
- Plan override fixes (codex): one shared strict parser, no duplicate cells (built, installed at the end).
- Validation queued for the candidate (MIMO_FL_DRISC_XNOC=1 MIMO_FL_DRISC_NOC1=0 MIMO_FL_RELAY_PRIM=6,9;11,8): op alone
  6 x 200K mesh calls (bit-exact markers), fused block traced (uniform, hot, real L20 decode / prefill routing) 4 x 40K
  + eager 2 x 30K, decode via prefill x20 samples, traced whole-model chunk S 256 4000 replays, fused block traced 1M;
  then default-config baselines.
- Perf follow-ups (day): keep the x reads on NOC1 beside the NOC1 multicast (XRD_DYN path hangs on the first fused
  call - fix); profile where the packed flat-alone +9% goes; per-model relay cells (RELAY_PRIM is GLM 12x10 specific)
  before any default change; K3 / other plans unaffected (env only).

### Wave 8 (02:00-04:10 UTC): candidate validation

Candidate = MIMO_FL_DRISC_XNOC=1 + MIMO_FL_DRISC_NOC1=0 + MIMO_FL_RELAY_PRIM="6,9;11,8":
- op alone (test_flat_stress, all 8 chips, bit-exact markers): 6 / 6 x 200K = 1.2M mesh calls (9.6M chip calls) clean.
  The candidate carries the amplified DRISC traffic (every DRISC write NOC0), which hung at ~20 / M with the multicast on
  NOC0 (~24 hangs expected over 1.2M).
- fused block traced (uniform, hot, real L20 decode / prefill routing): 160K clean; eager (uniform, L20 decode): 60K clean.
- decode via prefill, 20 BABILong samples (the workload that hung): 115K MoE calls clean, 20 / 20 answers (default 20 / 20).
- whole model traced chunk S 256 (start 2048), 4000 replays: 168K MoE calls clean; 244.5 ms / chunk vs 244.2 default
  (+0.1%): the op-level harness costs (+1..11% on synthetic routings) do not show in the model at S 256.
- running: fused block traced 1M calls; then whole-model traced S 640 perf (candidate vs default).
- fused block traced 1M calls (uniform, hot, L20 decode / prefill routing): clean.
- whole model traced S 640 (chunk 5120 at 51200), 500 replays: candidate 483.5 ms / chunk vs default 479.4 (+0.9%);
  per layer experts ~2.75-2.80 ms both. => whole-model cost: S 256 +0.1%, S 640 +0.9%.

### Wave 9 (04:40-06:45 UTC): final tallies

- Candidate: op alone 9 / 9 x 200K = 1.8M mesh calls clean (bit-exact); fused block traced S 640 300K clean (plus the
  S 256 1M / 160K traced, 60K eager, dvp 115K, traced whole-model chunk 168K above).
- Default config, op alone: 1 hang in 1.05M mesh calls (r0 at 50K; r1-r5 clean). Default fused block with the real
  decode / prefill routing (L20 + L3/10/30/44): 230K clean. Amplified fused block (every DRISC write NOC0, multicast
  NOC0): 120K clean; **amplified decode via prefill: HANG at 40.9K MoE calls - layer 11, its 974th call, device 2
  relay (1,4) NOC0 CB0** (the op-id map now names the layer in-model), where the candidate ran the same workload with
  the same DRISC traffic 115K clean.
- Summary: root cause = a Blackhole NoC hardware wedge (traffic-only reproduction): x relay multicast + DRISC writes into
  the multicast rectangle + Tensix <-> DRAM traffic on one NoC. Fix candidate (env): MIMO_FL_DRISC_XNOC=1
  MIMO_FL_DRISC_NOC1=0 MIMO_FL_RELAY_PRIM="6,9;11,8"; whole-model cost +0.1% (S 256) / +0.9% (S 640); op harness +1 /
  +5 / +11% (uniform / spread / packed). Not a default yet: per-model relay cells, the packed / spread op cost, review.
- Extension (07:08-09:00 UTC): candidate op alone 6 more x 200K -> 15 / 15 = 3.0M mesh calls clean; fused block traced
  S 640 600K more clean. Candidate totals: op alone 3.0M, fused traced 2.06M (S 256 1.16M + S 640 900K), fused eager
  60K, decode via prefill 115K, traced whole-model chunk 168K - zero hangs, op-alone outputs bit-exact.

### Wave 10 (09:00-10:00 UTC): x without a multicast - unicast chains (Marko's idea)

- Probe (traffic only, every class on its default NoC): x as unicast column chains (relay -> column heads, each core ->
  the next, unthrottled) 3 x 20K clean; + every DRISC write on NOC0 2 x 20K clean; the same-session multicast control
  wedged at once. -> no multicast, no wedge, without moving any NoC.
- Op: MIMO_FL_X_CHAIN=1 (new): the relay unicasts each block into ONE chain head per rectangle (se11_xmc XMC_CHAIN:
  writes, then after the acks the head's XARR += n), the gate/up cores forward down a tree with se5_recv's existing x
  chain (two successors: a trunk along the relay's row + every column; the receivers' NoC direction, NOC1 west /
  north; SE_X_CHAIN: non-heads report freed counts to their predecessor's SFREE / WORD, the head to the relay).
  Correct: worst per-token rel err 2.3e-3 on uniform / spread / packed (as before), 0 of 10 repeats differ; op alone
  20K smoke bit-exact. Perf, fused us (same session default 1231 / 1343 / 2215): uniform 1234 (neutral), spread 1570
  (+17%), packed 3081 (+39%). x ring depth (MIMO_FL_X_SLOTS_MAX) 16 / 24: 1607 / 3202, 1569 / 3075 - not the limit
  (12: deadlock - the row-pass compute needs >= Ht / KBLK = 16 slots; 32: does not fit). Suspect: the trunk cores
  forward every block twice from the receiver BRISC. Next: MIMO_FL_X_CHAIN=2, column chains (the relay feeds every
  column head, each core forwards once). (The flat-op unit test's Python builder requires an 11 x 10 grid - n/a here.)

### Wave 11 (10:00-10:40 UTC): optimizing the unicast x path (fused harness, S 256, us uniform / spread / packed)

Same-session default (multicast): 1230-1233 / 1341-1343 / 2209-2217.
- MIMO_FL_X_CHAIN=1 tree (trunk + columns; se5_recv BRISC forwards, 2 successors on the trunk): NOC1 1234 / 1569 / 3081,
  forwarding on NOC0 (MIMO_FL_X_CHAIN_NOC=0) 1233 / 1517 / 2656. Cut-through pieces (MIMO_FL_X_PIECES) / ring depth /
  relay pipelining (XMC_CHAIN_DEPTH: head writes with transaction ids, XARR after the acks, no per-batch barrier):
  no gain.
- =2 column chains (relay -> every column head): 1252 / 1765 / 3449 (relay egress bound).
- =3 snakes (down a column, one over, on down; each core forwards once; MIMO_FL_X_CHAIN_SNAKES): 1 snake NOC0
  1244 / 1562 / 2556; 2 snakes NOC0 1233 / 1482 / 2384, NOC1 1235 / 1434 / 2348, NOC1 + 2 pieces 1235 / 1417 / 2307;
  3 snakes 1235 / 1587 / 2911. Forward depth 10 / 14 (MIMO_FL_X_FWD_DEPTH): no gain -> the BRISC event loop is the limit.
- MIMO_FL_X_FWD_NC=1 (new se5_xfwd.cpp): a dedicated forwarder on the gate/up NCRISC (the receiver BRISC only consumes
  and publishes its consumed count in semaphore 2, its end in 3), on the other NoC than the receiver. Correct (rel
  err 2.3e-3, deterministic). 2 snakes, forwarder NOC0 + 2 pieces: **1234 / 1402 / 2241 (+0.2 / +4.5 / +1.4%)**;
  forwarder NOC0 1231 / 1409 / 2255; NOC1 1235-1236 / 1395-1397 / 2271-2276. S 640 (2 snakes, forwarder NOC0): fused
  2241 / 2181 / 5977 vs 2234 / 2163 / 5968 (+0.3 / +0.8 / +0.2%; the combine dominates), flat alone 1444 vs 1305 (+10%).
- Tuning the forwarder: 4 pieces 1231 / 1366 / 2216; 8 pieces 1232 / 1367 / 2218; 4 pieces + forward depth 3
  **1231 / 1360 / 2213** (base 1230-1232 / 1341-1343 / 2219-2221: 0.0 / +1.3 / -0.3%); depth 4 1232 / 1363 / 2212;
  depth 10 1232 / 1369 / 2217; forwarder on NOC1 (4 pieces) 1233 / 1365 / 2251; 3 snakes 1232 / 1565 / 2821 (relay).
  S 640, 4 pieces: fused 2235 / 2156 / 5962 vs 2234-2263 / 2163-2164 / 5962-5968 (neutral), flat alone 1346 / 1650 /
  4226 vs 1305 / 1614-1618 / 4196-4199 (+3 / +2 / +0.7%).
  => unicast candidate: MIMO_FL_X_CHAIN=3 MIMO_FL_X_CHAIN_SNAKES=2 MIMO_FL_X_FWD_NC=1 MIMO_FL_X_CHAIN_NOC=0
  MIMO_FL_X_PIECES=4 MIMO_FL_X_FWD_DEPTH=3 - no multicast, every other class on its default NoC, perf-neutral on the
  fused harness. Queued: whole-model traced chunk S 256 / 640 (vs default), K3 shape, then the hang validation (amp
  op alone, default op alone, fused block traced 1M).

### Wave 12 (10:45-11:35 UTC): unicast x on every model, two unicast bugs

Unicast config U = MIMO_FL_X_CHAIN=3 MIMO_FL_X_CHAIN_SNAKES=2 MIMO_FL_X_FWD_NC=1 MIMO_FL_X_CHAIN_NOC=0
MIMO_FL_X_PIECES=4 MIMO_FL_X_FWD_DEPTH=3, against the pre-fix default (linked NOC0 multicast). Fused harness
(scratch_flat_combine_rf.py, real weights), us uniform / spread / packed, rel err unchanged everywhere (2.2-2.4e-3):
| model, config | default | U |
| V4.1-Flash (ND 27, 10240 B rows) | 1166 / 1298 / 2702 | 1166 / 1300 / 2590 (-4% packed) |
| K3, production layout (K3_GU64: DRISC-only, HG NU 3, ND 28, NH 2, 16 acc on ports, depth 4) | 2288 / 2489 / 5381 | 2304 / 2493 / 5326 |
| GLM-Flash S 256, the model's (experts_ts.FLAT_ENV, 16 acc on ports, depth 4) | 1232 / 1344 / 2227 | 1231 / 1361 / 2222 |
| GLM-Flash S 640, the model's | 1509 / 1787 / 4819 | 1486 / 1783 / 4814 |
| GLM-Flash S 640, separate layout (12 acc + 7 ports, depth 3) | 1357 / 1702 / 4740 | 1368 / 1713 / 4737 |
(Harness configs: the earlier S 640 runs used FC_NACC 15 with the accumulators on ports - 2234 / 2163 / 5962, a
harness artifact; the model's 16 gives 1509. The separate layout is ~10% faster at S 640 uniform: the combine alone
1278 / 1436 / 2857 us is as long as the flat alone 1305 / 1609 / 4195 there. K3's earlier "allm" runs used the old
48 gate/up Tensix-reader layout: 2438 / 2684 / 5928 vs U 2439 / 2701 / 5928, neutral too.)
Whole model, GLM-Flash S 256 traced chunk (1000 replays): default 244.2 ms, U 244.4 ms.

**Bug 1 - whole-model hang (1 in ~13k calls, U, layer 39):** chip 3's flat op never ended; one gate/up core's
NCRISC forwarder (se5_xfwd) spinning, its receiver BRISC finished. se5_recv's x_cons follows compute at most one block
a loop pass, so the loop can end (all h sent) with it short of the call's blocks; END = x_cons + 1 was then short and
the forwarder's exit test fwd == (END - 1) x pieces never held (or could end it early, starving its successor). Fix:
se5_recv drains x_cons to num_v x x_per_v before publishing XCONS / END. Codex review agrees; no other early-exit or
cross-launch state issue (semaphores re-initialized per launch, relay words zeroed by se11_xmc); guard added:
MIMO_FL_X_CHAIN <= 3 (modes > 3 + the NCRISC forwarder would leave a relay word at zero). Whole model S 256 after the
fix: 42294 MoE calls clean.

**Bug 2 - GLM-5.3 (H 6144) deadlock with any chain mode at > 64 rows / expert:** the gate/up row-pass compute
(se3_compute, SE_GU_RP) waits for all Ht / KBLK = 24 x blocks of a sub-block before popping one, and the ring is 24.
DPRINT (MIMO_FL_SE_DBG=1, new debug probe: se5_recv / se5_xfwd counters): every core had 20 blocks, every forwarder
had forwarded them, compute consumed 0. The relay (se11_xmc XMC_CHAIN) only bumps the heads' XARR when it retires a
batch, and its waits (slot credit, next super-block, helper) did not retire: the last batch's 4 blocks were written
but never counted. Not reached on GLM-Flash / K3 / V4.1 (16 / 14 / 20 of 24 blocks). Fix: every relay wait loop
retires acknowledged batches (chain_retire(DEPTH), non-blocking). Probes before the fix, one variable each (all hung
at M 128 / 96, M 64 passed): DRISC-only, no NCRISC forwarder, column chains, 1 snake, 1 piece, X_SLOTS_MAX 24, E 8.
GLM-5.3 flat alone (test_flat_bench, glm53 E 16, dual DRISC + reader down columns), us / expert default -> U (first
fix, blocking retire): M 32 49.3 -> 49.0, 128 59.0 -> 59.1, 512 112.4 -> 123.5 (+10%), 1024 211.3 -> 238.1 (+13%);
hot 64x16 58.0 -> 58.0. Tuning (non-blocking retire, depth / pieces / snakes / NoC) queued.
GLM-5.3 fused flat + combine does not run at all (default config too): 12 KB rows leave the combine no ring slot in
L1 (dual / DRISC-only), and the reader-down-column layout (ND 24) hangs at the first call with the combine waiting
for rows (the reader tails' L1 hand-off) - a separate bring-up, not a unicast regression. Harness fix on the way: the
y writers come from the plan (plan["y_writers"], the factory's order: reader tails, then down cores by column width);
the harness took plan["rdn"]'s (reader, tail) pairs as cores.

### Wave 13 (11:35-11:55 UTC): one snake, unicast becomes the default

GLM-5.3 flat alone (E 16, us / expert M 128 / 512 / 1024), multicast 59.1 / 112.4 / 211.5. U (2 snakes) with the
non-blocking relay retire 59.0 / 123.7 / 238.7 - the retire was not the cost; forward depth 6 124.7 / 239.1, 2 pieces
124.4 / 238.7, 8 pieces 125.0 / 240.9, forwarder on NOC1 125.7 / 242.4, **1 snake 59.1 / 107.0 / 205.5** (-5 / -3% vs
multicast). Two snakes make the relay write every block twice; with GLM-5.3's ring full of one sub-block (24 / 24 x
blocks, row passes) the relay cannot run ahead and its egress shows. Same-session 1 vs 2 snakes, fused (us):
| | multicast | 2 snakes | 1 snake |
| GLM-Flash S 256 (model cfg) | 1232 / 1344 / 2227 | 1230 / 1361 / 2222 | 1230 / 1349 / 2224 |
| GLM-Flash S 640 (model cfg) | 1509 / 1787 / 4819 | 1505 / 1768 / 4814 | 1519 / 1793 / 4821 |
| K3 production | 2288 / 2489 / 5381 | 2304 / 2496 / 5316 | 2303 / 2508 / 5329 |
| V4.1-Flash | 1166 / 1298 / 2702 | 1167 / 1299 / 2591 | 1165 / 1285 / 2584 |
(flat alone, 1 snake vs multicast: GLM-Flash S 640 packed 4096 vs 4202, V4.1 packed 2503 vs 2615.)
Whole model GLM-Flash traced chunk: S 256 244.2 -> 244.4 ms (2 snakes, 42294 calls clean), S 640 479.4 -> 479.1 ms
(2 snakes + relay fix, 21294 calls clean).
**Default now (flat_routed_expert_program_factory.cpp XPath):** unicast snakes - 1 snake, 4 cut-through pieces (2 / 1
if a block does not split into 16 B aligned quarters), NCRISC forwarder on NOC0 with 3 pieces in flight, receiver
on NOC1 - whenever the plan allows (one primary per rectangle, full rectangles, no X1 / X2 / XFWD), else the
multicast; logged per program build. Env stays as probe overrides (MIMO_FL_X_CHAIN=0: the multicast).
Sign-off queued on the default: fused perf default vs MIMO_FL_X_CHAIN=0 (GLM-Flash S 256 / 640, K3, V4.1, GLM-5.3
flat), whole model S 256 x 3 / S 640 x 2, op stress amplified x 3 + default x 3 (200K each), block traced 1M.

### Wave 14 (11:55-13:10 UTC): a unicast data race, the command-buffer drain

op_stress_amp_unicast (test_flat_stress, 2 x 4, amplified: all DRISC jobs on NOC0; U with 2 snakes) FAILED its
bit-exact determinism check after 100K calls: chip 5's output differed from its gold once (x bfp8 / h bf16). The
multicast had run 800K amplified stress calls without a mismatch (all its failures were hangs).
Ruled out: the relay's source reuse (BH always counts transaction-id writes, so noc_async_writes_flushed covers them),
the forwarder's runtime args across arena moves (ring / words registered as arena / words addresses), semaphores 2 / 3
(no other user on the gate/up cores), the id semantics (BH ..._with_transaction_id_flushed = NIU_MST_REQS_OUTSTANDING_ID
== 0: acknowledged).
Cause (codex's first suspect, the write-side twin of FLAT_COMBINE_OVERLAP_WORKLOG.md 6.14): an id's outstanding count
only covers requests the NIU has taken; polled right after issuing, the last write can still sit in the write command
buffer and the id reads "done" before the data has landed - the forwarder / relay then bump the successor's XARR early
and it computes on stale x. Fix: wait for the write command buffer to drain (noc_cmd_buf_ready) before polling an id,
in se5_xfwd, se11_xmc's chain_retire and se5_recv's x / h chain forwarding.
After the fix (1 snake, the default): amplified 4 x 200K and default-config 1 x 200K bit-exact so far (1.0M calls;
the failing run had 100K). Older kernels poll ids the same way (se10_fwd weights, se_drisc_gu, se6_drecv h, se_yrm y)
and have run millions of bit-exact multicast calls - left alone, a possible latent issue.
Deferred (queue file deferred_unicast_signoff.json): whole model S 256 x 3 / S 640 x 2 and the fused block traced 1M
on the default.
