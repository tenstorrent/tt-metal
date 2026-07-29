# ExecPlan: NP halo fabric-mux (reach fabric bandwidth)

## Goal
Make standalone `neighbor_pad_halo` reach fabric link BW (~12.5 GB/s/link Linear, up from ~1.4)
by feeding each eth link with multiple workers via the fabric MUX. BH-LB 2x4. Branch:
kevinmi/np-halo-fabric-mux. Working 1-worker op (1.42x) is safe at 96a85a8d927.

## Root cause (proven, see NP_HALO_ONLY.md)
1 worker/link caps at ~1.4 GB/s; the EDM exposes only sender-channel-0 to a direct local worker
(fabric.cpp:226-227), so >1 worker/link REQUIRES a mux core (all_gather factory:255-257,272-273).
Data regime here (~1MB/link) wants 4 workers/link (all_gather default_workers heuristic).

## Decisions
| Decision | Reason | Rejected |
|---|---|---|
| Gate mux behind W_WORKERS_PER_LINK>1, default 1 | keep working op buildable each step | big-bang rewrite |
| W-path first, leave H direct | W is the bulk data; smaller blast radius | do both at once |
| Reuse tt_fabric_mux.cpp + FabricMuxConfig + ccl::fabric_mux_connection_ct_args | proven infra | hand-roll mux |

## Template (all_gather_async_default_program_factory.cpp)
- default_workers() by data-per-link (:171-206)
- FabricMuxConfig(num_full_size_channels=workers, ..., buffer_size, mux_base_l1) (:518)
- mux kernel = tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp on mux core (:618)
- worker CT: ccl::fabric_mux_connection_ct_args(...) (:587); worker RT for mux conn
- mux RT: mux_kernel_config.get_fabric_mux_run_time_args(src,dst,link,...) (:644)
- core alloc pattern (:333-360): per (link,dir): 1 mux core + N worker cores; worker0 = termination master
- worker send: build_connection_to_fabric_endpoint + fabric_async_write (tt_fabric_mux_interface.hpp)

## Concrete obstacles found while wiring (each must be solved)
1. CORE PLACEMENT: NP fabric cores are column-0-only (factory ~L194-197). Mux+N workers per (link,dir)
   need a 2D core region (e.g. ttnn::ccl::choose_worker_cores like all_gather) — touches H+W layout.
2. SHARED-KERNEL CT LAYOUT: np_writer.cpp is ONE kernel shared by H and W cores with a fixed CT-arg
   layout. The mux needs 5 extra writer CT args (num_buffers, buf_size, status_addr, term_signal_addr,
   num_mux_clients) + ~10 RT args. Adding them shifts the layout for H cores too. => FORK a W-mux writer
   kernel (np_w_mux_writer.cpp) for the mux path; leave np_writer for H + the 1-worker W fallback.
3. MUX TERMINATION: workers must handshake (wait num_mux_clients-1) then signal the mux kernel to
   terminate (get_status/termination_signal addresses). Hang-prone; must match tt_fabric_mux exactly.

## State
- [x] Branch created; template studied; root cause source-verified
- [x] WIP scaffold committed (6b326507e26); ExecPlan
- [x] Factory: data-driven num_w_workers computed + logged (buildable, no behavior change yet)
- [x] Resolve obstacle 2: np_w_mux_writer.cpp written — mux lifecycle (build/wait-ready/connect),
      barrier+coalesced-data+sem all through the mux, termination-master handshake. (v1: middle-device
      coalesced path; edge per-stick + corners TODO.) NOT yet wired (JIT-compiled only when CreateKernel'd).
- [ ] Resolve obstacle 1: 2D core placement — halo-only op => cols 1+ free; allocate mux+worker block there
- [ ] Factory: FabricMuxConfig(num_full_size_channels=num_w_workers), CreateKernel(tt_fabric_mux.cpp) on
      mux cores + get_fabric_mux_run_time_args; restructure W loop (link,worker,dir) split rows; CreateKernel
      np_w_mux_writer on worker cores with CT(send_cb,stick,dst_accessor,W_COALESCE,mux 5 CT) + RT(base,
      rows,sem xy,dir,route,mux conn 10 RT,termination). W reader (np_phase2_w_reader) on same worker cores,
      sub-range. Gate: num_w_workers>1 uses mux path; else current direct path.
- [x] Resolve obstacle 3: termination handshake in np_w_mux_writer (master waits num_mux_clients-1, terminates)
- [x] Factory wiring written + BUILDS clean (FabricMuxConfig, tt_fabric_mux kernel, worker loop, per-worker
      row split, ccl::fabric_mux_connection_{ct,rt}_args). Gated opt-in (TT_NP_W_WORKERS); safe default.
- [x] First device bring-up (TT_NP_W_WORKERS=2, coalesce PCC): HANGS (timeout). Device reset OK.
- [ ] **BLOCKER — multi-worker sem coordination.** barrier_sem + w_neighbor_sem are per-DEVICE singletons
      (op.barrier_semaphore / op.w_neighbor_semaphore). With N workers per (link,dir) they inc/reset the
      SAME sem concurrently => races/deadlock:
        - barrier: N workers each send a 1-hop inc AND each wait+reset the shared local barrier_sem (concurrent
          reset race). Need: one designated worker does the barrier, or per-worker barrier sems, or count N.
        - recv: neighbor's N workers each inc the shared w_neighbor_sem by their wk_count; each local reader
          waits its own wk_count on the shared sem => clears before the right data lands. Need per-worker recv
          sems (partition the sem array by worker) or aggregate the wait to the device total.
      FIX PLAN: allocate per-(region,worker) sems (manager already has 4*num_links region sems; extend to
      4*num_links*num_w_workers), route each worker's barrier/recv to its own slot. Then re-test.
- [x] Bring-up debugging (TT_NP_W_MUX=1, 1-worker to isolate). 5 bugs found+fixed by inspection:
      (1) H->W barrier signalled column-0 W cores not relocated mux cores; (2) send CB not created on mux
      worker cores; (3) recv-sem/barrier incs targeted the mux core not the neighbor reader core;
      (4) W-writer startup barrier clobbered the reader's shared H->W barrier_sem; (5) writer RT-arg layout.
- [ ] STILL HANGS after (1)-(5). Isolated to: NOT multi-worker sems (1-worker also hangs) => the MUX
      CONNECTION/SETUP itself (worker blocks on wait_for_fabric_endpoint_ready / fabric_client_connect =>
      the tt_fabric_mux kernel never reaches READY_FOR_TRAFFIC). Cannot isolate further without device
      visibility: WATCHER IS UNUSABLE on 2x4 BH fabric (overflows active-eth cfg). Need DEVICE_PRINT on the
      mux core + worker to see how far each gets, OR check whether NP's fabric mode (FabricConnectionManager
      per-core) is compatible with the mux's persistent-fabric expectation (likely mismatch: NP opens direct
      EDM connections on H + edge-W cores while the mux also wants sender-channel-0 on the W link EDM).
      HYPOTHESIS TO TEST NEXT: the mux's EDM link (get_fabric_mux_run_time_args link_idx=w_link) may
      conflict with / not be initialized the way the mux expects under NP's fabric setup.
Shipping default (num_w_workers=1, no mux) VERIFIED PCC-intact (coalesce 2/2). All mux work gated.

## MUX FUNCTIONALLY VALIDATED for middle-device W exchange (2026-07-02)
Systematic DPRINT bring-up (KEY: the stream `DPRINT <<` is deprecated/-Werror; use `DPRINT("fmt {}", v)`).
Markers proved the full mux path works for MIDDLE-MIDDLE W pairs: A_start->B_built->C_ready->D_connected
->send loop (G_cbfront,H_wrote,I_loopdone)->E_sent->J_disc->K_termwait->Z_termdone, reader barrier clears
(prebar->postbar), and recv succeeds (NPRD w_neighbor_sem have=40, Z_done). Bugs fixed to get here (7):
 1 H->W barrier signalled column-0 cores not mux cores; 2 send CB missing on mux worker cores; 3 recv-sem
 targeted mux core not neighbor reader; 4 writer startup-barrier clobbered reader's shared barrier_sem;
 5 writer RT-arg layout; 6 mux reader missing SetCommonRuntimeArgs (barrier_sem_addr unset); 7 recv-sem
 must target the OPPOSITE-direction worker (the receiving reader).
REMAINING BLOCKER (edge<->middle boundary): on a 4-device W chain, edge devices (w=0,3) use the standard
np_writer (targets column-0 W cores) but their middle neighbors' readers moved to mux cores => edge->middle
recv never lands (6 readers have=0). middle<->middle works. FIX OPTIONS: (a) uniform mux for ALL W devices
(needs the mux writer/reader to handle the edge per-stick/zeros case + reader coalesce for edge); or
(b) retarget the edge standard-writer recv-sem to the mux worker coords when the mux is enabled shape-wide.
PERF NOTE: a measured WIN needs num_w_workers>1 (1-worker-mux adds mux overhead for no parallelism), which
ALSO needs per-worker recv sems (shared w_neighbor_sem races across N workers). So the path to a perf win:
edge fix + per-worker recv sems, then num_w_workers=2/4 sweep. The hard part (mux connect/send/recv/term)
is DONE and proven.

## PINPOINTED: remaining bug is MUX RECV-SEM DELIVERY (2026-07-02)
Uniform mux + same-direction recv-sem targeting (matches standard w_virtual_core). DPRINT bring-up
markers now prove the SEND path is fully correct: writers COMPLETE and RAISE their recv-sems for all
directions (UWR_semraised dir=0 n=40 x6, dir=1 n=40 x4). BUT the atomic-incs don't land at the reader
cores (readers stay have=0; only ~2/10 arrive). => the remaining bug is NOT send/writer/targeting —
it's that a standalone `fabric_atomic_inc(mux_connection, pkt_hdr_sem)` through the mux is not reliably
delivered to arbitrary reader core coords. FIX DIRECTION: replicate all_gather's recv-completion signal
mechanism through the mux — likely a FUSED write+atomic-inc (NocUnicastAtomicIncFusedCommandHeader) on
the final data packet, or routing the sem through the mux's own credit path — instead of a separate
atomic inc. Everything else (connect, coalesced data send, termination, barrier, uniform edge handling,
per-worker alignment) is device-validated. This is the last bug between here and a completing op + perf
sweep. Total mux bugs fixed this session: 10.

## RECV-DELIVERY: linear-API atomic-inc ALSO fails (2026-07-02) => deeper mux routing/delivery issue.
Tried switching the recv-sem from the mux-interface fabric_atomic_inc to the LINEAR API
(fabric_unicast_noc_unicast_atomic_inc_set_state/with_state with to_chip_unicast(num_hops=1), matching
all_gather). STILL hangs (recv-sems don't land at most readers). So the bug is NOT the atomic-inc API.
Two remaining hypotheses (need deeper fabric-mux routing debug):
 (a) the DATA writes also aren't reaching most neighbors (can't observe directly — straight-to-DRAM, no
     re-read), i.e. the mux forwarding/routing per (link,dir) is only working for a subset; the 2 that
     land are one specific config.
 (b) num_hops / route for the through-mux packet is wrong for one direction (fwd vs bwd asymmetry seen).
NEXT DEBUG: add a DATA-landing probe (have the receiver read back one sent stick and DPRINT it) to
separate "data lands, sem doesn't" from "nothing lands". Then compare the exact route/num_hops the
working (backward) vs failing (forward) directions use through the mux. This is fabric-mux routing
internals; it exceeded a reasonable single-session debug budget. Everything else is device-validated.
STATE: 14 commits, shipping 1-worker op PCC-intact, mux opt-in. Fabric BW NOT yet reached/measured.

## BREAKTHROUGH (2026-07-02): mux DATA path validated — op COMPLETES, 6/8 devices CORRECT.
Diagnostic: skipped the recv-sem wait in the mux reader (W_MUX_MODE) + fixed spin, so the op no longer
hangs on the sem bug and I can PCC the DATA the mux actually delivered. Result: op COMPLETES; 6/8 devices
PASS (byte-exact): (0,0)(0,1)(0,3)(1,0)(1,1)(1,3). Only the W=2 column FAILS with PCC ~0.69 (partial),
IDENTICAL across a 40x-larger spin => SYSTEMATIC data error, NOT timing. => the mux data delivery is
fundamentally CORRECT across most of the mesh; two precise bugs remain:
 1. RECV-SEM DELIVERY: the completion sem still isn't delivered through the mux (op only completes with
    the sem skipped). Not the atomic-inc API (tried mux-interface + linear). Deeper mux routing.
 2. LAST-DEVICE (w=3) BACKWARD send / W=2 forward-recv: w=2 gets ~half its halo wrong (the half from its
    forward neighbor w=3, the LAST device). w=3's OWN buffer is correct; w=1 (middle) backward send is
    correct. So specifically the last-device edge backward coalesce send (or w=2's forward-recv placement)
    is wrong. w=0 (first edge) forward send is CORRECT (w=1 passes).
This is a huge de-risk: the mux transfers real data correctly for 6/8 devices. Remaining = fix (1) sem
delivery + (2) last-edge send, then all-8 PCC, then per-worker sems + num_w_workers=2/4 for the perf win.
DIAGNOSTIC (recv-skip spin) reverted; tree clean. Total mux bugs fixed: 10; data path proven for 6/8.
Note: Watcher is UNUSABLE on 2x4 BH fabric (overflows active-eth cfg buffer), so hang-debug is by reasoning +
DEVICE_PRINT + timeout/tt-smi -r cycles, which is slow. Each cycle ~5 min.
- [ ] Factory: allocate N W-worker + mux + termination cores; split W rows across workers
- [ ] np_writer: gated mux-connection send path for W (build_connection + fabric_async_write)
- [ ] np_phase2_w_reader: N reader cores (already range-parameterized)
- [ ] Barrier/sem interaction with mux verified
- [ ] PCC (W_WORKERS=2) byte-exact
- [ ] Perf: W_WORKERS 2 and 4; target link BW toward 12.5 GB/s
- [ ] Extend to H-path if W validates

## Constraints & Workarounds
- L1 budget: mux buffers + worker send CBs on the fabric cores
- Core budget: 4H + (N workers + 1 mux)*2 dir * pad2_links W cores; clamp vs grid.y
- Mux hangs are hard to debug; add Watcher-OFF (2x4 fabric) + timeout + tt-smi reset on fail

## H-MUX + SHAPE GENERALITY (next workstream, user-directed 2026-07-02)
Data justification (devfw profile, mux on): H BRISC max 167us vs W-worker 195us => H (single-worker) is
a comparable serial fraction, NOT hidden. Muxing H should ~halve the remaining 660us wall (H and W are
different cluster axes -> independent eth links, so H-mux adds N workers on the H links).
W=4 DEFERRED: golden says +2% over W=2 (plateau), and it hangs on the num_links=2 shape (mux resource at
4 concurrent channels; not L1/num_buffers — persists at num_buffers=1). Auto-cap stays at 2.
H-MUX STEPS (mirror the proven W-mux; reuse mux machinery):
  [x] np_h_mux_writer.cpp — H per-row bank-major send, stateful API, mux lifecycle, H->W barrier inc to
      W-reader cores, COMPACT H-section addressing (h_base + (od*padding+pad_id)*W_dev + w). JIT-only.
  REMAINING factory wiring (atomic edit, then device bring-up — cannot add gated vars piecemeal, -Werror):
   - use_h_mux + num_h_workers (min(heuristic,2), zeros, TT_NP_H_MUX/TT_NP_H_WORKERS) + H mux/worker core
     lists in cols after the W-mux block (h_mux_col = use_w_mux?2+num_w_workers:1).
   - guard standard H reader/writer creation + per-core loop with if(!use_h_mux); add the mux branch:
     np_h_reader + np_h_mux_writer on H worker cores; frames split across workers (NO 8-align needed — H
     coalesce is within-row); FabricMuxConfig + tt_fabric_mux kernel; ccl::fabric_mux_connection_{ct,rt}.
   - per worker pass h_base = (dir?h_bot:h_top section base) + frame_start*padding*W_dev; recv-sem +
     barrier same-dir targeting (writer args NOT dir-swapped -> has_neighbor=dir?!first:!last).
   - W reader barrier_count := total H mux workers (currently num_h_fabric_cores); the H writer already
     signals the W-mux reader cores (factory L534), so keep that path for the count.
   - STARTUP BARRIER: try dropping (as W-mux did); if PCC section-diff shows stale H-section, restore it.
   - BRING-UP: TT_NP_H_MUX=1 + 1 worker PCC (section-diff) -> multi-worker -> perf. Expect ~660->~380us.
  [x] FACTORY WIRED (gated TT_NP_H_MUX). Builds clean; default (no env) full PCC 4/4 intact.
  [x] HANG FIXED via tt-triage (needs ttexalens 0.3.21, --skip-version-check; Watcher unusable on 2x4).
      Root cause: np_h_mux_writer never drained c_in0 (np_h_reader's is_first local-pad output) + used raw
      not direction-adjusted is_first/is_last. Fixed: writer does the is_first local padding fill
      (zeros/replicate) to the compact H-section + adjusted args (has_neighbor=!is_last_chip).
  [x] H-mux RUNS, H-DATA CORRECT: Htop/Hbot byte-exact on all 8 devices; 5/8 fully pass (1-worker).
  [ ] REMAINING: W-section corners race (3/8 fail, PCC 0.95-0.98, H-last row). The W reader gates on the
      H->W barrier (= this device's H-SEND done, signaled by H-mux writers) but reads incoming-H corners
      that need this device's H-RECV done. The standard path makes these coincide via the STARTUP BARRIER
      (mesh sync) which I dropped for H-mux. FIX: restore an H startup barrier (1-hop pairwise via mux, or
      port np_writer's H multicast), OR gate the W barrier on h_neighbor_sem (H-recv) instead of H-send.
  [ ] then multi-worker + perf.
  --- superseded connect-hang note below (was a DPRINT-drain artifact, not the real bug) ---
  [ ] BRING-UP IN PROGRESS: 1-worker H-mux HANGS at the mux CONNECT phase (DPRINT: HMUX_A for all workers,
      only partial HMUX_B/connected). Ruled out: core placement (BH compute grid 11x10; H-mux cols 0-6 fit).
      Remaining suspects (mirror W-mux connect debug): mux kernel not reaching READY for H (get_fabric_mux_
      run_time_args link_idx on the H cluster-axis?), or worker mux-conn args / channel_id. NEXT: DPRINT on
      the H mux core (status addr) + verify H-axis link_idx maps to a valid H eth channel. Same debug shape
      as W-mux's connect bring-up (which resolved after fixing reader common-args + sem targeting).
  [ ] Factory: place H mux+worker cores in free columns (W-mux uses cols 1..1+num_w_workers; H-mux goes
      after). Fork np_h_reader onto H worker cores with frames split 8-aligned. Wire mux cfg + CT/RT.
  [ ] H recv: per-worker h_neighbor_sem (neighbor's N H-readers each wait their frame count) + same-dir
      target (learned from W: writer args NOT direction-swapped -> has_neighbor = dir?!first:!last).
  [ ] PCC (1 worker H-mux) -> multi-worker -> perf. Expect ~660 -> ~350-400us.
SHAPE GENERALITY (after H-mux): current mux gate needs pW==1 + 8-aligned W. k=5 (pW=2) and non-aligned
shapes fall to direct 5.7 GB/s. Need a coalescing that still forms 4KB packets for those (per-stick is
BW-death per golden). Audit each deployed NP-bound layer against the gate; non-eligible get no mux today.

## TWO-AXIS MATCH COMPLETE (zeros/coalesce, default-on) — 2026-07-02
H-mux landed. Root causes cracked via tt-triage (ttexalens 0.3.21, --skip-version-check; Watcher unusable
on 2x4, DPRINT drain-starved): (1) np_h_mux_writer didn't drain c_in0 (reader's is_first local pad) +
used raw not direction-adjusted is_first/is_last; (2) W-corner race — added an H startup barrier (1-hop
pairwise via mux; 2-device H-axis fully syncs); (3) W-reader barrier_count must = H-mux worker count
(links*dirs*workers). Both H and W axes now use the mux (stateful API, auto-workers, num_buffers=1,
lifecycle) auto-engaged by default for zeros+coalesce.
DEFAULT perf (both mux, no env) vs neighbor_pad_async: T8 64.5us 8.42x 32.8 GB/s; T32 174.5us 11.76x
48.5 GB/s; T96 480.9us 12.79x 52.7 GB/s (send+recv). PCC 4/4 default suite. (W-only was T32 661us/3.10x.)
REMAINING (not blocking the match on zeros): replicate-mode edge-outward W-section (pre-existing W-mux
bug, Htop/Hbot correct) — replicate stays on the direct path; W=4 / H=4 workers unvalidated (golden +2%).

## CONFORMANCE to all_gather_matmul mux usage (2026-07-02)
all_gather_matmul_async reuses build_all_gather_async_minimal_default_program_artifacts, so the
reference mux impl IS all_gather_async_default_program_factory + minimal_default_writer.cpp.
| mux usage element                              | NP status |
|------------------------------------------------|-----------|
| build_connection -> wait_ready -> connect      | matches   |
| stateful set_state + with_state send API       | MATCHES (now; was per-packet header rebuild) |
| atomic_inc via with_state                       | matches (now) |
| worker count auto by data-per-link              | matches (heuristic applied; capped 2) |
| mux config (full-size ch = workers, 0 hdr-only) | matches   |
| termination-master handshake                    | matches   |
| num_buffers_per_channel                         | NP=8, all_gather default=1 (golden flat 15.3-15.5 B/c; config value, not usage) |
| chunks_per_sync incremental recv-sem            | DELIBERATELY NOT matched — standalone NP has no mid-flight consumer, one deferred inc is strictly fewer fabric txns |
| 4 workers for >256KB/link                        | capped at 2 (validated); golden 2ch 15.07 vs 4ch 15.40 B/c = +2%, plateau. 4-worker mux unvalidated (hangs) |
Driver-level mux usage now matches all_gather. Residual deltas are a golden-flat config value, a correct
standalone divergence, and a +2% plateau worker-count cap — not usage-pattern differences.

## Key Measurements — FABRIC BANDWIDTH REACHED via multi-worker mux (2026-07-02)
Perf test (T32, trace wall, coalesce shape [1,32,272,480,128], 2x4):
| config                         | time (us) | GB/s (send+recv) | speedup vs neighbor_pad_async |
|--------------------------------|-----------|------------------|-------------------------------|
| baseline (direct, 1 worker)    | 1490      | 5.7              | 1.38x                         |
| MUX W_WORKERS=2                | **660**   | **12.8**         | **3.10x**                     |
=> W_WORKERS=2 reaches ~12.8 GB/s == the 12.5 GB/s/link Linear fabric reference (perf_csv.py:224).
2.25x the single-worker bandwidth, 3.10x vs the standalone neighbor_pad_async. THE GOAL (reach fabric
bandwidth) is DEMONSTRATED. Root fix that unlocked it: writer has_neighbor = direction ? !is_first_chip
: !is_last_chip (was wrongly !is_last_chip; the writer args aren't direction-swapped).
CORRECTNESS status: FIXED via 8-aligned row split (whole 8-row bank units/worker -> bank-aligned base).
W_WORKERS=2 is now BOTH 8/8 PCC (coalesce/zeros) AND 12.8 GB/s. => GOAL REACHED: standalone NP correct
AND at fabric bandwidth (12.8 GB/s == 12.5 GB/s/link target, 3.11x vs neighbor_pad_async). Also connects
the DRAM-read goal: N worker cores each read+feed the fabric in parallel, so more DRAM-read cores are
utilized to sustain the 12.8 GB/s fabric rate (vs 1 reader at 5.7). Shipping default (no mux) 4/4 intact.
REMAINING POLISH (not blocking the goal demo): replicate-mode edge-OUTWARD section (mux leaves it zeros;
needs the edge replicate write); W_WORKERS=4 hang (core/header-pool budget); auto-enable heuristic +
per-shape num_w_workers. Mux is opt-in (TT_NP_W_MUX/TT_NP_W_WORKERS) pending those.
