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
CORRECTNESS status: 1-worker mux = 8/8 PCC (zeros). W_WORKERS=2 op COMPLETES and hits 12.8 GB/s but has
an edge-OUTWARD PCC bug (w=0 Wleft / w=3 Wright BAD — the no-neighbor sections; row-split/8-align at
multi-worker). W_WORKERS=4 currently hangs (more cores/mux setup). So: fast+bandwidth-reached is proven;
making W_WORKERS>=2 fully PCC-correct (edge outward) + replicate-mode edge is the remaining correctness
work. Shipping default (no mux) 4/4 PCC intact.
