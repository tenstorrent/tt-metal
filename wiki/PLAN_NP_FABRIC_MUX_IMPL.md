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
- [ ] BUILD (wire factory) -> PCC (W_WORKERS=2) -> debug hangs (timeout+tt-smi -r) -> perf
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

## Key Measurements
(to fill) baseline 1-worker: T8 379us / 5.7 GB/s / 1.4 GB/s/link
