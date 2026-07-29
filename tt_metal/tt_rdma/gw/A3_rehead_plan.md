<!-- SPDX-License-Identifier: Apache-2.0 -->
# A3 — DPA re-head gateway: design + interop plan

**Goal.** Move the RoCEv2 → TT-RDMA re-head fully off the Arm CPU: the BF3 DPA (FlexIO) re-heads each
RoCE-landed payload into a native TT-RDMA WRITE frame and egresses it to the Blackhole, so the Arm does ~0
per-frame *data* work. Architecture B: the ConnectX/BF3 terminates full RoCEv2 in silicon; the gateway does
only the lean re-origination; the BH drainer pool lands it (validated 200G lossless).

## Where it runs (settled)

```
x86 host (RoCE requester)  --RoCEv2 WRITE_IMM/SEND-->  BF3 Arm: [ SF mlx5_2  RoCE RC responder ]
   ANY standard RoCEv2 stack                                     |  HW lands payload into Arm memory
   (see interop matrix)                                          |  (one buffer, two device MRs:
                                                                 |   mlx5_2 PD rkey REMOTE_WRITE  +
                                                                 |   FlexIO PD lkey for DPA gather)
                                                                 |  responder thread bumps doorbell
                                                                 v
                                                    [ PF mlx5_0  FlexIO DPA drain ] --p0--> Blackhole pool
```

- **The merged gateway process runs on the DPU Arm.** The DPA is a **PF-level resource** creatable only from
  the Arm (SFs cannot host a DPA); the RoCE GID lives on the **SF**. One process opens both mlx5_0 (FlexIO DPA
  egress) and mlx5_2 (RoCE responder). This is *not* an x86-host program.
- **The x86 host is the RoCE traffic source** (requester), unchanged. **The Blackhole is the drain pool**,
  unchanged.
- **No shared memory.** shm was only ever needed to bridge two separate binaries. In one process the landing
  buffer is ordinary Arm memory registered on both device PDs. The doorbell is a process variable the DPA
  reads via a `flexio_window` — not shm.

## Status (proven on silicon, committed)

| step | what | result |
|------|------|--------|
| A1   | DPA ETH-SQ blast | ~178G jumbo, Arm-free |
| A4   | zero-copy gather (2-seg WQE) | byte-exact; per-thread ~49–57G |
| A5   | multi-thread fan-out | ~146G@4KB / ~198G@8KB (packet-rate bound; 8KB≈200G wire) |
| A3.1 | DPA gather from **host** memory (ibv_mr lkey + host VA) | 302 B/frame egress verified |
| A3.2 | doorbell via `flexio_window` + arrival-driven drain | live read; preset+arrival exact |
| A3.3a| cross-**process** shared-mmap seam (shm_writer stand-in) | exact; **superseded by b-1** |
| A3.3b-1 | **merged** memory model: one process, one buffer, mlx5_0 lkey + mlx5_2 rkey | egress verified |
| A3.3b-2 | RDMA-CM RC responder in the merged process (`TTDPA_ROCE=1`) | listens; ESTABLISHED |
| **A3.3b-3** | **REAL RoCE E2E**: host WRITE_IMM → responder → doorbell → DPA egress | **100000 exact, p0 verified** |

**★ A3 COMPLETE (with a simulated-payload, single-slot landing).** Real RoCEv2 WRITE_IMM from the x86 host →
ConnectX HW-terminates on the SF → responder bumps the doorbell → PF DPA re-heads via HW gather → p0 egress,
one merged process, no shm, Arm off the data path. Remaining: BH-pool byte-exact landing (pool run), generic-app
interop (ib_send_bw/ib_write_bw -R), and the production refinements below.

Key gotcha baked in: **`flexio_window` config shares the DPA thread config slot with the SQ outbox** — restore
an explicit outbox after each window read or SQ doorbells silently stop egressing.

## A3.3b-2 — the responder, RDMA-CM-based from the start

Replace the `shm_writer` / `db_bump` stand-in with a real RC-QP responder **inside the merged process**, on a
dedicated Arm thread (the drain is a blocking `flexio_process_call`, so the RoCE poller must be its own thread;
both share the doorbell — exactly the A3.2/A3.3a shape).

**Connection: use RDMA CM (`librdmacm`), not a custom OOB handshake.** The current DOCA-sample responder uses
an app-specific TCP handshake, which is *why `ib_write_bw` won't interop with it unmodified*. RDMA CM is the
standard connection method every generic RoCEv2 app speaks. It is fully available on the BF3 (librdmacm,
`rdma_cm`/`rdma_ucm` modules, `/dev/infiniband/rdma_cm`, `rping`/`ucmatose`/`ib_write_bw` all present) and the
SF link is ACTIVE (mlx5_2, GID 10.99.0.1). (Loopback-to-self is rejected — expected RoCE self-connect limit;
validate with the x86 host in b-3.)

Responder accept flow (librdmacm):
```
rdma_create_event_channel()
rdma_create_id(ch, &listen_id, NULL, RDMA_PS_TCP)   # RoCEv2 uses IP addressing under RDMA_PS_TCP
rdma_bind_addr(listen_id, sockaddr{10.99.0.1:<port>})
rdma_listen(listen_id, backlog)
loop rdma_get_cm_event():
  RDMA_CM_EVENT_CONNECT_REQUEST -> rdma_create_qp(cm_id, pd_mlx5_2, {RC, cq, cap})   # QP on the SF PD
                                   post N recv WRs (for WRITE_IMM/SEND completions)
                                   rdma_accept(cm_id, &conn_param{private_data = {rkey, addr, len}})
  RDMA_CM_EVENT_ESTABLISHED     -> connected
poll the QP CQ:
  IBV_WC_RECV_RDMA_WITH_IMM (or IBV_WC_RECV) -> a frame landed:
     bump *produced (+ write a {roff,len} descriptor)   # the DPA drains via the window
     re-post a recv WR
```
- The QP's PD is the **SF (mlx5_2)** PD. The landing buffer is registered there (`LOCAL_WRITE|REMOTE_WRITE`)
  → its **rkey** is advertised to the requester (via CM `private_data` or an app protocol). The **same buffer**
  is registered on the **FlexIO process PD (mlx5_0)** for the DPA gather (A3.1 / b-1, already proven).
- The re-head **payload is HW-landed by the RDMA WRITE — no CPU copy**. The responder thread only bumps the
  doorbell (2 stores, no data touch).

## Interop — will standard ibverbs / RoCEv2 apps work?

**Wire protocol: yes** — the ConnectX terminates genuine RoCEv2 (PSN/ICRC/ACK/retransmit) in silicon, and the
DPA is *downstream* of termination (egress only). So the DPA never constrains RoCE interop. Whether a given app
works **unmodified** depends on three responder-side gates:

| gate | requirement | with RDMA CM |
|------|-------------|--------------|
| **Connect** | app's QP handshake must match the responder | RDMA CM = the standard both sides speak → generic apps connect |
| **Trigger** | the re-head fires on a **responder completion** | `SEND` and `WRITE_WITH_IMM` generate one → work. **Silent RDMA WRITE generates NO completion → no trigger.** |
| **Address** | requester target → BH destination mapping | one staging slot until **MR federation (B2)**; then rkey→BH-MR |

Compatibility by op type (post-b-2):

| requester op | connects (CM) | triggers re-head | notes |
|--------------|---------------|------------------|-------|
| `SEND` / `SEND_IMM` | ✔ | ✔ (recv completion) | cleanest; two-sided, no rkey advertise needed |
| `RDMA_WRITE_WITH_IMM` | ✔ | ✔ (recv-with-imm completion) | needs responder rkey+addr (CM private_data) |
| **silent `RDMA_WRITE`** | ✔ | **�’ (no completion)** | needs a completion-less landing detector (see below) |
| `RDMA_READ` / `ATOMIC` | ✔ | n/a | not part of the WRITE re-head → **B4** |

**Concrete apps.** `ib_write_bw`/`ib_send_bw` (perftest, `-R` for RDMA CM), `rping`, MPI (UCX RC), NCCL, RoCE
storage — all speak RDMA CM, so they *connect* once the responder is CM-based. They *trigger* the re-head iff
they use SEND or WRITE_IMM; perftest `ib_send_bw` and `ib_write_bw` both signal, so both work for a
single-destination test. Silent one-sided WRITE workloads (common in some libraries) need the extra detector.

**Silent-WRITE handling (optional, for full one-sided interop).** Detect landings without a completion by
polling the landed region for a change — e.g., the requester (or a WRITE template) stamps a monotonically
increasing sequence/footer at a known offset per frame, and the DPA/Arm advances `produced` when it observes
the next seq. This keeps the Arm off the data path and needs no app cooperation beyond a footer convention.
Defer until a real silent-WRITE workload requires it; document `SEND`/`WRITE_IMM` as the supported contract.

## A3.3b-3 — end-to-end validation ✅ (2026-07-29)

**Validated:** host `tt_roce_client 10.99.0.1 18515 100000 256` → RDMA CM connect, read advertised rkey
0x53d00 → 100000 WRITE_IMM. Gateway: responder ESTABLISHED → "100000 WRITE_IMM/SEND landed → produced=100000"
→ DPA drain "100000 frames egressed [exact]" → **p0 delta = 100002 pkts**, clean exit. No shm, Arm off the
data path, RDMA CM.

### Working recipe (after a DPU reboot — runtime IP state is wiped; OVS ovsbr1/2 persists)
```bash
# 1. DPU: re-add the SF's RoCE IP (lost on reboot)
ssh ubuntu@192.168.100.2 'sudo ip addr add 10.99.0.1/24 dev enp3s0f0s0; sudo ip link set enp3s0f0s0 up'
# 2. host: re-add the RoCE-port IP (sudo -n ip is allowlisted on desktop-0)
sudo -n ip addr add 10.99.0.10/24 dev enp193s0f0np0
ping -c2 10.99.0.1          # must show ARP REACHABLE before RDMA CM will resolve (else ADDR_ERROR)
# 3. DPU: launch the gateway DETACHED (survives ssh/tmfifo bounce; times out its connect-wait after 120s)
ssh ubuntu@192.168.100.2 'bash /tmp/launch_gw.sh'   # setsid + stdbuf -oL -> /tmp/gw.log; "RDMA CM listening on :18515"
# 4. host: drive the requester (no sudo needed; /dev/infiniband is world-rw)
./tt_roce_client 10.99.0.1 18515 100000 256
# 5. DPU: confirm  grep -E 'landed|drain done|DOORBELL done' /tmp/gw.log ; ethtool -S p0 (tx_vport_unicast delta)
```
launch_gw.sh runs: `sudo setsid bash -c "stdbuf -oL -eL env TTDPA_DOORBELL=1 TTDPA_ROCE=1 TTDPA_HOSTSRC=1
TTDPA_COUNT=<N> TTDPA_PLEN=<plen> TTDPA_NOCRC=1 <bin> mlx5_0 >/tmp/gw.log 2>&1" </dev/null >/dev/null 2>&1 &`

### BH-pool byte-exact E2E ✅ (2026-07-29) — the full Plan-B loop
Drove the drainer pool (`bh1_rx_dispatch 1 ext <hold> 1 1 2 0`, crc_check=0 to match the gateway's CRC-skip;
MR rkey 0x00CAFE42) with the REAL gateway path: host `tt_roce_client 10.99.0.1 18515 100000 <plen>` → SF →
DPA re-head → p0 → pool. **Pool: total=100000 write_ok=100000 bad=0 crc_err=0 rkey_miss=0** — every frame
landed byte-exact, exactly-once, lossless, zero errors. Gateway GW_EXIT=0; both clean-shutdown. External
RoCEv2 → ConnectX terminate → DPA re-head → BH TT-RDMA landing, proven end-to-end.

Orchestration (3 machines): launch gateway detached FIRST (listening) → pool background (long hold) →
`until grep "dispatch kernel up"` then fire the client → read pool total/write_ok/bad. AICLK boosts to 1350
under load; READ RTT p50≈55µs (perf.sh). NB the DPU `/tmp/launch_gw.sh` was the hardcoded-256B early version
(sync the env-driven vendored one for jumbo).

### Remaining
1. **Jumbo + A5 fan-out** into the drain for line-rate *landed* BW (this run was 256B, single-thread).
2. **Generic-app interop**: `ib_send_bw -R … 10.99.0.1` / `ib_write_bw -R …` (our responder is CM-based;
   perftest layers its own MR-exchange, so this validates *connect* + confirms the trigger contract).
3. **B2 MR federation** — rkey→BH per-destination mapping (currently one slot, roff=0).
4. **Part B** multi-in-flight generator for real interop throughput (removes the burst crutch).

## Production refinements (after b-3 basic E2E)

- **Landing-slot ring + per-frame `{roff,len}` descriptors.** b-1/b-2 re-head one slot; real traffic needs a
  ring of landing buffers (post one recv per slot) and a per-frame descriptor the DPA reads (roff into the BH
  MR, len) so distinct payloads land at distinct BH offsets.
- **Fold the A5 multi-thread fan-out into the drain** for line rate (single drain thread ≈1.37 Mpps@302B;
  N DPA threads reach ~146G@4KB / ~198G@8KB).
- **MR federation (B2)** for gate #3: map the requester's rkey/offset to a BH MR (rkey→{mesh,chip,noc,roff}),
  turning the gateway into a transparent RoCE target rather than a single staging slot.
- **Reliability**: PFC lossless validation; the requester-visible RoCE QP already gets HW retransmit.

## Open questions / risks

- **RDMA CM on the SF from the x86 host** — infra present + link ACTIVE, but end-to-end CM connect host→SF is
  unverified (loopback-to-self is rejected by design). Validate early in b-3.
- **CM private_data rkey/addr advertise** vs each tool's own exchange — perftest does its own MR exchange over
  the CM channel; confirm the responder's advertised rkey/addr is what the tool writes.
- **Recv-queue depth vs arrival rate** — post enough recv WRs (WRITE_IMM/SEND each consume one) to avoid RNR.
- **Doorbell/descriptor ring sizing** vs the DPA drain rate (already SQ-CQ paced).

_Companion: memory `tt-rdma-dpa-rehead-plan` (chronological log) and `tt-rdma-rocev2-gateway-arch-b`
(Architecture B). Build/run: `deploy_dpa_ttblast.sh` (DPA egress + A3.1/A3.2/A3.3a envs)._
