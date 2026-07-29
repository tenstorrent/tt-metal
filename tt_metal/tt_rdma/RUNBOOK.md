# TT-RDMA RUNBOOK — drainer pool (Phase 3.1) + RoCEv2 gateway (Arch-B)

Operational build/run/eval steps for the **line-rate RX drainer pool** and the **BF3 RoCEv2→TT-RDMA gateway
bridge**. Complements the older harnesses (`bh0/regression.sh` correctness, `bh0/perf.sh` latency/bw,
`CLAUDE.md` firmware flash). All commands run on the bench host (desktop-0) unless marked **[DPU]**.

> `TTH=/home/alex/mpi-shfs/tenstorrent/tt-metal-external-eth` (repo root). `export TT_METAL_HOME=$TTH`.
> DPU = BF3 embedded in desktop-0, `ubuntu@192.168.100.2` (pass `ubuntu`), rails p0/p1 → Blackhole dev1.

## 0. Bench prereqs (redo after any host/DPU reboot)
- Both rails 200G: `cat /sys/class/net/enp193s0f0np0/speed` (=200000). If down after reboot, force 200G:
  `sudo ethtool -s enp193s0f0np0 autoneg off speed 200000` (+ `...f1np1`). See `bh0/*` + `[[bh-bf3-port-mapping]]`.
- Jumbo MTU + tmfifo IP self-heal: `bash tt_metal/tt_rdma/bh0/setup_bench_persistence.sh` (installs keepers).
- **[DPU] `/tmp` is tmpfs — wiped on DPU reboot.** Rebuild the senders/bridge (see §2, §3).

## 1. Drainer pool — build + run + correctness
Build the pool test:
```
export TT_METAL_HOME=$TTH
cmake --build build_Release --target bh1_rx_worker_test
BIN=./build_Release/tests/tt_metal/tt_metal/tt_rdma_bh0/bh1_rx_worker_test
```
Run: `$BIN <device> <ext|int> <steps> <nworkers> <stride>`  (steps ≈ seconds×4; grid caps at **8 workers**).
Correctness at small frames uses the HOST raw sender (`bh0/tt_rdma_bf3_send`, allowlisted path in memory):
```
# launch pool (8 workers, 288B stride), wait for "Fire the DOCA sender", then fire the host sender:
$BIN 1 ext 10 8 288 &
sudo -n <allowlisted>/tt_rdma_bf3_send enp193s0f0np0 6000000 02:00:00:00:00:02 0x1af6 0x10 256 0x00CAFE42 0 4 32
```
Result line asserts: `processed == delivered`, `exactly-once HOLDS`, `eth drop=0`. Per-run accounting uses a
**baseline** (PKT_END is cumulative across runs) so `delivered = produced_seen - baseline`.

## 2. 200G line-rate acceptance (DOCA HW-TX source)
The CPU sender is fps-bound (~17.8G); the real line-rate source is `doca_ttblast` (DOCA Eth-Tx, ~198G).
```
# [DPU] (re)build the HW-TX sender:
bash tt_metal/tt_rdma/gw/deploy_doca_sender.sh          # -> /tmp/doca_ttblast on the DPU
# drive the pool at line rate (4112B stride matches doca_ttblast's 4080B payload + 32B hdr):
$BIN 1 ext 16 8 4112 &                                  # wait for "Fire the DOCA sender"
ssh ubuntu@192.168.100.2 'sudo /tmp/doca_ttblast -d mlx5_0 -m 02:00:00:00:00:02'   # mlx5_0=p0 -> BH idx2
```
**Expected (validated 2026-07-28):** `~198 Gbps, 100% kept up, lapped=0, exactly-once HOLDS, eth drop=0`.
The pool is NOT the bottleneck. Board does not bounce under the blast (keepers hold).

## 3. RoCEv2 gateway bridge (Arch-B, Phase B1) — build + eval
Architecture: ConnectX HW terminates full-spec RoCEv2; the bridge (`gw/rdma_bridge_sample.c`) re-heads each
WRITE_IMM to a TT-RDMA frame out p0; the pool (unchanged) lands it. See `[[tt-rdma-rocev2-gateway-arch-b]]`.
```
# [DPU] build the bridge (and the stock requester for testing):
bash tt_metal/tt_rdma/gw/deploy_rdma_bridge.sh          # -> /tmp/doca_ttbridge
# stock requester build (one-off): gcc against /opt/mellanox/doca/samples/doca_rdma/rdma_write_immediate_requester
```
RoCE setup: uplink ports have no GID; use SF `mlx5_2` (add IPv4 `10.99.0.1/24` to `enp3s0f0s0` → **GID idx 1**).
Bridge env: `TTBRIDGE_IFACE=p0 TTBRIDGE_DMAC=02:00:00:00:00:02 TTBRIDGE_RKEY=0x00CAFE42 TTBRIDGE_PLEN TTBRIDGE_MAX`.
**✅ B1 E2E COMPLETE + byte-exact verified (2026-07-28)** via the host-requester path (avoids the earlier
same-SF UAR-exhaustion blocker). desktop-0 `mlx5_0` (full PF) is the RoCEv2 requester → DPU `mlx5_2` bridge →
BH pool lands the RoCE payload byte-exact. Host↔DPU-SF RoCE just works: DPU **ovsbr1 already bridges pf0hpf +
en3f0pf0sf0 + p0** on one L2 domain (no new eSwitch steering rule).
```
# host: give mlx5_0 a RoCEv2 IPv4 GID (⚠ re-add if flushed; GID index drifts -> re-check show_gids)
sudo ip addr add 10.99.0.10/24 dev enp193s0f0np0
show_gids mlx5_0 | grep 10.99          # note the v2 (RoCEv2) index
# host: build the stock requester once (host has DOCA 3.4; uverbs/rdma_cm are world-rw -> no sudo needed)
gcc ... /opt/mellanox/doca/samples/doca_rdma/rdma_write_immediate_requester/* -> /tmp/doca_ttreq_host
# [DPU] bridge = RoCE-CM server + p0 egress. TTBRIDGE_BURST=N re-emits N TT frames per WRITE_IMM (validation
#       knob; default 1 = real 1:1) so ONE RoCE write makes a DENSE stream the pool's ring accounting validates:
sudo env TTBRIDGE_IFACE=p0 TTBRIDGE_MAX=1 TTBRIDGE_PLEN=256 TTBRIDGE_BURST=4096 /tmp/doca_ttbridge -d mlx5_2 -g 1 -cm -lp 51000
# BH pool (stride 288 = 32B hdr + 256B payload); fire the requester EARLY (before the MR-invalidate at hold/2):
bh1_rx_worker_test 1 ext 30 8 288 &
/tmp/doca_ttreq_host -d mlx5_0 -g <v2idx> -cm -sa 10.99.0.1 -lp 51000 -sat ipv4 -w "<payload>"
```
Result: processed==delivered==4096, valid>0, exactly-once HOLDS, drop=0; land zone == the exact RoCE string.
(Pool "LAND FAIL" is a doca_ttblast-magic "TTWR" assertion, not a real failure.) A **single sparse frame**
reads valid=0 (lands byte-exact per recv_probe, but the worker's `(index%nslots)*stride` slot map needs a
dense stream) — hence TTBRIDGE_BURST.

### B3 HW-TX egress (✅ prototype, 2026-07-28)
The bridge can egress via **DOCA Eth-TX HW-TX** instead of raw AF_PACKET — the same datapath doca_ttblast
proved at line rate, folded into the bridge process (RoCE responder on mlx5_2 + Eth-TX on mlx5_0, two PEs,
one thread). Select with `TTBRIDGE_EGRESS=doca` (now the DEFAULT; `=raw` for the AF_PACKET fallback),
`TTBRIDGE_TXDEV=mlx5_0`. Build now links `doca-eth doca-flow` + eth_common.c/eth_flow_common.c (see
`deploy_rdma_bridge.sh`).
```
sudo env TTBRIDGE_EGRESS=doca TTBRIDGE_TXDEV=mlx5_0 TTBRIDGE_PLEN=4080 TTBRIDGE_MAX=1 TTBRIDGE_BURST=<n> \
  /tmp/doca_ttbridge -d mlx5_2 -g 1 -cm -lp 51000
```
Correctness identical to B1 (byte-exact land, exactly-once, drop=0). `TTBRIDGE_TXBATCH=N` (default 64; 1 =
unbatched) sets frames per HW-TX `doca_task_batch`. **Perf (jumbo 4080B): batch=64 ~56 Gbps / 1.7 Mpps vs
unbatched ~47 G vs raw ~13 G.** **B3.1a (batching) done** — modest ~20% win; the wall is the **per-frame
memcpy** (4080B × 1.7 Mpps ≈ 6.5 GB/s, one Arm core saturated).

**B3.1b (zero-copy scatter-gather) DONE — NEGATIVE RESULT.** `TTBRIDGE_TXZC=1` chains a 46B header buf to a
payload buf pointing into a TX-side view of the RDMA responder mmap (`set_max_send_buf_list_len(2)`), so no
per-frame memcpy. **Correct (byte-exact land, drop=0) but ~20× SLOWER: ~1.8 Gbps vs batched-copy ~39 Gbps**
(500k jumbo took 9.1 s vs 0.42 s) — the DOCA **CPU-datapath 2-buf gather doesn't pipeline** (~18 µs/frame),
no send errors. So SG zero-copy is **not** the line-rate lever; kept gated OFF for reference. **The real
line-rate path = contiguous single-buf with NO per-send memcpy** (doca_ttblast hits 143–198G that way): land
the RoCE payload directly into a TX-ready frame slot (reserve 46B header space in the responder MR; unify the
RX landing ring with the TX frame ring), then TX single-buf. Bigger restructure — deferred. **Production
egress = B3.1a batched-copy (~39–56G).**

### DPA egress — full Arm offload (Phase A1, ✅ ~178G, 2026-07-28)
Move the re-head EGRESS onto the BF3 **DPA** (FlexIO) so the Arm does ZERO per-frame work. Prototype =
`gw/dpa_ttblast/dpa_ttblast.patch` over the stock FlexIO `packet_processor` sample: the DPA kernel gains
`tt_blast()` (build one 0x1AF6 WRITE frame + CRC, post `count` sends on its ETH SQ, SQ-CQ paced) invoked
host→DPA via `flexio_process_call` (a `__dpa_rpc__` entry). Deploy (provisions meson/ninja wheels to the
offline DPU, applies the patch, builds via dpacc+meson):
```
bash tt_metal/tt_rdma/gw/deploy_dpa_ttblast.sh            # build -> ~/flexio_samples/build/.../flexio_packet_processor
bash tt_metal/tt_rdma/gw/deploy_dpa_ttblast.sh --run      # blast 500k x 4080B jumbo on mlx5_0
```
**Measured ~178 Gbps jumbo (5.38 Mpps), Arm-free** (vs B3.1a copy ~56G, raw ~13G) → full offload is line-rate.
BH lands it exactly-once, drop=0. **Gotchas:** (1) it's `flexio_process_call`, NOT an event handler (which
only fires on an RX CQE). (2) the stock TX→vport rule matches **dst MAC == SMAC**, so A1 sends with dst=SMAC
(02:42:7e:7f:eb:02); the BH still routes unicast-"other" → RXQ2. Phase C changes the rule to keep dst=BH.
(3) DPA runs on the PF (mlx5_0), not the SF. **Next (DPA):** A2 (per-frame read landed payload) → A3 (trigger
off the RoCE RQ completion; SF-RoCE vs PF-DPA cross-device) → A4 full re-head → A5 perf.
**Next (egress overall):** B2 (MR federation via RISC1 doorbell 3.1e).

## 4. Regression / perf gates (run every phase change)
```
TT_METAL_HOME=$TTH bash tt_metal/tt_rdma/bh0/regression.sh   # correctness PASS/FAIL (small frames)
TT_METAL_HOME=$TTH bash tt_metal/tt_rdma/bh0/perf.sh          # latency + bandwidth vs baseline (add 'rebaseline' to reset)
```
Then the pool + 200G acceptance (§1, §2) as the line-rate gate. **Never SIGTERM an active-eth kernel**
mid-run — let it clean-shutdown via the stop flag, else the eth core wedges (needs `tt-smi -r` + re-force 200G).

## Gotchas index
- DPU `/tmp` wiped on reboot → rebuild via `gw/deploy_*.sh` + `cc -O2 -o /tmp/tt_send tt_rdma_bf3_send.c -lpthread`.
- Switchdev eSwitch drops host-PF→uplink on the slow path → run senders/bridge **on the DPU Arm**.
- Don't churn `ethtool -s` mid-test (self-inflicts link bounces).
- Head-publish must use RDMA-L1 (TX_BUF0), not the RCB/DBG region (base-FW-owned, stale as a NoC source).
