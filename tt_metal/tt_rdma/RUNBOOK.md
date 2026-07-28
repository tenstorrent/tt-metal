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
```
# [DPU] bridge = RoCE-CM server + p0 egress:
sudo env TTBRIDGE_IFACE=p0 TTBRIDGE_MAX=<n> TTBRIDGE_PLEN=<p> /tmp/doca_ttbridge -d mlx5_2 -g 1 -cm -lp 51000
# requester = RoCE-CM client:
sudo /tmp/doca_ttreq -d mlx5_2 -g 1 -cm -sa 10.99.0.1 -lp 51000 -sat ipv4 -w "<payload>"
```
**Known blocker (open):** two RoCE processes on one SF → `Failed to create UAR` (per-SF UAR exhaustion); the
two SFs are on different PFs (no loopback path). **Fix path:** use desktop-0 `mlx5_0` (full PF) as the
requester → DPU `mlx5_2` bridge (real gateway topology; needs eSwitch host→SF steering). Verify: pool sees
valid TT-RDMA WRITE landings byte-exact. Then B2 (MR federation via RISC1 doorbell), B3 (DPA line-rate egress).

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
