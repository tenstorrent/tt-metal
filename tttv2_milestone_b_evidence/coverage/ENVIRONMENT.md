# `mb-coverage` — environment

Recorded 2026-08-27, unattended, by the Milestone B step-7 coverage job.

## Tree

| | |
| --- | --- |
| Repository | `/proj_sw/user_dev/ctr-apbernal/tt-metal` |
| Branch | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` |
| Commit at start | `0c1ccd8557c7cb25cd1ca300d522eab1ed5db733` |
| Milestone A base for the boundary greps | `bc6ad03bfc2` (`Re-run the Milestone A device matrix and host gate at the committed tree`) |
| Milestone A reference branch | `gongyu/tttv2_wh_glx_2d_modules` — read only, never written |

## Host

| | |
| --- | --- |
| Hostname | `wh-glx6u-05-special-ctr-apbernal-for-reservation-117587` |
| Kernel | `6.8.0-83-generic` |
| CPUs | 64 |
| RAM | 566 GiB total, ~309 GiB available at start |
| Python | 3.10.21, `python_env/` in the repo (pre-built, **not** recreated) |
| torch | 2.11.0+cpu |
| transformers | 5.12.1 |
| `TT_METAL_HOME` | `/proj_sw/user_dev/ctr-apbernal/tt-metal` |
| `PYTHONPATH` | `/proj_sw/user_dev/ctr-apbernal/tt-metal` |
| tt-smi | 5.2.0 |
| `tenstorrent` kernel module | 2.4.1 |

### `HF_HOME`

**Unset in this job's inherited environment**, exactly as `mb-qwen` warned. Every
pytest invocation in this job exported it explicitly:

```sh
export HF_HOME=/proj_sw/user_dev/hf_data
```

With it unset, Llama's real-checkpoint host tests do not fail — they **skip**,
because `snapshot_download` falls through to the network and gets a 401 on a
gated repo. A green run that quietly skipped its only real-checkpoint coverage
is not evidence.

### Checkpoints present

| Checkpoint | State |
| --- | --- |
| `meta-llama/Llama-3.3-70B-Instruct` | present under `/proj_sw/user_dev/hf_data/hub` |
| `Qwen/Qwen3-32B` | **absent** — `~/.cache/huggingface/hub/models--Qwen--Qwen3-32B` holds `config.json` only. ~65 GB still has to be fetched. |

## Mesh — BLOCKED (infra), unchanged from `mb-qwen`

```text
ls /dev/tenstorrent | wc -l        32   <- the house-rules check. It LIES: stale nodes.
ls /sys/class/tenstorrent | wc -l  21   <- the real count.
missing: 0 1 2 3 4 5 6 7 10 11 14       (11 of 32)
present: 8 9 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
```

Identical to the set `mb-qwen` recorded. `tt-smi -ls` aborts inside `tt_umd`
topology discovery:

```text
Read 0xffffffff over PCIe ID 17: the board should be reset.
tt_umd/device/tt_device/tt_device.cpp:242 — TopologyDiscovery::get_connected_devices()
```

and every `ttnn` cluster open fails the same way at
`tt_device.cpp:398 — TTDevice::is_pcie_hung`.

**No recovery attempt was spent.** Two prior jobs used both of their permitted
attempts on this exact fault and both failed: `tt-smi -glx_reset` cannot run
because it must open `/dev/tenstorrent/7` first, and `tt-smi -r` re-initialises
only the visible boards before failing on PCIe ID 17. Neither can bring a board
back that is not on the bus. This needs an IPMI power cycle of the tray or a
host reboot, which is outside what an unattended job may do. Spending a third
job's attempts on a fault two jobs have already characterised would have been
waste, so this job recorded the state and moved to what could be measured.

Evidence: `logs/00_mesh_state_20260827T020110Z.log`,
`logs/01_device_open_attempt_partition_20260827T020120Z.log`, and the three
`logs/12_device_attempt_*.log`.

## Device-run procedure actually used

No pytest process in this job ever held the mesh, because no pytest process in
this job could open it. Every device invocation was a single foreground
`timeout … python -m pytest … > LOG 2>&1` with no pipe, `pgrep` checked before
each one, and it errored in ~2 s at cluster open. Nothing was killed; no PID was
signalled at any point.

## Test-count sanity

| Suite | Count |
| --- | --- |
| New host coverage (`models/common/tests/models/galaxy/test_step7_*.py`) | 162, all passing, three fresh processes |
| New Llama device coverage | 17 collected, **0 executed** |
| New Qwen device coverage | 16 collected, **0 executed** |
