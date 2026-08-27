# `mb-qwen` environment

Recorded 2026-08-27 by the `mb-qwen` job, unattended.

## Tree

| Item | Value |
| --- | --- |
| Repository | `/proj_sw/user_dev/ctr-apbernal/tt-metal` |
| Branch | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` |
| Commit at job start | `309d3481a68e7979d7d3cc4241dcd6c2a5f872f0` |
| Host | `wh-glx6u-05-special-ctr-apbernal-for-reservation-117587` |
| Kernel | `6.8.0-83-generic` |

## Python

| Item | Value |
| --- | --- |
| Interpreter | `/proj_sw/user_dev/ctr-apbernal/tt-metal/python_env/bin/python` |
| Python | 3.10.21 |
| torch | 2.11.0+cpu |
| transformers | 5.12.1 |
| tt-smi | 5.2.0 |

The venv and `build/` were reused, never rebuilt, as the brief required.

`transformers` 5.12.1 ships the full Qwen3 stack — `Qwen3Config`,
`Qwen3ForCausalLM`, `Qwen3Attention`, `Qwen3RMSNorm`, `Qwen3RotaryEmbedding` —
so an independent HF reference for every host claim in this job was available
**without the checkpoint weights**. That is what made the host qualification
possible on a night when neither the mesh nor the weights were.

## Mesh: DOWN for the whole job

**No device work was possible. Not one device test ran.**

```text
ls /dev/tenstorrent | wc -l        32     <- stale nodes, misleading
ls /sys/class/tenstorrent | wc -l  21     <- the real count
```

Eleven boards are absent from `sysfs` entirely and are not on the PCIe bus:

```text
missing: 0 1 2 3 4 5 6 7 10 11 14
present: 8 9 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
```

A WH Galaxy `(8, 4)` mesh needs all 32. `tt-smi -ls` aborts inside `tt_umd`
during topology discovery:

```text
Error in detecting devices!
Read 0xffffffff over PCIe ID 17: the board should be reset.
Location: /project/device/tt_device/tt_device.cpp:242
  tt::umd::TTDevice::init_tt_device(...)
  tt::umd::TopologyDiscovery::get_connected_devices()
```

**`ls /dev/tenstorrent | wc -l` is not a mesh health check on this host.** It
returned the expected 32 while eleven of those nodes were stale. The house-rules
run procedure opens with that check; `ls /sys/class/tenstorrent | wc -l` is the
one that tells the truth. This is worth fixing in the procedure.

### Both recovery attempts were spent, and both failed

The house rules allow two. Logs: `logs/01_…`, `logs/02_…`.

1. **`tt-smi -glx_reset`** — fails immediately:
   `[Errno 6] No such device or address: '/dev/tenstorrent/7'`. The reset issues
   `USER_RESET` on all 32 devices before the IPMI step, so it must open the very
   nodes that are gone. This is the same failure `mb-llama` recorded.
2. **`tt-smi -r`** (the Galaxy 6U path tt-smi itself suggests) — ran ~4.5 min,
   reset only the 21 enumerated devices, then failed re-initialising:
   `Read 0xffffffff over PCIe ID 17`. It also warns that CPLD FW ≥ v1.16 is
   required for `-r` on Galaxy and to use `-glx_reset` otherwise.

Neither path can recover a board that is not on the PCIe bus. This needs an
**IPMI power cycle of the tray or a host reboot** — outside what an unattended
job may do. Recorded as `BLOCKED (infra)`.

Note the failing board ID moved between jobs — `mb-llama` reported board 7,
this job sees PCIe ID 17 — and eleven boards are now missing rather than one, so
the fault has **widened since the previous night**, not stayed put.

## Checkpoint: Qwen3-32B weights are NOT on this machine

A second, independent blocker, exactly as `mb-llama`'s handoff warned:

```text
~/.cache/huggingface/hub/models--Qwen--Qwen3-32B     12K   config.json only
/proj_sw/user_dev/hf_data/hub/                       no models--Qwen--Qwen3-32B
                                                     (only an empty .locks entry)
```

`HF_HOME` is **unset** in this job's environment. The shared cache
`/proj_sw/user_dev/hf_data` holds `Llama-3.3-70B-Instruct` and `gpt-oss-120b`
but **no Qwen3-32B**. So even with a healthy mesh, the full-model and
teacher-forced-accuracy steps could not have run tonight without first
downloading ~65 GB.

The `config.json` **is** present, and it is authoritative — this job used it to
settle the checkpoint contract and the QKV-bias question (see `REPORT.md`).

## What this environment did allow

Host-only work, which is where all of this job's evidence comes from:

* 13 new host tests in
  `models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py`,
  run in **three fresh processes**, 13 passed each time;
* the pure-host regression gate;
* static verification of the 64-head geometry, the ring widths, the residual
  dtype and the `wo` placement pairing.

`import ttnn` and importing `models.common.models.galaxy.recipes` do **not**
open a device and are safe with the mesh down — that is what let the geometry be
derived from production code rather than restated.

### The "host" regression gate is not host-only

The brief's regression command includes `models/common/tests/modules`, which
collects **device** suites. With the mesh down they error rather than skip:

```text
397 passed, 2054 skipped, 3273 deselected, 289 errors
```

All 289 errors are device-open failures (`Read 0xffffffff`, at
`conftest.py:452` and `ttnn/distributed/distributed.py:631`) in
`*_wh_galaxy*.py` plus the three `moe/` device suites. **Zero `FAILED`.**
`logs/20_…` is that run; `logs/21_…` is the same gate with the device suites
excluded.
