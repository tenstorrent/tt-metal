# Environment Baseline — WH Galaxy Milestone A Device Evidence

Recorded once, before any device pytest process was started.

## Wall-clock

| Event | UTC |
| --- | --- |
| Bootstrap start (`run_overnight_device_evidence.sh`) | 2026-08-24T17:40:20Z |
| tt-metal build complete | 2026-08-24T18:02:51Z |
| Python environment complete | 2026-08-24T18:22:17Z |
| Agent start | 2026-08-24T18:22:46Z |
| **Device evidence run start (this baseline)** | **2026-08-24T18:27:07Z** |

## Commit under test

```
$ git rev-parse HEAD
de4c8f4e659cb7a0dfd255f0b34d33f42f43a0bd

$ git rev-parse --abbrev-ref HEAD
gongyu/tttv2_wh_glx_2d_modules
```

Commit subject: `add reusable WH Galaxy 2D modules`.

```
$ git status --short
?? run_overnight_device_evidence.sh
?? tttv2_milestone_a_device_evidence/
?? tttv2_milestone_a_device_evidence_agent.md
```

No tracked file is modified. The three untracked entries are the overnight driver script, this
evidence directory, and the agent brief — none of them is repository source or test code.

## Submodules

```
$ git submodule status --recursive
 29125b7ad8b5513eeaa4417ed92892bf39c8bd74 models/demos/t3000/llama2_70b/reference/llama (heads/main)
 117100515bb21d9a6b3a8f0eee50ecd91f961408 tt_metal/third_party/tracy (v0.13.3-tt.0-9-g11710051)
 7b2176e2fe913089f8cd2be9dfb738ead6e7aa27 tt_metal/third_party/tt-cluster-descriptors (remotes/origin/public-release-prep-8-g7b2176e)
 9904682cc18cb4ebb63cb9681613a24345fbfacc tt_metal/third_party/umd (v0.9.5-232-g9904682c)
```

Every entry has a leading space: all submodules are initialized and at the recorded SHA.

## Host

| Item | Value |
| --- | --- |
| Hostname | `wh-glx6u-05-special-ctr-apbernal-for-reservation-116669` |
| Kernel | `Linux 6.8.0-83-generic x86_64` |
| CPUs | 64 |
| RAM | 566 GiB total |
| `tt-smi` version | 5.2.0 |
| `tenstorrent` kernel module | 2.4.1 |

## Devices

```
$ ls /dev/tenstorrent
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31

$ ls /dev/tenstorrent | wc -l
32
```

32 device nodes present — a complete WH Galaxy `(8, 4)` mesh. The Step 2 stop condition
("fewer than 32 device nodes") does not apply.

`tt-smi -ls` reports 32 UMD chip IDs, all `Wormhole` / `tt-galaxy-*-L` boards across PCI buses
`0000:01`–`0000:08`, `0000:41`–`0000:48`, `0000:81`–`0000:88`, `0000:c1`–`0000:c8`.

Raw captures:

- `logs/01_tt_smi_before.log` — taken by the bootstrap at 17:40Z, before the build.
- `logs/01b_tt_smi_before_agent.log` — taken by this agent at 18:27Z, immediately before the
  first device pytest process.
- `logs/99_tt_smi_after.log` — final state, taken after the last device run.

## Build

Built by the bootstrap script, not by this agent (the brief forbids rebuilding).

```
$ ./build_metal.sh --enable-ccache
INFO: Build type: Release
INFO: Running: cmake -B build_Release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=build_Release -DENABLE_CCACHE=TRUE \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF -DTT_UNITY_BUILDS=ON \
  -DTT_ENABLE_LIGHT_METAL_TRACE=ON -DWITH_PYTHON_BINDINGS=ON \
  -DENABLE_DISTRIBUTED=ON -DENABLE_FAKE_KERNELS_TARGET=OFF \
  -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake
```

| Item | Value |
| --- | --- |
| Build type | `Release` (ccache enabled, Ninja, clang-20 / libstdc++ toolchain) |
| Build directory | `build_Release` |
| Build log | `logs/02_build.log` (exit 0, 17:40:22Z → 18:02:51Z) |
| Distributed | `ENABLE_DISTRIBUTED=ON` |
| Light metal trace | `TT_ENABLE_LIGHT_METAL_TRACE=ON` |

## Python environment

```
$ ./create_venv.sh          # logs/03_create_venv.log, exit 0, 18:02:51Z -> 18:22:17Z
$ source python_env/bin/activate
$ python --version
Python 3.10.21
$ command -v python
/proj_sw/user_dev/ctr-apbernal/tt-metal/python_env/bin/python
$ python -c 'import ttnn; print("ttnn import OK")'
ttnn import OK
```

Exported for every device run:

```
TT_METAL_HOME=/proj_sw/user_dev/ctr-apbernal/tt-metal
PYTHONPATH=/proj_sw/user_dev/ctr-apbernal/tt-metal
```

`pytest.ini` supplies `addopts = --import-mode=importlib -vvs -rA --durations=25
--junitxml=generated/test_reports/most_recent_tests.xml`; the run command adds
`-v -rA --color=no --showlocals -p no:cacheprovider` on top of it.

## Test selection

`logs/00_collect.log` — the brief's broad collection command over the seven module directories
with `-k wh_galaxy`: `27/5513 tests collected (5486 deselected)`.

Six of those 27 are host-only name matches, not device cases:
`test_resolution_fails_closed_on_non_wh_galaxy` (3 parametrizations each in
`modules/rmsnorm/test_rmsnorm_2d.py` and `modules/mlp/test_mlp_2d.py`). Those sibling
`test_*_2d.py` files are explicitly out of scope in the brief.

`logs/00_collect_nodeids.log` — collection restricted to the seven `*_wh_galaxy.py` files:
**21 device node IDs**, exactly matching the brief's expected matrix.
