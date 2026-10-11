# fabric_all_gather_ttnn (ttnn.bringup.fabric_all_gather)

- Source: `ttnn/cpp/ttnn/operations/experimental/fabric_all_gather` on `mstaletovic/mimo-v2-dp` @ `b45a923d73c`. New op: it exists on that branch only, not in ttnn/cpp on this tree (no source op to keep in sync,
  no upstream tests to carry: `tests/source.yaml` is empty).
- Python: `ttnn.bringup.*` (was `ttnn.experimental.*` on the source branch)
- Brought in for: glm53_flash_d_p_lb (MoE on the LoudBox, the MiMo-V2 all-gather MoE design: same hidden / expert
  width / top-8 as MiMo-V2.6)
- Used by: glm53_flash_d_p (tt/experts_ag.py, GLM_EXPERTS_MODE=ag; the whole model on the LoudBox via glm53_flash_d_p_lb)

A drop-in for `ttnn.experimental.high_bw_all_gather` (same signature, validation, program-cache rules and output) with
its own program: per link a port core placed by NoC1 hops to its Ethernet core, store-and-forward relays gated by arrival
counters; runs at ~84% of a LoudBox link (README.md). Its device operation derives from the stock high_bw_all_gather's
and shares its argument resolution: `high_bw_all_gather_build_operation_args` is defined in the stock op but only
declared in its header on the source branch, so this fork declares it in `fabric_all_gather.cpp` (the stock op is not
edited; the stock namespace is spelled out in full). Also here: the Python prototype `fabric_all_gather_py.py`
(from `ttnn/ttnn/operations/examples/fabric_all_gather`: importable as ttnn.bringup.fabric_all_gather_ttnn; its
`plan()` is the planning code ttnn.bringup.fabric_reduce_scatter uses), its README / report (`README_python.md`,
`report_python.md`) and CLI (`python -m ttnn.bringup.fabric_all_gather_ttnn`).

Mechanical changes (fork_op.py on a temporary checkout of the source folder): namespace `ttnn::operations::bringup`,
CMake target `ttnn_op_bringup_fabric_all_gather_ttnn`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

Tests: `tests/unit/` (the source branch's `tests/ttnn/unit_tests/operations/ccl/test_fabric_all_gather_op.py`,
`test_fabric_all_gather_device_time.py` and `operations/examples/test_fabric_all_gather.py` for the Python one).

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Direct hops only
- What: plan() rejects a hop between chips that are not direct fabric neighbours (ValueError "not a direct link",
  via the new binding ttnn.get_neighbor_eth_directions), for every scheme. Supported groups are unchanged.
- Why: a 2D fabric forwards toward any chip, so get_forwarding_link_indices alone accepted e.g. a ring along a
  LoudBox row (no wrap cable): its closing hop was routed back through the line and the call deadlocked.
- Needed by: glm53_flash_d_p_lb (fabric_reduce_scatter ring / full mesh, which plans with this module)
- Files: fabric_all_gather_py.py; ttnn/cpp/ttnn-nanobind/fabric.cpp + ttnn/ttnn/__init__.py (the binding)
