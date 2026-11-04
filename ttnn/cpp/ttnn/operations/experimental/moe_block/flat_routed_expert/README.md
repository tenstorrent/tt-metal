# flat_routed_expert

Flat spatially pipelined routed experts on Blackhole: every local expert's GLU FFN of one chip in **one program**,
each expert's weights streamed once through a spatial pipeline. Per-expert token counts and region rows are read on
device, so one cached program serves every routing.

    y_e = act(x_e @ Wg_e, x_e @ Wu_e) @ Wd_e      for every local expert e

## Entry points

| | Use |
|---|---|
| `ttnn.experimental.moe_block.flat_routed_expert` | The plain op: y as bfp8 / bf16 tiles or row-major bf16 in DRAM. |
| `flat_routed_expert_chip_plan` (C++) | One chip's plan: the mesh plan with that chip's gate/up placement (`auto_tensix`). The plain op and routed_experts_combine build one program per chip from it. |
| `device/flat_program.hpp` (C++) | The program-building form: `add_flat_routed_expert_program` adds the flat expert's kernels, CBs and semaphores to a caller's per-chip `Program` (routed_experts_combine composes flat + combine in one program); `override_flat_routed_expert_addresses` patches its buffer addresses on cache hits; `with_combine_overlap` / `set_flat_combine_report` wire the y writers' reports to the combine. The header documents the contract (cores, semaphores, CB ids, arena, hash). |
| `ensure_flat_routed_expert_resources(MoEBlockResources&, config)` (C++) | The plan and persistent done / relay words, owned by the MoE block's resources. |
| `flat_routed_expert_plan`, `create_flat_routed_expert_done_words`, `FlatLayoutOptions` | The plan's layout facts (`as_dict()`), the done words, the layout choices. |

`flat_routed_expert_weights.py` (Python, here until it moves under `ttnn/ttnn/_experimental`): `FlatRoutedExpert`
(plan, weights packed into the plan's bank layout by `flat_weight_regions` + `bank_sharded`, expert id table, done
words), `act_ref`. The packing is host torch work done once per weight set, so it stays in Python.

The program hash is the attributes (`FlatRoutedExpertParams`, including `FlatLayoutOptions`) and the tensors' specs:
no buffer address. Every address (x, y, counts, regions, ids, weights, done words, the per-call arena and relay
words) is re-applied on a cache hit by `override_runtime_arguments`; plans are cached per mesh device and shape.

## Gate/up weight reads

Every gate/up core reads its own weight blocks from its weight slot's bank (`se_gu_read.hpp`): the receiver loop
(`se5_recv.cpp`) grants free landing slots by count, the reads fill granted slots in order and publish a landed-blocks
count the receiver and the compute consume. Slot r lives in bank r % banks (two slots a bank). Which core reads on
which NoC (the NCRISC on NOC0 beside the x forwarder, or the receiver BRISC on NOC1) and which cells serve which slot
is `FlatLayoutOptions.gu_source`; every source gives the same weight layout, schedule and bit-identical y.

## Roles and kernels (`device/kernels`)

| Role | Cores | Kernels |
|---|---|---|
| Schedule leader | reader cell 0 | `se_dyn_lead.cpp` (reads the call's counts / regions once; the other cores pull its compact copy) |
| x relays: 2 primaries + 2 helpers | next to the DRAM reader columns | `se11_xrd.cpp`, `se11_tz.cpp`, `se13_xhelp.cpp`, `se11_xmc.cpp` |
| gate/up | two full rectangles (64 cores, 48 on DSv4.1-Flash) | `se5_recv.cpp`, `se5_xfwd.cpp` (+ `se_gu_read.hpp`), `se3_compute.cpp` (+ `se_glu_faces.hpp`) |
| down | <= 32 cores, chains of 7 (4 with units) | `se6_drecv.cpp`, `se6_dw.cpp`, `se6_dcompute.cpp` (+ `se_yrm.hpp`, `se_cmb_done.hpp`) |

`se_dyn.hpp` builds the per-call schedule (active experts, sub-blocks, pinning of a dominant expert) identically on
every core; another op rebuilds it from the counts with `flat_schedule_defines`.

## The plan (fixed)

* The gate/up cores read their own weights; even splits (gate/up: the 2 It weight tiles of a K row over <= 64 cores, pairs where possible, else
  16-column units; down: the H tile columns over <= 32 cores). An intermediate size without an even split fails.
* x read once by the helper relays, fed along one unicast snake per gate/up rectangle (both rectangles full).
* Sub-blocks up to 128 rows; 64 with `rows_hint <= 320` on plans with gate/up row passes.

| | GLM-5.3 | GLM-5.3-Flash | K2.7 | K3 | DSv4.1-Flash | M3 |
|---|---|---|---|---|---|---|
| gate/up cores x tiles | 64 x 2 | 64 x 2 | 64 x 2 | 64 x 3 (units) | 48 x 3 (units) | 64 x 3 |
| down cores x columns | 32 x 6 | 32 x 4 | 32 x 7 | 28 x 4 | 32 x 5 | 32 x 6 |

## FlatLayoutOptions

| Field | Default | Meaning |
|---|---|---|
| `rows_hint` | -1 | expected rows per expert (the MoE block: `round(chunk * top_k / experts)`) |
| `x_helpers` | -1 (1) | x helper relays per rectangle; only 1 |
| `x_ready` | -1 | > 0: the relays wait for the dispatch ports' count in their x-ready word (dispatch in the same program) |
| `pin_ratio` | 8 | pin an expert only at this many times the next one's rows |
| `gu_source` | `tensix` | which gate/up core reads which weight blocks, on which NoC: `tensix` (each gate/up core's NCRISC reads its own blocks on NOC0), `tensix_split` (a bank's first slot's cores read on NOC1 from the receiver BRISC, the second slot's on NOC0), `tensix_noc1` (all on NOC1), `placed` (`gu_place`), `auto_tensix` (below); same layout, schedule and bit-identical y. Without per-core words each slot's gate/up cores are placed by hops from the bank's NoC endpoint the reads go through |
| `gu_read_depth` | -1 (2) | weight reads in flight a gate/up core, 1..6 |
| `gu_read_pieces` | -1 (1) | each weight block read as 1 / 2 / 4 reads (more, smaller reads are slower in the op on GLM shapes; K3 in the MoE block takes 2) |
| `gu_place` | [] | `gu_source` `placed`: per gate/up cell `x << 24 \| y << 16 \| bank << 8 \| how` (1 NOC0, 2 NOC1); 2 x RG cores a bank |
| `gu_source` `auto_tensix` | | the gate/up placement chosen per chip at plan time by a NoC link-load search (`device/flat_gu_place.hpp`) over that chip's own geometry (harvested columns, DRAM tiles: `tt-metalium/experimental/dram_geometry.hpp`): `flat_routed_expert_chip_plan`; chips of one geometry share it, and a chip takes the mesh's compromise placement when that loads its busiest link less. Same weight layout on every chip. `flat_gu_placement_report` (host only, also on a mock cluster) shows the per-chip / reference / compromise model |

## Tests

`tests/ttnn/unit_tests/operations/experimental/moe_block/flat_routed_expert/`: `test_flat_routed_expert.py` (device:
PCC vs torch, bit-exact vs the Python reference's saved outputs with `MOE_BLOCK_REFS`, host time per call),
`test_flat_routed_expert_host.py` (host only: weight layout), `dump_ref_flat_routed_expert.py` (the reference cases,
run in the Python reference tree).
