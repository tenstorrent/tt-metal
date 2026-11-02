# flat_routed_expert

Flat spatially pipelined routed experts on Blackhole: every local expert's GLU FFN of one chip in **one program**,
each expert's weights streamed once through a spatial pipeline. Per-expert token counts and region rows are read on
device, so one cached program serves every routing.

    y_e = act(x_e @ Wg_e, x_e @ Wu_e) @ Wd_e      for every local expert e

## Entry points

| | Use |
|---|---|
| `ttnn.experimental.moe_block.flat_routed_expert` | The plain op: y as bfp8 / bf16 tiles or row-major bf16 in DRAM. |
| `device/flat_program.hpp` (C++) | The program-building form: `add_flat_routed_expert_program` adds the flat expert's kernels, CBs and semaphores to a caller's per-chip `Program` (routed_experts_combine composes flat + combine in one program); `override_flat_routed_expert_addresses` patches its buffer addresses on cache hits; `with_combine_overlap` / `set_flat_combine_report` wire the y writers' reports to the combine. The header documents the contract (cores, semaphores, CB ids, arena, DRISC, hash). |
| `ensure_flat_routed_expert_resources(MoEBlockResources&, config)` (C++) | The plan, persistent done / relay words and the DRISC senders, owned by the MoE block's resources. |
| `flat_routed_expert_plan`, `create_flat_routed_expert_done_words`, `FlatDriscSenders`, `FlatLayoutOptions` | The plan's layout facts (`as_dict()`), the done words, the persistent DRISC senders, the layout choices. |

`flat_routed_expert_weights.py` (Python, here until it moves under `ttnn/ttnn/_experimental`): `FlatRoutedExpert`
(plan, weights packed into the plan's bank layout by `flat_weight_regions` + `bank_sharded`, expert id table, done
words, DRISC senders), `act_ref`. The packing is host torch work done once per weight set, so it stays in Python.

The program hash is the attributes (`FlatRoutedExpertParams`, including `FlatLayoutOptions`) and the tensors' specs:
no buffer address. Every address (x, y, counts, regions, ids, weights, done words, the per-call arena and relay
words) is re-applied on a cache hit by `override_runtime_arguments`; plans are cached per mesh device and shape.

## DRISC senders

The gate/up weights are read and forwarded by persistent kernels on the DRAM cores (`se_drisc_gu.cpp`), launched once
per mesh device through the experimental tt-metal API `tt-metalium/experimental/dram_core_kernels.hpp` (slow dispatch
beside fast dispatch) and kept running across calls. `FlatDriscSenders::get(mesh, plan.drisc_n)` launches or shares
them; they stop when the last holder drops them, on `stop()` (after finishing the command queue), or at mesh close
(the API runs the senders' stop protocol there). Every slot comes from the DRAM cores: a bank's free tile sends its
first slot, the bank's NOC1-endpoint tile its second (switch mode). The program checks the senders run.

## Roles and kernels (`device/kernels`)

| Role | Cores | Kernels |
|---|---|---|
| DRISC weight senders | 2 DRAM tiles a bank (persistent) | `se_drisc_gu.cpp`, `se_drisc.hpp` |
| Launcher | reader cell 0 | `se_drisc_go.cpp` (posts every slot's job, `se_drisc_post.hpp`) |
| x relays: 2 primaries + 2 helpers | next to the DRAM reader columns | `se11_xrd.cpp`, `se11_tz.cpp`, `se13_xhelp.cpp`, `se11_xmc.cpp` |
| gate/up | two full rectangles (64 cores, 48 on DSv4.1-Flash) | `se5_recv.cpp`, `se5_xfwd.cpp`, `se3_compute.cpp` (+ `se_glu_faces.hpp`) |
| down | <= 32 cores, chains of 7 (4 with units) | `se6_drecv.cpp`, `se6_dw.cpp`, `se6_dcompute.cpp` (+ `se_yrm.hpp`, `se_cmb_done.hpp`) |

`se_dyn.hpp` builds the per-call schedule (active experts, sub-blocks, pinning of a dominant expert) identically on
every core; another op rebuilds it from the counts with `flat_schedule_defines`.

## The plan (fixed)

* DRISC only; even splits (gate/up: the 2 It weight tiles of a K row over <= 64 cores, pairs where possible, else
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
| `drisc_tiles` | `both` | DRISC tiles a bank sends its gate/up slots from: `both` (free + NOC1 endpoint, every call), `free_only`, `noc1_endpoint_only` (one tile serves both of the bank's slots; the latter leaves the free tiles to another resident kernel) |

## Tests

`tests/ttnn/unit_tests/operations/experimental/moe_block/flat_routed_expert/`: `test_flat_routed_expert.py` (device:
PCC vs torch, bit-exact vs the Python reference's saved outputs with `MOE_BLOCK_REFS`, host time per call),
`test_flat_routed_expert_host.py` (host only: weight layout), `dump_ref_flat_routed_expert.py` (the reference cases,
run in the Python reference tree).
