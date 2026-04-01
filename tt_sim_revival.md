# TTSim Revival

## Build & Run

Set the following env vars:
`export TT_SIM_PATH=/your/favourite/path/here`
`export TT_METAL_SIMULATOR="$TT_SIM_PATH/libttsim.so"`
`export TT_METAL_SLOW_DISPATCH_MODE=1`
`export TT_MULTIPROC_SIM_ENABLE=1`

### TTSim repo
`git clone git@github.com:tenstorrent/ttsim-private.git`

`git checkout nzhao/eth-io-rebased-logs`

`cd src`

`../make.py _out/release_wh/libttsim.so _out/release_bh/libttsim.so`

#### For Wormhole:
`cp ~/ttsim-private/src/_out/release_wh/libttsim.so $TT_SIM_PATH`

`cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml $TT_SIM_PATH/soc_descriptor.yaml`

#### For Blackhole:
`cp ~/ttsim-private/src/_out/release_bh/libttsim.so $TT_SIM_PATH`

`cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml $TT_SIM_PATH/soc_descriptor.yaml`


### TTMetal repo

`git checkout nzhao/multidev-sim-rid-rank` (Make sure to update submodules)

`./build_metal.sh --build-metal-tests`

`cp build/simulation/umd/child_process_tt_sim_chip $TT_SIM_PATH`


#### Test single host is working
Check `tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors` for desired mock cluster descriptor

`export TT_METAL_MOCK_CLUSTER_DESC_PATH=/your/favorite/cluster/here`

For example:

Assuming we set `export TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_6u_cluster_desc.yaml`

Make sure you ttsim files copied correspond to BH architecture

`build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml`

Should see the test run with periodic timing and the following at the end.
```
2026-04-01 14:34:04.578 | info     | EmulationDriver | Sending exit signal to remote... (tt_sim_chip_impl.cpp:144)
[34521870] branch_mispredicts=666632
[34521870] icache_misses=155544
[34521870] dcache_misses=1771088357
[34521870] 383.9 seconds (89.9 KHz)
```

### Multihost
`alias tt-run=ttnn/ttnn/distributed/ttrun.py`

Try to run a simple multihost test.

Note: The following requires the ttsim WH architecture files and/or increase timeout
```
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_quad_host_cluster_desc_mapping.yaml --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml --mpi-args "--tag-output --allow-run-as-root" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x16_quad_galaxy\""
```

### WIP for BH GLX 4 stage pipeline

### -----------------------------UNCLEAR WHETHER THE FOLLOWING IS IMPORTANT-----------------------------
Ridvan has some changes to multi-mesh allocation which have been cherry-picked onto a rebase of Austin's ttsim changes onto main. This includes changes to umd which have undergone the same process

His changes introduce changes which require generation of mock mapping and rank bindings files for `tt-run`.

To generate the mock mapping, run a dummy test passing in the MGD and a `--mock-cluster-rank-binding`

Note: The mock cluster rank binding should now have all ranks map to the same cluster descriptor instead of separate ones
Note: The rank bindings file will have non-sequential `TT_VISIBLE_DEVICES` which are sourced from generation

For example with BH GLX, we run a dummy test to generate the `--rank-binding` and `mock-cluster-rank-binding` files which are used for tt-run. The results are places in `$TT_METAL_HOME/generated/ttrun/`
```
tt-run --mesh-graph-descriptor $TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_2x4_pipeline.textproto --mock-cluster-rank-binding $TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_single_galaxy_cluster_desc_mapping.yaml build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml
```

^ Sometimes this generation fails due to socket addresses in use, not caused by zombine processes from previous runs. Currently no workaround for this.

### NOTE
This part of the workflow is still WIP. This comment regarding blitz decode pipeline test probably means we need to use the old workflow (which is why we are using `generate_rank_bindings.py` below).

```
// FIXME: With topology mapper, stage_index % num_procs may no longer map to the rank that owns
// the entry/exit ASICs (tray_id, asic_location); multi-rank-per-host can place those chips on a different rank.
```
### ----------------------------------------------------------------------------------------------------

For 4 stage BH GLX rank binding generation:
`python3 tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py`

Now we are able to run (WIP) the Deepseek 4 stage galaxy test. This uses `--rank-binding` from `generate_rank_bindings.py` and `--mock-cluster-rank-binding` which has the rank to cluster mapping. Use the mock cluster rank binding (which with Ridvan's changes has all ranks mapping to the same cluster instead of separate clusters for the same machine)
```
tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_single_galaxy_cluster_desc_mapping.yaml --rank-binding /localdev/nzhao/tt-metal/bh_galaxy_split_4x2_multi_mesh_rank_binding.yaml --mpi-args "--tag-output --allow-run-as-root" bash -c "source ./python_env/bin/activate && pytest -svv models/demos/deepseek_v3_b1/tests/unit_tests/test_lm_head_sampling.py::test_pipeline_block_4stage_galaxy_1_iteration"
```

This currently throws a runtime error during blitz pipeline creation due to H2D/D2H socket creation which is unsupported in umd sim.

`RuntimeError: Failed to create SysmemBuffer for MMIO device 0`


### Common Errors
```
[840403] ERROR: UnimplementedFunctionality: e_tile_mmio_wr32: addr=0xfffffffc
terminate called after throwing an instance of 'std::runtime_error'
  what():  TT_THROW @ /localdev/nzhao/tt-metal/tt_metal/third_party/umd/device/simulation/process_manager.cpp:157: tt::exception
info:
Failed to read response message: Connection reset by peer
```

`rm -rf tt_metal/pre-compiled` then try again



```
[1,1]<stderr>: terminate called after throwing an instance of 'std::runtime_error'
[1,1]<stderr>:   what():  TT_THROW @ /localdev/nzhao/tt-metal/tt_metal/third_party/umd/device/simulation/eth_connection.cpp:77: tt::exception
[1,1]<stderr>: info:
[1,1]<stderr>: Server socket failed to bind socket: Address already in use
```

Zombie process from previous run needs to be killed: `pkill -9 -f "child_process_tt_sim_chip"`


`No matching ASIC ID`

Rank bindings file is incorrect. Tricky ->  using `generate_rank_bindings.py` and try to swap `TT_VISIBLE_DEVICES` of different ranks
