# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
from ttnn.operations.conv.tt_py_op import TTPyOp
from ttnn.operations.conv.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
    generate_untilize_with_halo_kernel_configs,
)
from tt_lib.utils import _nearest_y
from ttnn.operations.conv.sliding_window_op_utils import (
    SlidingWindowOpParamsWithParallelConfig,
    get_hash_from_sliding_window_op_params,
    get_sliding_window_op_output_shard_nhw_size,
    calculate_shard_grid,
)
import ttnn
import torch
import struct


class TTPyUntilizeWithHalo(TTPyOp):
    def __init__(
        self,
        device,
        sliding_window_op_params: SlidingWindowOpParamsWithParallelConfig,
        halo_reader_patterns_cache,
        pad_val=0x0,
        is_out_tiled=True,
        transpose_mcast=True,
        mesh_mapper=None,
    ):
        self.sliding_window_op_params = sliding_window_op_params
        self.device = device
        self.mesh_mapper = mesh_mapper
        self.transpose_mcast = transpose_mcast
        sliding_window_op_params_hash = get_hash_from_sliding_window_op_params(sliding_window_op_params)
        self.set_op_configs(
            device, sliding_window_op_params_hash, sliding_window_op_params, halo_reader_patterns_cache, is_out_tiled
        )
        assert sliding_window_op_params_hash in halo_reader_patterns_cache
        utwh_kernel_configs = halo_reader_patterns_cache[sliding_window_op_params_hash]

        def utwh_(activation):
            return ttnn.untilize_with_halo_v2(
                activation,
                utwh_kernel_configs["padding_config"],
                utwh_kernel_configs["local_config"],
                utwh_kernel_configs["remote_config"],
                pad_val=pad_val,
                ncores_nhw=self.sliding_window_op_params.num_cores_nhw,
                max_out_nsticks_per_core=utwh_kernel_configs["max_out_nsticks_per_core"],
                memory_config=utwh_kernel_configs["out_mem_config"],
                remote_read=utwh_kernel_configs["remote_read"],
                transpose_mcast=self.transpose_mcast,
            )

        self.utwh = utwh_

    # override abstract methods from base class TTPyOp
    def set_op_configs(
        self,
        device,
        sliding_window_op_params_hash,
        sliding_window_op_params,
        halo_reader_patterns_cache,
        is_out_tiled=True,
    ):
        if sliding_window_op_params_hash not in halo_reader_patterns_cache:
            stride_h = sliding_window_op_params.stride_h
            stride_w = sliding_window_op_params.stride_w
            pad_h = sliding_window_op_params.pad_h
            pad_w = sliding_window_op_params.pad_w
            window_h = sliding_window_op_params.window_h
            window_w = sliding_window_op_params.window_w
            input_n = sliding_window_op_params.batch_size
            input_h = sliding_window_op_params.input_h
            input_w = sliding_window_op_params.input_w
            # TODO: Had to add this (should this be shard grid?)
            num_cores_w = sliding_window_op_params.num_cores_w
            num_cores_h = sliding_window_op_params.num_cores_h
            num_cores_nhw = sliding_window_op_params.num_cores_nhw
            act_reshard_num_cores_nhw = sliding_window_op_params.act_reshard_num_cores_nhw
            assert num_cores_nhw > 0
            # TODO: send input_nhw_shape to generate functions (no need for C)
            # output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups
            sliding_window_op_all_params = [1, 1, window_h, window_w, stride_h, stride_w, pad_h, pad_w, 1, 1]
            input_nchw_shape = [input_n, 1, input_h, input_w]
            pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
                sliding_window_op_all_params, input_nchw_shape
            )
            sliding_window_output_shard_nhw_size = get_sliding_window_op_output_shard_nhw_size(
                num_cores_nhw,
                input_n,
                input_h,
                input_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                window_h,
                window_w,
                is_out_tiled,
            )
            if is_out_tiled:
                untilize_w_halo_input_nhw_size_to_shard_evenly = _nearest_y(
                    input_n * input_h * input_w, num_cores_nhw * 32
                )
            else:
                untilize_w_halo_input_nhw_size_to_shard_evenly = _nearest_y(input_n * input_h * input_w, num_cores_nhw)
            untilize_with_halo_input_shard_nhw_size = (int)(
                untilize_w_halo_input_nhw_size_to_shard_evenly / num_cores_nhw
            )
            req_conv_input_shard_start_end, tensor_metadata = decompose_conv_into_shards_and_generate_tensor_metadata(
                data_top_left_indices,
                pad_metadata,
                input_w + (2 * pad_w),
                sliding_window_output_shard_nhw_size,
                untilize_with_halo_input_shard_nhw_size,
                num_cores_nhw,
                window_h,
                window_w,
                act_reshard_num_cores=act_reshard_num_cores_nhw,
                input_nhw_height=input_n * input_h * input_w,
            )

            shard_grid, shard_layout = calculate_shard_grid(
                (num_cores_w, num_cores_h), num_cores_nhw, self.transpose_mcast
            )
            block_sharding = shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED

            def get_memory_config(shard_shape, buffer_type=ttnn.BufferType.L1_SMALL):
                shard_orientation = (
                    ttnn.ShardOrientation.ROW_MAJOR
                    if not block_sharding
                    else (ttnn.ShardOrientation.COL_MAJOR if self.transpose_mcast else ttnn.ShardOrientation.ROW_MAJOR)
                )
                shard_halo = False
                shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)
                mem_layout = (
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED if block_sharding else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                )
                mem_config = ttnn.MemoryConfig(mem_layout, buffer_type, shard_spec)
                return mem_config

            def gen_per_core_gather_data_uint16_tensor(config: list):
                assert type(config) is list
                if block_sharding:
                    if self.transpose_mcast:
                        assert len(config) == num_cores_w, f"{len(config)} {num_cores_w}"
                    else:
                        assert len(config) == num_cores_h, f"{len(config)} {num_cores_h}"
                else:
                    assert len(config) == num_cores_nhw, f"{len(config)} {num_cores_nhw}"
                assert type(config[0]) is list
                assert len(config[0]) > 0

                torch_tensor = torch.tensor(config, dtype=torch.short)
                shard_shape = [1, torch_tensor.shape[-1]]

                if block_sharding:
                    if self.transpose_mcast:
                        torch_tensor = torch_tensor.repeat(1, num_cores_h)
                    else:
                        torch_tensor = torch_tensor.repeat(1, num_cores_w)

                torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
                tt_tensor = ttnn.from_torch(
                    torch_tensor, dtype=ttnn.DataType.UINT16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mesh_mapper
                )
                tt_tensor = tt_tensor.to(device, get_memory_config(shard_shape)) if device is not None else tt_tensor
                return tt_tensor

            def core_id_to_physical_coord(core_id):
                if block_sharding:
                    if self.transpose_mcast:
                        core_coord = ttnn.CoreCoord(core_id, 0)
                    else:
                        core_coord = ttnn.CoreCoord(0, core_id)
                else:
                    core_coord = ttnn.CoreCoord(core_id % num_cores_w, core_id // num_cores_w)

                # HACK: Using first device which may have different harvesting than other chips. Logic should be pushed into op
                if isinstance(device, ttnn.Device):
                    worker_core = device.worker_core_from_logical_core(core_coord)
                else:
                    worker_core = device.get_device(0).worker_core_from_logical_core(core_coord)
                return (worker_core.x, worker_core.y)

            remote_read = act_reshard_num_cores_nhw > 0

            (
                padding_config,
                local_config,
                remote_config,
                max_out_nsticks_per_core,
            ) = generate_untilize_with_halo_kernel_configs(
                tensor_metadata,
                req_conv_input_shard_start_end,
                core_id_to_physical_coord,
                remote_read=remote_read,
            )

            padding_config_tensor = gen_per_core_gather_data_uint16_tensor(padding_config)
            local_config_tensor = gen_per_core_gather_data_uint16_tensor(local_config)
            remote_config_tensor = gen_per_core_gather_data_uint16_tensor(remote_config)

            # shard_shape[1] filled in with incoming activations in c++ code
            out_shard_shape = [untilize_with_halo_input_shard_nhw_size, 0]

            halo_reader_patterns_cache[sliding_window_op_params_hash] = {
                "max_out_nsticks_per_core": max_out_nsticks_per_core,
                "padding_config": padding_config_tensor,
                "local_config": local_config_tensor,
                "remote_config": remote_config_tensor,
                "out_mem_config": get_memory_config(out_shard_shape, buffer_type=ttnn.BufferType.L1),
                "remote_read": remote_read,
            }

        return

    def __call__(self, activation):
        return self.utwh(activation)
