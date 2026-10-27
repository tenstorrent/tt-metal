// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_kv_cache_load_slice_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts
NlpKVCacheLoadSliceDeviceOperation::NlpKVCacheLoadSliceProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& output_tensor_start = operation_attributes.output_tensor_start;

    const auto& output_shape = output.padded_shape();
    const auto& input_shape = a.padded_shape();

    // This should allocate a DRAM buffer on the device
    auto shard_spec = output.shard_spec().value();
    auto all_cores = shard_spec.grid;
    auto num_cores_total = all_cores.num_cores();
    auto core_range = *all_cores.ranges().begin();
    auto num_cores_x = core_range.grid_size().x;
    uint32_t num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
    auto num_tiles_per_core = num_units_per_shard_height * num_units_per_shard_width;

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // The Metal 2.0 binding layer works with the Metalium tensor type; extract once.
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(data_format);

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    // The output shard, borrowed in place: the reader fills it, the writer only waits on it.
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    uint32_t num_input_tiles = num_tiles_per_core;
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = data_format,
        .borrowed_from = OUTPUT,
    };

    // Shared reader and writer config setup
    uint32_t num_unpadded_tiles_head_dim = output_shape[-1] / TILE_WIDTH;
    uint32_t num_unpadded_tiles_seqlen_dim = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_tiles_seqlen_dim =
        (input_shape[-2] / TILE_HEIGHT - num_unpadded_tiles_seqlen_dim) * (input_shape[-1] / TILE_WIDTH);

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/device/kernels/dataflow/"
            "reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "src",
                },
            },
        // Reader compile-time args
        .compile_time_args =
            {
                {"num_tiles", num_tiles_per_core},
                {"num_unpadded_tiles_head_dim", num_unpadded_tiles_head_dim},
                {"num_unpadded_tiles_seqlen_dim", num_unpadded_tiles_seqlen_dim},
                {"num_padded_tiles_seqlen_dim", num_padded_tiles_seqlen_dim},
                {"num_readers", num_cores_total},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    uint32_t start_id = ttnn::operations::data_movement::get_tiled_start_offset(a, output_tensor_start);
    const uint32_t num_tiles_shifted_per_core = input_shape[-2] * input_shape[-1] / TILE_HW;

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};

        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"start_id", start_id}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_units", num_tiles_per_core}});

        start_id += num_tiles_shifted_per_core;
    }

    ProgramSpec spec{
        .name = "nlp_kv_cache_load_slice",
        .kernels = {reader, writer},
        .dataflow_buffers = {out_dfb},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                // Borrow-only: backs the out DFB; no kernel binds it as an accessor.
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "all_cores",
                    .kernels = {READER, WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
