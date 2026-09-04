// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_program_factory.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeWHShardedProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Declared function-locally: this op's factories share one translation unit in the unity
    // build, so file-scope names would collide across them.
    const DFBSpecName IN0{"in0"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const auto& input_tensor = tensor_args.input;
    const auto& input = input_tensor.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    tt::DataFormat src0_dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_dfb_data_format);
    tt::DataFormat dst_dfb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_dfb_data_format);

    const auto tile = input_tensor.tensor_spec().tile();
    const uint32_t tile_hw = tile.get_tile_hw();

    IDevice* device = input_tensor.device();

    bool fp32_dest_acc_en = src0_dfb_data_format == tt::DataFormat::Float32;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto shard_spec = input_tensor.shard_spec().value();
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto& all_cores = shard_spec.grid;
    uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

    ProgramSpec spec{.name = "transpose_wh_sharded"};

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()});

    // Both buffers are built on the io tensors' own sharded L1 memory rather than on
    // Program-lifetime storage, so the compute kernel transposes the resident shard in place.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = src0_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = src0_dfb_data_format,
        .borrowed_from = INPUT,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT,
        .entry_size = dst_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = dst_dfb_data_format,
        .borrowed_from = OUTPUT,
    });

    // Reader and writer are shared Metal 2.0 forks owned by the eltwise/unary and
    // data_movement/sharded ops; their binding vocabulary and named argument sets are fixed
    // by those kernels.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = create_writer_datamovement_config(device->arch()),
    });

    // Legacy built a ComputeConfigDescriptor directly, setting only fp32_dest_acc_en and
    // unpack_to_dest_mode; every other field kept its Metal default, which ComputeGen1Config
    // reproduces.
    ComputeGen1Config compute_hw{.enable_32_bit_dest = fp32_dest_acc_en};
    if (src0_dfb_data_format == tt::DataFormat::Float32) {
        compute_hw.unpack_modes.insert({IN0, UnpackMode::UnpackToDest});
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/"
                  "transpose_wh_sharded_metal2.cpp",
        // Legacy left opt_level unset on a ComputeConfigDescriptor, which resolves to O3;
        // a Metal 2.0 KernelSpec defaults to O2, so the level is restated explicitly here.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = IN0,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = OUT,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .runtime_arg_schema = {.runtime_arg_names = {"NHtWt", "HtWt", "N", "Ht", "Wt"}},
        .hw_config = compute_hw,
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = total_cores,
    });

    auto padded_shape = input_tensor.padded_shape();
    auto shard_shape = shard_spec.shape;
    uint32_t H = padded_shape[2];
    uint32_t Hs = shard_shape[0], Ws = shard_shape[1];
    uint32_t Hts = Hs / tile.get_height();
    uint32_t Wts = Ws / tile.get_width();
    uint32_t Ht = H / tile.get_height();
    uint32_t Ht_per_shard = std::min(Ht, Hts);
    uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;
    uint32_t HtWt_tile_size = Ht_per_shard * Wts;
    uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

    auto bbox = all_cores.bounding_box();
    std::vector<CoreCoord> cores =
        grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);
    const uint32_t num_active = all_cores.num_cores();

    ProgramRunArgs run_args;
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER};
    ProgramRunArgs::KernelRunArgs compute_run_args{.kernel = COMPUTE};

    // Cores outside the shard grid still run the kernels; legacy handed them a zero-filled
    // argument set so they fall straight through their loops, and that is preserved here.
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const bool active = i < num_active;
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, cores[i], {{"num_tiles_per_core", active ? num_blocks : 0}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, cores[i], {{"num_units", active ? num_blocks : 0}});
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            cores[i],
            {{"NHtWt", active ? num_blocks : 0},
             {"HtWt", active ? HtWt_tile_size : 0},
             {"N", active ? num_hw_blocks_per_shard : 0},
             {"Ht", active ? Ht_per_shard : 0},
             {"Wt", active ? Wts : 0}});
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args.emplace(INPUT, input);
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
