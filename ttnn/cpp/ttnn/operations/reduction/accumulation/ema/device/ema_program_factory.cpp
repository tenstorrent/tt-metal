// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ema_device_operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <bit>
#include <utility>

namespace ttnn::prim {

using namespace tt::tt_metal;

constexpr auto ema_buffer_depth = 2;

namespace {

using namespace tt::tt_metal::experimental;

// The constants below carry a per-factory prefix so this file and the sibling accumulation factory
// can safely share a translation unit: both are sources of the unity-built ttnn_op_reduction
// target, which merges their anonymous namespaces. The spec-name strings they hold are scoped to
// one ProgramSpec, so those need no prefix and are identical in both factories.
const KernelSpecName EMA_READER{"reader"};
const KernelSpecName EMA_WRITER{"writer"};
const KernelSpecName EMA_COMPUTE{"compute"};

// src carries input tiles from the reader to compute; dst carries results from compute to the
// writer. prev is a round trip through the packer: compute packs a transposed tile out and reads it
// straight back to transpose again, so it is both producer and consumer of that buffer.
const DFBSpecName EMA_SRC{"src"};
const DFBSpecName EMA_DST{"dst"};
const DFBSpecName EMA_PREV{"prev"};

const TensorParamName EMA_INPUT{"input"};
const TensorParamName EMA_OUTPUT{"output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts EmaDeviceOperation::EmaProgramFactory::create_program_artifacts(
    const EmaParams& operation_attributes, const EmaInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::tt_metal::experimental;

    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    IDevice* device = &input.mutable_device();

    // Grid sizing
    // -----------
    // If empty grid size, use all cores
    auto grid_size = operation_attributes.grid_size;
    if ((grid_size.x == 0) && (grid_size.y == 0)) {
        grid_size = device->compute_with_storage_grid_size();
    }
    auto num_cores_available = grid_size.x * grid_size.y;

    // Compute total_tiles to determine core split
    auto input_shape = input.padded_shape();
    auto num_batches = input_shape[1];
    auto num_channels = input_shape[2];
    auto num_samples_per_channel = input_shape[3];

    auto num_channel_tiles = num_channels / input.tensor_spec().tile().get_height();
    auto tiles_per_channel = num_samples_per_channel / input.tensor_spec().tile().get_width();

    auto total_batch_channel_tiles = num_batches * num_channel_tiles;

    // We pick the maximum number of cores (from the available) that divides total_tiles equally
    auto [num_cores, total_batch_channel_tiles_per_core] = get_max_cores_divisible_by_tiles_per_core_tiles(
        total_batch_channel_tiles, num_cores_available, /*request_even=*/false);

    // We now have the number of cores to use, compute per core parameters
    auto all_cores = CoreRangeSet(grid_to_cores(num_cores, grid_size.x, grid_size.y, false));

    validate_reduce_op_program_grid(
        "EMA", all_cores, device->compute_with_storage_grid_size(), nullptr, false, {{&tensor_return_value, "output"}});
    log_debug(
        tt::LogOp,
        "EmaProgramFactory: grid_size=({}, {}), num_cores={}, total_batch_channel_tiles={}",
        grid_size.y,
        grid_size.x,
        num_cores,
        total_batch_channel_tiles);

    auto total_tiles_per_core = total_batch_channel_tiles_per_core * tiles_per_channel;

    // Precompute the alpha and beta bits
    // Used by the EMA SFPU instructions
    // ----------------------------------
    auto alpha_bits = std::bit_cast<uint32_t>(operation_attributes.alpha);
    auto beta_bits = std::bit_cast<uint32_t>(1.0f - operation_attributes.alpha);

    // Dataflow buffer specs
    // ---------------------
    auto src_data_format = datatype_to_dataformat_converter(input.dtype());
    auto dst_data_format = datatype_to_dataformat_converter(output.dtype());

    auto src_tile_size = input.tensor_spec().tile().get_tile_size(src_data_format);
    auto dst_tile_size = output.tensor_spec().tile().get_tile_size(dst_data_format);

    DataflowBufferSpec src_dfb{
        .unique_id = EMA_SRC,
        .entry_size = src_tile_size,
        .num_entries = ema_buffer_depth,
        .data_format_metadata = src_data_format,
    };

    DataflowBufferSpec dst_dfb{
        .unique_id = EMA_DST,
        .entry_size = dst_tile_size,
        .num_entries = ema_buffer_depth,
        .data_format_metadata = dst_data_format,
    };

    DataflowBufferSpec prev_dfb{
        .unique_id = EMA_PREV,
        .entry_size = src_tile_size,
        .num_entries = 1,
        .data_format_metadata = src_data_format,
    };

    // Create kernel specs
    // -------------------
    // This factory targets Gen1 (Wormhole / Blackhole) only. The two data movement kernels place
    // the reader on RISCV_0 and the writer on RISCV_1, which is the reverse of the conventional
    // assignment, so neither matches a role default and neither can go through the
    // architecture-agnostic reader / writer helpers without changing where they run. Spelling out
    // the Gen1 config is therefore what preserves the placement, and it pins this whole factory to
    // Gen1: a Gen2 build would need placement decisions that cannot be derived from these values.
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    KernelSpec reader{
        .unique_id = EMA_READER,
        .source = "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_reader.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = EMA_SRC,
            .accessor_name = "src",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = EMA_INPUT,
            .accessor_name = "src",
        }},
        .compile_time_args = {{"total_tiles_per_core", total_tiles_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"src_start_tile"}},
        .hw_config = DataMovementHardwareConfig{DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = reader_noc,
        }},
    };

    KernelSpec writer{
        .unique_id = EMA_WRITER,
        .source = "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_writer.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = EMA_DST,
            .accessor_name = "dst",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = EMA_OUTPUT,
            .accessor_name = "dst",
        }},
        .compile_time_args = {{"total_tiles_per_core", total_tiles_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_start_tile"}},
        .hw_config = DataMovementHardwareConfig{DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = writer_noc,
        }},
    };

    KernelSpec compute{
        .unique_id = EMA_COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/compute/ema_compute.cpp",
        // O3 is the optimization level a compute kernel is built at; the CompilerOptions default
        // (O2) is the data-movement level, so compute kernels state it explicitly.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = EMA_SRC,
                 .accessor_name = "src",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = EMA_DST,
                 .accessor_name = "dst",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             // Both ends of the transpose round trip belong to this one kernel.
             DFBBinding{
                 .dfb_spec_name = EMA_PREV,
                 .accessor_name = "trp",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = EMA_PREV,
                 .accessor_name = "trp",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             }},
        .compile_time_args =
            {{"total_batches_per_core", total_batch_channel_tiles_per_core},
             {"tiles_per_channel", tiles_per_channel},
             {"alpha_bits", alpha_bits},
             {"beta_bits", beta_bits}},
        // Translates the TTNN ComputeKernelConfig this op resolves into its Metal 2.0 equivalent.
        // The helper picks the alternative matching the architecture, but that does not make the
        // program portable: the data movement kernels above are Gen1-only, so the whole factory is.
        .hw_config = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config),
    };

    // Set runtime args
    // ---------------

    KernelRunArgs reader_run_args{.kernel = EMA_READER};
    KernelRunArgs writer_run_args{.kernel = EMA_WRITER};

    uint32_t src_start_tile = 0;
    uint32_t dst_start_tile = 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& node : range) {
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, node, {{"src_start_tile", src_start_tile}});
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, node, {{"dst_start_tile", dst_start_tile}});
            src_start_tile += total_tiles_per_core;
            dst_start_tile += total_tiles_per_core;
        }
    }

    ProgramSpec spec{
        .name = "ema",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = {std::move(src_dfb), std::move(dst_dfb), std::move(prev_dfb)},
        .tensor_parameters =
            {TensorParameter{.unique_id = EMA_INPUT, .spec = input.tensor_spec()},
             TensorParameter{.unique_id = EMA_OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "ema",
            .kernels = {EMA_READER, EMA_WRITER, EMA_COMPUTE},
            .target_nodes = all_cores,
        }},
    };

    // The compute kernel takes no runtime arguments, so it contributes no KernelRunArgs entry.
    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args = {{EMA_INPUT, input}, {EMA_OUTPUT, output}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
