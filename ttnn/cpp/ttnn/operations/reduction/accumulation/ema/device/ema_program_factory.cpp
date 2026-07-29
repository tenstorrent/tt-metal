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
using namespace tt::tt_metal::experimental;

constexpr auto ema_buffer_depth = 2;

ttnn::device_operation::ProgramArtifacts EmaDeviceOperation::EmaProgramFactory::create_program_artifacts(
    const EmaParams& operation_attributes, const EmaInputs& tensor_args, Tensor& tensor_return_value) {
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

    // Program-scope resource names
    // ----------------------------
    // Each resource is declared once here and bound by name on the kernels that use it; the kernel
    // sees a generated handle (dfb::<accessor_name>, tensor::<accessor_name>) rather than an index or
    // an address. The program-scope id keeps the host's word for the resource and the accessor name
    // keeps the kernel's, which for the transpose staging buffer are different words: the host calls
    // it "prev", the kernel "trp".
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName SRC{"src"};
    const DFBSpecName DST{"dst"};
    const DFBSpecName PREV{"prev"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ProgramSpec spec;
    spec.name = "ema";

    // Dataflow buffer config
    // ----------------------
    // Placement is derived from the kernel bindings below, so no node range is declared here.
    auto src_data_format = datatype_to_dataformat_converter(input.dtype());
    auto dst_data_format = datatype_to_dataformat_converter(output.dtype());

    auto src_tile_size = input.tensor_spec().tile().get_tile_size(src_data_format);
    auto dst_tile_size = output.tensor_spec().tile().get_tile_size(dst_data_format);

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SRC,
        .entry_size = src_tile_size,
        .num_entries = ema_buffer_depth,
        .data_format_metadata = src_data_format,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DST,
        .entry_size = dst_tile_size,
        .num_entries = ema_buffer_depth,
        .data_format_metadata = dst_data_format,
    });

    // The compute kernel round-trips one tile through this buffer to transpose it back, so it holds a
    // single tile rather than a double-buffered pair.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = PREV,
        .entry_size = src_tile_size,
        .num_entries = 1,
        .data_format_metadata = src_data_format,
    });

    // Tensor parameters
    // -----------------
    // The kernels reach the input and output through these; no buffer address travels as an argument.
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    // Create kernel specs
    // -------------------
    // This op inverts the conventional RISC assignment (the reader runs on RISCV_0 and the writer on
    // RISCV_1) while keeping the conventional NOC assignment, so neither DM kernel matches a
    // reader/writer default and both configs are spelled out field by field.
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_reader.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC,
            .accessor_name = "src",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = INPUT,
            .accessor_name = "src",
        }},
        .compile_time_args = {{"total_tiles_per_core", total_tiles_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"src_start_tile"}},
        .hw_config =
            DataMovementGen1Config{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = reader_noc,
                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
            },
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_writer.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = DST,
            .accessor_name = "dst",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "dst",
        }},
        .compile_time_args = {{"total_tiles_per_core", total_tiles_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_start_tile"}},
        .hw_config =
            DataMovementGen1Config{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = writer_noc,
                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
            },
    });

    // PREV is bound twice on the compute kernel, as both endpoints: the compute kernel is the buffer's
    // only toucher and drives both FIFO ends itself, packing a tile in and unpacking it back out to
    // transpose it. Both bindings share one accessor name, so the kernel drives them through a single
    // DataflowBuffer object.
    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/compute/ema_compute.cpp",
        // Compute kernels build at O3; the KernelSpec default is O2, so it is stated here.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = SRC,
                 .accessor_name = "src",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = DST,
                 .accessor_name = "dst",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = PREV,
                 .accessor_name = "trp",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = PREV,
                 .accessor_name = "trp",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             }},
        .compile_time_args =
            {{"total_batches_per_core", total_batch_channel_tiles_per_core},
             {"tiles_per_channel", tiles_per_channel},
             {"alpha_bits", alpha_bits},
             {"beta_bits", beta_bits}},
        .hw_config = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config),
    });

    // Kernel placement
    // ----------------
    // All three kernels run on every node, which also satisfies the local dataflow buffer rule that a
    // buffer's producer and consumer share work-unit membership.
    spec.work_units.push_back(
        WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores});

    // Set runtime args
    // ---------------
    // The compute kernel reads no runtime args, so it gets no entry here.
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    uint32_t src_start_tile = 0;
    uint32_t dst_start_tile = 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"src_start_tile", src_start_tile}});
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"dst_start_tile", dst_start_tile}});
            src_start_tile += total_tiles_per_core;
            dst_start_tile += total_tiles_per_core;
        }
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(INPUT, input);
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
