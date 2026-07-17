// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include "untilize_multi_core_program_factory.hpp"
#include "ttnn/operations/experimental/quasar/untilize/device/untilize_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

// Metal 2.0 port of the multi-core untilize factory. Three input modes (selected on the host):
//   (C) interleaved      -> reader_unary_start_id_metal2 (TensorAccessor read)
//   (A) block-reader     -> reader_unary_sharded_blocks_metal2 (sharded L1/DRAM, used for uneven/DRAM)
//   (B) even sharded     -> reader_unary_sharded_metal2 + input DFB borrowed_from the input shard buffer
// Writer and compute are common. Host work-distribution is preserved verbatim from the legacy factory.
ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& output) {
    const auto& a = tensor_args.input;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    IDevice* device = a.device();
    Buffer* src0_buffer = a.buffer();

    uint32_t tensor_width = a.padded_shape()[-1];
    uint32_t tensor_height = a.physical_volume() / tensor_width;

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    bool input_is_sharded = a.is_sharded();
    std::vector<CoreCoord> ordered_cores_with_data;

    uint32_t num_tiles_per_row = tensor_width / tile_width;
    uint32_t num_tiles_per_col = tensor_height / tile_height;

    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_compute_cores,
         compute_core_range,
         full_compute_core_range,
         cliff_compute_core_range,
         num_rows_per_full_core,
         num_rows_per_cliff_core] = ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);

    // Default values are for interleaved input.
    uint32_t num_input_blocks_across_width = 1;
    uint32_t num_tiles_per_input_block = num_tiles_per_row;
    uint32_t num_input_blocks_per_full_core = num_rows_per_full_core;
    uint32_t num_input_blocks_per_cliff_core = num_rows_per_cliff_core;
    uint32_t input_shard_height = 0;
    uint32_t input_shard_width = 0;
    if (input_is_sharded) {
        ShardSpec input_shard_spec = a.shard_spec().value();
        input_shard_height = input_shard_spec.shape[0];
        input_shard_width = input_shard_spec.shape[1];
        num_compute_cores = input_shard_spec.grid.num_cores();

        num_input_blocks_across_width = tt::div_up(tensor_width, input_shard_width);
        num_tiles_per_input_block = input_shard_width / tile_width;
        num_input_blocks_per_full_core = input_shard_height / tile_height;
        num_input_blocks_per_cliff_core = 0;

        ordered_cores_with_data = get_optimal_worker_cores_for_sharded_tensor(a);
        compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));
        full_compute_core_range = compute_core_range;
        cliff_compute_core_range = CoreRangeSet();
    }

    bool has_uneven_sharding = false;
    if (input_is_sharded) {
        uint32_t height_remainder = tensor_height % input_shard_height;
        uint32_t width_remainder = tensor_width % input_shard_width;
        has_uneven_sharding = (height_remainder != 0) || (width_remainder != 0);
    }

    const bool input_is_dram_sharded = input_is_sharded && src0_buffer->buffer_type() == BufferType::DRAM;
    // Block reader: streams from L1/DRAM shard block-by-block (uneven or DRAM sharding).
    // Even sharding uses a zero-copy DFB borrowed from the shard buffer.
    bool use_block_reader = input_is_sharded && (has_uneven_sharding || input_is_dram_sharded);
    bool use_backed_cb = input_is_sharded && !use_block_reader;

    // Input CB sizing (mirrors legacy).
    uint32_t input_cb_num_tiles;
    if (use_backed_cb) {
        input_cb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core;
    } else {
        input_cb_num_tiles =
            (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    }
    uint32_t output_cb_num_tiles =
        (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;

    // Writer geometry (mirrors legacy).
    uint32_t output_element_size = output.element_size();
    uint32_t output_page_width = tensor_width;
    uint32_t output_num_blocks_across_width = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        if (output.shard_spec().has_value()) {
            output_page_width = output.shard_spec().value().shape[1];
        } else {
            output_page_width = output.nd_shard_spec().value().shard_shape[-1];
        }
        output_num_blocks_across_width = tt::div_up(tensor_width, output_page_width);
    }
    uint32_t num_cols_per_input_block = num_tiles_per_input_block * tile_width;
    uint32_t num_cols_per_output_block = output_page_width;

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};    // legacy c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy c_16
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

    // ---- DataflowBuffers. Even-sharded input borrows the input shard buffer's L1 (zero-copy). ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = input_cb_num_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    if (use_backed_cb) {
        in_dfb.borrowed_from = INPUT;
    }
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = output_cb_num_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reader KernelSpec (mode-specific) ----
    const std::filesystem::path kdir("ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/dataflow/");
    KernelSpec reader;
    reader.unique_id = READER;
    reader.dfb_bindings = {
        DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}};
    reader.hw_config = ttnn::create_reader_datamovement_config(device->arch());
    if (use_block_reader) {
        reader.source = kdir / "reader_unary_sharded_blocks_metal2.cpp";
        reader.tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}};
        reader.compile_time_args = {{"tiles_per_block", num_tiles_per_input_block}};
        reader.runtime_arg_schema = {.runtime_arg_names = {"start_shard_id", "num_blocks"}};
    } else if (use_backed_cb) {
        // Even sharded: no TensorAccessor read; the DFB is borrowed from the input buffer.
        reader.source = kdir / "reader_unary_sharded_metal2.cpp";
        reader.runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}};
    } else {
        // Interleaved input.
        reader.source = kdir / "reader_unary_start_id_metal2.cpp";
        reader.tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}};
        reader.runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_page_id"}};
    }

    // ---- Writer KernelSpec (common). Dead legacy CTA output_stick_size dropped. ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir / "writer_unary_stick_layout_split_rows_multi_core_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"tile_height", tile_height},
             {"num_tiles_per_input_block", num_tiles_per_input_block},
             {"num_output_blocks_across_width", output_num_blocks_across_width},
             {"output_element_size", output_element_size},
             {"num_cols_per_input_block", num_cols_per_input_block},
             {"num_cols_per_output_block", num_cols_per_output_block}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_input_blocks_to_process",
                  "height_wise_input_block_start_index",
                  "num_unpadded_cols_per_input_block",
                  "width_wise_output_block_start_index",
                  "num_cols_already_processed_in_first_output_block"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute KernelSpec(s) (common; full + optional interleaved cliff) ----
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    auto make_compute_hw = [&]() -> ComputeHardwareConfig {
        ttnn::ComputeKernelConfig hw{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
        ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(device->arch(), hw);
        if (fp32_dest_acc_en) {
            std::visit(
                [&](auto& c) { c.unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest); }, compute_hw);
        }
        return compute_hw;
    };
    const std::filesystem::path compute_source(
        "ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/compute/"
        "untilize_variable_num_blocks_metal2.cpp");
    auto make_compute = [&](const KernelSpecName& id) {
        return KernelSpec{
            .unique_id = id,
            .source = compute_source,
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args = {{"per_core_block_tile_cnt", num_tiles_per_input_block}},
            .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt"}},
            .hw_config = make_compute_hw(),
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    const bool has_full = !full_compute_core_range.ranges().empty();
    const bool has_cliff = !cliff_compute_core_range.ranges().empty();
    if (has_full) {
        kernels.push_back(make_compute(COMPUTE_FULL));
        work_units.push_back(WorkUnitSpec{
            .name = "untilize_mc_full",
            .kernels = {READER, WRITER, COMPUTE_FULL},
            .target_nodes = full_compute_core_range});
    }
    if (has_cliff) {
        kernels.push_back(make_compute(COMPUTE_CLIFF));
        work_units.push_back(WorkUnitSpec{
            .name = "untilize_mc_cliff",
            .kernels = {READER, WRITER, COMPUTE_CLIFF},
            .target_nodes = cliff_compute_core_range});
    }

    // ---- Per-core runtime args (mirrors legacy work-distribution loop verbatim) ----
    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;
    KernelRunArgs::RuntimeArgValues compute_full_node_args;
    KernelRunArgs::RuntimeArgValues compute_cliff_node_args;

    uint32_t tile_start_index = 0;
    bool is_row_major = input_is_sharded ? a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR : true;
    std::vector<CoreCoord> full_cores = input_is_sharded
                                            ? ordered_cores_with_data
                                            : corerange_to_cores(full_compute_core_range, std::nullopt, is_row_major);

    auto push_reader_args = [&](const CoreCoord& core,
                                uint32_t core_index,
                                uint32_t num_input_blocks_to_process,
                                uint32_t num_tiles_to_read) {
        if (use_block_reader) {
            AddRuntimeArgsForNode(
                reader_node_args,
                core,
                {
                    {"start_shard_id", core_index},
                    {"num_blocks", num_input_blocks_to_process},
                });
        } else if (use_backed_cb) {
            reader_node_args["num_tiles_per_core"][core] = num_tiles_to_read;
        } else {
            AddRuntimeArgsForNode(
                reader_node_args,
                core,
                {
                    {"num_tiles", num_tiles_to_read},
                    {"start_page_id", tile_start_index},
                });
        }
    };

    for (uint32_t i = 0; i < full_cores.size(); ++i) {
        CoreCoord core = full_cores[i];
        uint32_t height_wise_input_block_start_index =
            (i / num_input_blocks_across_width) * num_input_blocks_per_full_core;
        uint32_t width_wise_input_block_index = i % num_input_blocks_across_width;

        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;
        if (input_is_sharded) {
            bool is_last_input_shard_in_row = width_wise_input_block_index == num_input_blocks_across_width - 1;
            if (is_last_input_shard_in_row) {
                uint32_t shard_width = a.shard_spec().value().shape[1];
                num_unpadded_cols_per_input_block =
                    num_cols_per_input_block - (tt::round_up(tensor_width, shard_width) - tensor_width);
            }
        }

        uint32_t num_input_blocks_to_process = num_input_blocks_per_full_core;
        if (input_is_sharded) {
            uint32_t shard_height = a.shard_spec().value().shape[0];
            uint32_t height_wise_shard_index = i / num_input_blocks_across_width;
            uint32_t num_shards_height_wise = tt::div_up(tensor_height, shard_height);
            bool is_last_input_shard_in_col = height_wise_shard_index == num_shards_height_wise - 1;
            if (is_last_input_shard_in_col) {
                num_input_blocks_to_process = num_input_blocks_per_full_core -
                                              (tt::round_up(tensor_height, shard_height) - tensor_height) / tile_height;
            }
        }

        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        push_reader_args(core, i, num_input_blocks_to_process, num_tiles_to_read);

        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        AddRuntimeArgsForNode(
            writer_node_args,
            core,
            {
                {"num_input_blocks_to_process", num_input_blocks_to_process},
                {"height_wise_input_block_start_index", height_wise_input_block_start_index},
                {"num_unpadded_cols_per_input_block", num_unpadded_cols_per_input_block},
                {"width_wise_output_block_start_index", width_wise_output_block_start_index},
                {"num_cols_already_processed_in_first_output_block", num_cols_already_processed_in_first_output_block},
            });

        if (has_full) {
            compute_full_node_args["per_core_block_cnt"][core] = num_input_blocks_to_process;
        }

        tile_start_index += num_tiles_per_input_block * num_input_blocks_per_full_core;
    }

    // Cliff core (interleaved input only).
    std::vector<CoreCoord> cliff_cores = corerange_to_cores(cliff_compute_core_range, std::nullopt, is_row_major);
    if (!cliff_cores.empty()) {
        CoreCoord cliff_core = cliff_cores[0];
        uint32_t height_wise_input_block_start_index = full_cores.size() * num_input_blocks_per_full_core;
        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;
        uint32_t num_input_blocks_to_process = num_input_blocks_per_cliff_core;

        // Cliff core (interleaved only) always starts at the first output block, column 0.
        uint32_t width_wise_output_block_start_index = 0;
        uint32_t num_cols_already_processed_in_first_output_block = 0;
        AddRuntimeArgsForNode(
            writer_node_args,
            cliff_core,
            {
                {"num_input_blocks_to_process", num_input_blocks_to_process},
                {"height_wise_input_block_start_index", height_wise_input_block_start_index},
                {"num_unpadded_cols_per_input_block", num_unpadded_cols_per_input_block},
                {"width_wise_output_block_start_index", width_wise_output_block_start_index},
                {"num_cols_already_processed_in_first_output_block", num_cols_already_processed_in_first_output_block},
            });

        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        // Cliff core only exists for interleaved input.
        AddRuntimeArgsForNode(
            reader_node_args,
            cliff_core,
            {
                {"num_tiles", num_tiles_to_read},
                {"start_page_id", tile_start_index},
            });

        if (has_cliff) {
            compute_cliff_node_args["per_core_block_cnt"][cliff_core] = num_input_blocks_to_process;
        }
    }

    ProgramSpec spec{
        .name = "untilize_multi_core",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    std::vector<KernelRunArgs> kra;
    kra.push_back(KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)});
    kra.push_back(KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)});
    if (has_full) {
        kra.push_back(KernelRunArgs{.kernel = COMPUTE_FULL, .runtime_arg_values = std::move(compute_full_node_args)});
    }
    if (has_cliff) {
        kra.push_back(KernelRunArgs{.kernel = COMPUTE_CLIFF, .runtime_arg_values = std::move(compute_cliff_node_args)});
    }
    run_args.kernel_run_args = std::move(kra);
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
