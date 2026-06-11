// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <tt_stl/reflection.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "untilize_single_core_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {

// File-scope kernel paths. Names are unique across the untilize device/ sibling .cpp files to avoid
// unity-build collisions. These are op-private Metal 2.0 copies of the shared kernels (the legacy kernels
// are still consumed positionally by the un-migrated variants and by untilize_with_unpadding via the
// shared file paths, so they must not be touched).
constexpr const char* SINGLE_CORE_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id_m2.cpp";
constexpr const char* SINGLE_CORE_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
    "writer_unary_stick_layout_split_rows_single_core_m2.cpp";
constexpr const char* SINGLE_CORE_COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_m2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts UntilizeSingleCoreProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_volume = tile_height * tile_width;

    uint32_t num_tiles = a.physical_volume() / tile_volume;
    uint32_t num_blocks_across_height = a.physical_volume() / a.padded_shape()[-1] / tile_height;
    uint32_t num_columns_of_blocks = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        uint32_t output_shard_width;
        if (output.shard_spec().has_value()) {
            output_shard_width = output.shard_spec().value().shape[1];
        } else {
            output_shard_width = output.nd_shard_spec().value().shard_shape[-1];
        }
        num_columns_of_blocks = a.padded_shape()[-1] / output_shard_width;
    }
    uint32_t num_tiles_per_column_row = a.padded_shape()[-1] / num_columns_of_blocks / tile_width;

    // Determine how much L1 space we can use for input and output CBs,
    // ensuring that we don't intrude into other L1 storage space
    uint32_t max_l1_size =
        (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Determine the max number of tiles that can be in any CB at a given time (1 input CB + 1 output CB = 2 total CBs)
    uint32_t max_tiles_per_cb = max_l1_size / (input_single_tile_size + output_single_tile_size);

    // Determine how many tiles each block will store.
    // Currently we require that the number of tiles in a row is divisible by the number of blocks in a row, or
    // equivalently the number of tiles in a row is divisible by the number of tiles in a block.
    uint32_t num_tiles_per_block = num_tiles_per_column_row;
    if (num_tiles_per_block > max_tiles_per_cb) {
        for (uint32_t i = max_tiles_per_cb; i > 0; --i) {
            if (num_tiles_per_column_row % i == 0) {
                num_tiles_per_block = i;
                break;
            }
        }
    }

    uint32_t num_blocks_per_column_row = num_tiles_per_column_row / num_tiles_per_block;
    uint32_t output_single_block_width_size = num_tiles_per_block * TILE_WIDTH * output.element_size();
    uint32_t num_blocks = num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height;

    const m2::DFBSpecName input_dfb{"src0"};
    const m2::DFBSpecName output_dfb{"out0"};

    m2::ProgramSpec spec;
    spec.name = "untilize_single_core";

    // Input CB and Output CB: both L1-allocated, sized to one block of tiles (formerly CBs c_0 / c_16).
    uint32_t input_cb_num_tiles = num_tiles_per_block;
    uint32_t output_cb_num_tiles = num_tiles_per_block;
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = input_dfb,
            .entry_size = input_single_tile_size,
            .num_entries = input_cb_num_tiles,
            .data_format_metadata = input_cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = output_dfb,
            .entry_size = output_single_tile_size,
            .num_entries = output_cb_num_tiles,
            .data_format_metadata = output_cb_data_format,
        }};

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    // Tilized reader on NCRISC (RISCV_1 / NOC1): reads interleaved input tiles into src0.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{SINGLE_CORE_READER_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(input_dfb, "cb_id_in0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_page_id"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    // Untilized writer on BRISC (default Gen1 config): consumes out0, writes row-major sticks to output.
    // The (formerly positional) compile-time scalars become named compile-time args.
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{SINGLE_CORE_WRITER_KERNEL_PATH},
        .dfb_bindings = {m2::ConsumerOf(output_dfb, "cb_id_out0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .compile_time_args =
            {{"tile_height", tile_height},
             {"num_blocks_across_height", num_blocks_across_height},
             {"num_output_columns_of_blocks", num_columns_of_blocks},
             {"num_blocks_per_output_column_row", num_blocks_per_column_row},
             {"num_tiles_per_output_block", num_tiles_per_block},
             {"output_single_block_width_size", output_single_block_width_size}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    // Compute kernel: consumes src0, produces out0. DST_ACCUM_MODE mirrors the legacy define for 32-bit
    // formats; UnpackToDestFp32 on the input DFB when fp32_dest_acc_en (formerly the per-CB
    // unpack_to_dest_mode vector entry on src0).
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    m2::ComputeHardwareConfig compute_hw{.fp32_dest_acc_en = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        compute_hw.unpack_to_dest_mode.emplace(input_dfb, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
    }
    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{SINGLE_CORE_COMPUTE_KERNEL_PATH},
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings = {m2::ConsumerOf(input_dfb, "src_cb_id"), m2::ProducerOf(output_dfb, "out_cb_id")},
        .compile_time_args = {{"per_core_block_cnt", num_blocks}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config = std::move(compute_hw),
    };

    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};

    // Local DFBs (src0, out0) require their producer AND consumer kernels co-located in the same
    // WorkUnitSpec on each node: reader produces src0 (consumed by compute), compute produces out0
    // (consumed by writer). All three run on the single core.
    spec.work_units = {m2::WorkUnitSpec{
        .name = "untilize_single_core",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
        .target_nodes = core}};

    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    // Reader run-time args: start at page 0, read all tiles. The writer takes only compile-time scalars,
    // but it still binds the `output` tensor parameter, whose address the framework fills from this
    // kernel's KernelRunArgs entry — so the writer needs an (otherwise-empty) per-core entry too. The
    // compute kernel binds no tensor and takes only compile-time args, so it gets no KernelRunArgs entry.
    reader_args.runtime_arg_values.push_back(
        {CoreCoord{0, 0}, {{"num_tiles", num_tiles}, {"start_page_id", uint32_t{0}}}});
    writer_args.runtime_arg_values.push_back({CoreCoord{0, 0}, {}});
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));

    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(a.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace ttnn::prim
