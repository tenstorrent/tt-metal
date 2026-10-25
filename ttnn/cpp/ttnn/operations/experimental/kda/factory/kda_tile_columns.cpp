// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/factory/kda_tile_columns.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim::kda_factory_detail {

namespace {

namespace m2 = tt::tt_metal::experimental;

// Tile rows of a TILE tensor, every leading dimension folded in.
uint32_t tile_rows(const Tensor& tensor) {
    return tensor.physical_volume() / tensor.padded_shape()[-1] / tt::constants::TILE_HEIGHT;
}

uint32_t row_tiles(const Tensor& tensor) { return tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH; }

// The output's tile rows and tile columns in its [rows, columns] view (head-major outputs fold their heads in).
uint32_t output_tile_rows(const TileColumnActivationParams& params, const Tensor& output) {
    return params.output_head_tiles == 0 ? tile_rows(output)
                                         : tile_rows(output) / static_cast<uint32_t>(output.padded_shape()[0]);
}

uint32_t output_row_tiles(const TileColumnActivationParams& params, const Tensor& output) {
    return params.output_head_tiles == 0 ? row_tiles(output)
                                         : row_tiles(output) * static_cast<uint32_t>(output.padded_shape()[0]);
}

}  // namespace

void check_tile_column_activation(
    const TileColumnActivationParams& params,
    const Tensor& input,
    const Tensor& output,
    std::string_view operation_name) {
    using ttnn::operations::unary::utils::uses_dedicated_compute_kernel;
    constexpr std::array accepted = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32};
    for (const auto& [tensor, name] :
         {std::pair{&input, "column activation input"}, std::pair{&output, "column activation output"}}) {
        check_allocated_device_tensor(*tensor, operation_name, name);
        check_layout(*tensor, tt::tt_metal::Layout::TILE, operation_name, name);
        check_dtype_in(*tensor, accepted, "BFLOAT16 or FLOAT32", operation_name, name);
        check_same_device(input, *tensor, operation_name, name);
    }
    check_interleaved(input, operation_name, "column activation input");
    TT_FATAL(
        params.output_head_tiles == 0 ||
            (output.padded_shape().rank() == 3 && row_tiles(output) == params.output_head_tiles),
        "{}: a head-major column activation output must be [heads, rows, 32 * {}], got {}",
        operation_name,
        params.output_head_tiles,
        output.padded_shape());
    TT_FATAL(params.columns > 0, "{}: the column activation needs at least one tile column", operation_name);
    TT_FATAL(
        tile_rows(input) == output_tile_rows(params, output),
        "{}: the column activation's input and output have {} and {} tile rows",
        operation_name,
        tile_rows(input),
        output_tile_rows(params, output));
    TT_FATAL(
        params.input_column + params.columns <= row_tiles(input) &&
            params.output_column + params.columns <= output_row_tiles(params, output),
        "{}: tile columns [{}, {}) of the input ({} wide) or [{}, {}) of the output ({} wide) leave the tensor",
        operation_name,
        params.input_column,
        params.input_column + params.columns,
        row_tiles(input),
        params.output_column,
        params.output_column + params.columns,
        output_row_tiles(params, output));
    for (const auto& op : params.op_chain) {
        // The chain runs in the generic unary kernel's SFPU_OP_CHAIN_0.
        TT_FATAL(
            !uses_dedicated_compute_kernel(op.type(), input.dtype()),
            "{}: the column activation's {} needs a dedicated unary kernel",
            operation_name,
            op.type());
    }
    TT_FATAL(
        !params.compute_kernel_config.packer_l1_acc,
        "{}: packer_l1_acc=true is unsupported by the column activation",
        operation_name);
}

void add_tile_column_activation(
    m2::ProgramSpec& spec,
    m2::ProgramRunArgs& run_args,
    const TileColumnActivationParams& params,
    const Tensor& input,
    const Tensor& output,
    std::span<const tt::tt_metal::CoreCoord> cores) {
    const uint32_t tiles = tile_rows(input) * params.columns;
    TT_FATAL(!cores.empty(), "The column activation needs at least one core");
    const auto used = cores.first(std::min<size_t>(cores.size(), tiles));
    const auto core_set = merged_core_set(used);

    const m2::KernelSpecName reader_name{"tile_columns_reader"};
    const m2::KernelSpecName writer_name{"tile_columns_writer"};
    const m2::KernelSpecName compute_name{"tile_columns_compute"};
    const m2::DFBSpecName in_dfb{"tile_columns_in"};
    const m2::DFBSpecName out_dfb{"tile_columns_out"};
    const m2::TensorParamName input_name{"tile_columns_input"};
    const m2::TensorParamName output_name{"tile_columns_output"};

    const auto input_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    // Double-buffered single tiles: the chain's SFPU sets the pace, the reader and writer keep a tile ahead of it.
    for (const auto& [name, format] : {std::pair{in_dfb, input_format}, std::pair{out_dfb, output_format}}) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = name, .entry_size = tt::tile_size(format), .num_entries = 2, .data_format_metadata = format});
    }
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = input_name, .spec = input.mesh_tensor().tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = output_name, .spec = output.mesh_tensor().tensor_spec()});
    run_args.tensor_args.emplace(input_name, input.mesh_tensor());
    run_args.tensor_args.emplace(output_name, output.mesh_tensor());

    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/dataflow/";
    spec.kernels.push_back(m2::KernelSpec{
        .unique_id = reader_name,
        .source = kernel_dir + "reader_tile_columns.cpp",
        .dfb_bindings = {m2::DFBBinding{in_dfb, "in", m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{input_name, "input"}},
        .compile_time_args =
            {{"columns", params.columns}, {"row_tiles", row_tiles(input)}, {"first_column", params.input_column}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_tile", "num_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });
    spec.kernels.push_back(m2::KernelSpec{
        .unique_id = writer_name,
        .source = kernel_dir + "writer_tile_columns.cpp",
        .dfb_bindings = {m2::DFBBinding{out_dfb, "out", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{output_name, "output"}},
        .compile_time_args =
            {{"columns", params.columns},
             {"row_tiles", row_tiles(output)},
             {"first_column", params.output_column},
             {"head_tiles", params.output_head_tiles},
             {"head_rows", output_tile_rows(params, output)}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_tile", "num_tiles"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    // The generic unary kernel and its chain defines, as ttnn's unary ops build them.
    std::map<std::string, std::string> defines =
        ttnn::operations::unary::utils::get_block_defines(params.op_chain, "0", "0", input.dtype());
    ttnn::operations::unary::utils::add_input_dtype_defines(input.dtype(), defines);
    auto compute_hw = ttnn::to_compute_hardware_config(params.compute_kernel_config);
    if (input_format == tt::DataFormat::Float32) {
        // ttnn::unary's rule (unary.cpp sets preserve_fp32_precision exactly for a FLOAT32 input, and the unary
        // program factory then unpacks it to DST under fp32_dest_acc_en, else to the source registers): an FP32 input
        // reaches a 32-bit DST unrounded, a 16-bit one through the source registers. Unlike ttnn::unary, which forces
        // fp32_dest_acc_en on for a FLOAT32 input or output, the DST width is the caller's compute_kernel_config. A
        // BFLOAT16 input takes the default (source registers) either way, as in ttnn::unary.
        compute_hw.unpack_modes[in_dfb] = params.compute_kernel_config.fp32_dest_acc_en
                                              ? tt::tt_metal::UnpackMode::UnpackToDest
                                              : tt::tt_metal::UnpackMode::UnpackToSrc;
    }
    spec.kernels.push_back(m2::KernelSpec{
        .unique_id = compute_name,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu_metal2.cpp",
        .compiler_options =
            {.defines = m2::Table<std::string, std::string>(defines),
             .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {m2::DFBBinding{in_dfb, "in", m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{out_dfb, "out", m2::DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = std::move(compute_hw),
    });
    spec.work_units.push_back(m2::WorkUnitSpec{
        .name = "tile_columns", .kernels = {reader_name, writer_name, compute_name}, .target_nodes = core_set});

    m2::KernelRunArgs reader_args{.kernel = reader_name};
    m2::KernelRunArgs writer_args{.kernel = writer_name};
    m2::KernelRunArgs compute_args{.kernel = compute_name};
    const uint32_t base = tiles / used.size();
    const uint32_t remainder = tiles % used.size();
    uint32_t first = 0;
    for (uint32_t index = 0; index < used.size(); ++index) {
        const uint32_t count = base + (index < remainder ? 1u : 0u);
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values, used[index], {{"first_tile", first}, {"num_tiles", count}});
        m2::AddRuntimeArgsForNode(
            writer_args.runtime_arg_values, used[index], {{"first_tile", first}, {"num_tiles", count}});
        m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, used[index], {{"num_tiles", count}});
        first += count;
    }
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.kernel_run_args.push_back(std::move(compute_args));
}

std::vector<tt::tt_metal::CoreCoord> idle_cores(tt::tt_metal::CoreCoord grid, const tt::tt_metal::CoreRangeSet& busy) {
    std::vector<tt::tt_metal::CoreCoord> idle;
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            if (!busy.contains(tt::tt_metal::CoreCoord{x, y})) {
                idle.emplace_back(x, y);
            }
        }
    }
    return idle;
}

}  // namespace ttnn::experimental::prim::kda_factory_detail
