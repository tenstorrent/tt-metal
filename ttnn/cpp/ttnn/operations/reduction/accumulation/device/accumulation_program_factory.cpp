// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation/device/accumulation_device_operation_types.hpp"
#include "accumulation_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

#include "tt-metalium/base_types.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/work_split.hpp>

#include <bit>
#include <map>
#include <string>
#include <utility>

namespace ttnn::prim {

using AccumulationProgramFactory = AccumulationDeviceOperation::AccumulationProgramFactory;

namespace {

using namespace tt::tt_metal::experimental;

// The constants below carry a per-factory prefix so this file and the sibling EMA factory can
// safely share a translation unit: both are sources of the unity-built ttnn_op_reduction target,
// which merges their anonymous namespaces. The spec-name strings they hold are scoped to one
// ProgramSpec, so those need no prefix and are identical in both factories.
const KernelSpecName ACCUM_READER{"reader"};
const KernelSpecName ACCUM_WRITER{"writer"};
// The compute kernel is instantiated once per work-split core group. The two instances are
// identical apart from the nodes they run on; the per-core work count is a runtime argument.
const KernelSpecName ACCUM_COMPUTE_G1{"compute_g1"};
const KernelSpecName ACCUM_COMPUTE_G2{"compute_g2"};

// SRC carries input tiles from the reader to compute; DST carries results from compute to the
// writer. ACC holds the running accumulation: the compute kernel packs the partial total out and
// reads it straight back on the next iteration, so it is both producer and consumer. Its
// single-entry depth is deliberate, so that every reserve_back lands on the same address.
const DFBSpecName ACCUM_SRC{"src"};
const DFBSpecName ACCUM_DST{"dst"};
const DFBSpecName ACCUM_ACC{"acc"};

const TensorParamName ACCUM_INPUT{"input"};
const TensorParamName ACCUM_OUTPUT{"output"};

}  // namespace

// calculate the offset between consecutive tiles between accumulation axis and last dimension
uint32_t AccumulationProgramFactory::calc_input_tile_offset(
    const Shape& input_shape, const int32_t& dim, uint32_t tile_height, uint32_t tile_width) {
    uint32_t input_tile_offset{1};
    for (int32_t i = dim + 1; i < input_shape.rank() - 2; ++i) {
        input_tile_offset *= input_shape[i];
    }
    if (input_shape.rank() > 1) {
        input_tile_offset *= (input_shape[-2] / tile_height);
    }
    if (input_shape.rank() > 0) {
        input_tile_offset *= (input_shape[-1] / tile_width);
    }

    return input_tile_offset;
}

ttnn::device_operation::ProgramArtifacts AccumulationProgramFactory::create_program_artifacts(
    const AccumulationParams& operation_attributes,
    const AccumulationInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input_tensor{tensor_args.input_tensor.mesh_tensor()};
    const auto& output_tensor{tensor_return_value.mesh_tensor()};
    const auto& input_shape{input_tensor.padded_shape()};

    IDevice* device{&input_tensor.mutable_device()};

    const auto dst_cb_data_format{datatype_to_dataformat_converter(output_tensor.dtype())};

    const uint32_t input_rank{input_tensor.padded_shape().rank()};

    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;
    TT_FATAL(num_cores_y != 0, "Compute grid y-dimension must be non-zero");

    const int32_t dim{
        (operation_attributes.dim >= 0) ? operation_attributes.dim : (input_rank + operation_attributes.dim)};

    const auto& tile = input_tensor.tensor_spec().tile();
    // how many tiles along accumulation axis
    const uint32_t tiles_per_row{input_tensor.padded_shape()[dim]};
    TT_FATAL(tiles_per_row != 0, "tiles_per_row must be non-zero (got 0 for dim={})", dim);
    // all work units (product of all row lengths besides the accumulation row)
    const uint32_t num_rows_total{input_tensor.physical_volume() / tile.get_tile_hw() / tiles_per_row};
    // tiles between consecutive tiles along accumulation row
    const uint32_t input_tile_offset{calc_input_tile_offset(input_shape, dim, tile.get_height(), tile.get_width())};
    TT_FATAL(input_tile_offset != 0, "input_tile_offset must be non-zero (got 0 for dim={})", dim);

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_rows_total);

    TT_FATAL(
        num_cores > 0,
        "Accumulation (cumsum/cumprod) requires at least one worker core; num_rows_total={}",
        num_rows_total);

    validate_reduce_op_program_grid("Accumulation", all_cores, grid, nullptr, true, {{&tensor_return_value, "output"}});

    constexpr uint32_t in_tiles = 4;
    constexpr uint32_t acc_tiles = 1;
    constexpr uint32_t out_tiles = 4;

    auto acc_dataformat = datatype_to_dataformat_converter(output_tensor.dtype());
    if (!is_integer_format(acc_dataformat)) {
        acc_dataformat = DataFormat::Float32;
    }
    auto acc_dataformat_name = fmt::format("DataFormat::{}", acc_dataformat);

    const auto input_dataformat = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_dataformat = datatype_to_dataformat_converter(output_tensor.dtype());

    auto make_dfb = [](const DFBSpecName& unique_id, const tt::DataFormat& data_format, uint32_t num_entries) {
        return DataflowBufferSpec{
            .unique_id = unique_id,
            .entry_size = tt::tile_size(data_format),
            .num_entries = num_entries,
            .data_format_metadata = data_format,
        };
    };

    // The accumulator is read back at full 32-bit precision, so it is unpacked straight into Dest
    // rather than through SrcA/B. The input takes the same route whenever it is not the format the
    // FPU path handles natively. Omitting a DFB is the UnpackToSrc default; the output DFB is only
    // produced into, never consumed, so it needs no entry.
    ComputeUnpackModes unpack_modes;
    unpack_modes[ACCUM_ACC] = UnpackMode::UnpackToDest;
    if (input_dataformat != DataFormat::Float16_b) {
        unpack_modes[ACCUM_SRC] = UnpackMode::UnpackToDest;
    }

    std::map<std::string, std::string> defines_kernel_args = {};

    if (is_integer_format(dst_cb_data_format)) {
        defines_kernel_args["BINARY_OP_INIT"] = operation_attributes.op == AccumulationOp::CUMSUM
                                                    ? "add_int_tile_init"
                                                    : fmt::format("mul_int_tile_init<{}>", acc_dataformat_name);
        defines_kernel_args["BINARY_OP"] = operation_attributes.op == AccumulationOp::CUMSUM
                                               ? fmt::format("add_int_tile<{}>", acc_dataformat_name)
                                               : fmt::format("mul_int_tile<{}>", acc_dataformat_name);
        defines_kernel_args["FILL_TILE"] = fmt::format("fill_tile_int<{}>", acc_dataformat_name);
    } else {
        defines_kernel_args["BINARY_OP_INIT"] =
            operation_attributes.op == AccumulationOp::CUMSUM ? "add_binary_tile_init" : "mul_binary_tile_init";
        defines_kernel_args["BINARY_OP"] =
            operation_attributes.op == AccumulationOp::CUMSUM ? "add_binary_tile" : "mul_binary_tile";
        defines_kernel_args["FILL_TILE"] = "fill_tile_bitcast";
    }

    float default_acc_value = 0.f;
    if (operation_attributes.op == AccumulationOp::CUMPROD) {
        default_acc_value = 1.f;
        if (is_integer_format(dst_cb_data_format)) {
            // Kernel reinterprets the 4-byte CT arg as int32 in the integer path; pack the bit
            // pattern 0x00000001 so it lands as integer 1, not float 1.0f's bit pattern.
            default_acc_value = std::bit_cast<float>(1U);
        }
    }

    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // fp32_dest_acc_en will be True for FLOAT32 inputs (set below), so use HiFi3 as default on Wormhole B0.
    const auto is_wormhole = device->arch() == tt::ARCH::WORMHOLE_B0;
    const auto default_math_fidelity =
        (is_wormhole && output_tensor.dtype() == DataType::FLOAT32) ? MathFidelity::HiFi3 : MathFidelity::HiFi4;

    // The reader and the writer address the same tile sequence, one on the input and one on the
    // output, so they take the same argument list.
    const Group<std::string> dataflow_rta_names{
        "num_rows_per_core",
        "tiles_per_row",
        "input_tile_offset",
        "start_id",
        "low_rank_offset",
        "high_rank_offset",
        "flip"};

    KernelSpec reader{
        .unique_id = ACCUM_READER,
        .source = AccumulationProgramFactory::KERNEL_PATHS[0],
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = ACCUM_SRC,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = ACCUM_INPUT,
            .accessor_name = "input",
        }},
        .runtime_arg_schema = {.runtime_arg_names = dataflow_rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = ACCUM_WRITER,
        .source = AccumulationProgramFactory::KERNEL_PATHS[2],
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = ACCUM_DST,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = ACCUM_OUTPUT,
            .accessor_name = "output",
        }},
        .runtime_arg_schema = {.runtime_arg_names = dataflow_rta_names},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    const ComputeHardwareConfig compute_config{
        .fpu_math_fidelity = default_math_fidelity,
        .sfpu_precision_mode = Precision::Precise,
        .enable_32_bit_dest = true,
        .double_buffer_dest = true,
        .unpack_modes = unpack_modes,
    };

    const KernelSpec::CompilerOptions::Defines compute_defines(defines_kernel_args);

    auto make_compute = [&](const KernelSpecName& unique_id) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = AccumulationProgramFactory::KERNEL_PATHS[1],
            // O3 is the optimization level a compute kernel is built at; the CompilerOptions
            // default (O2) is the data-movement level, so compute kernels state it explicitly.
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = ACCUM_SRC,
                     .accessor_name = "in",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = ACCUM_DST,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 // Both ends of the accumulator FIFO belong to this one kernel.
                 DFBBinding{
                     .dfb_spec_name = ACCUM_ACC,
                     .accessor_name = "acc",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = ACCUM_ACC,
                     .accessor_name = "acc",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 }},
            .compile_time_args = {{"default_acc_value", std::bit_cast<uint32_t>(default_acc_value)}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "tiles_per_row"}},
            .hw_config = compute_config,
        };
    };

    const bool group_2_present = !core_group_2.ranges().empty();

    KernelRunArgs reader_run_args{.kernel = ACCUM_READER};
    KernelRunArgs writer_run_args{.kernel = ACCUM_WRITER};
    KernelRunArgs compute_g1_run_args{.kernel = ACCUM_COMPUTE_G1};
    KernelRunArgs compute_g2_run_args{.kernel = ACCUM_COMPUTE_G2};

    for (uint32_t i{0}, tile_offset = 0; i < num_cores; ++i) {
        NodeCoord node{i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(node)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(node)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            node,
            {{"num_rows_per_core", num_tiles_per_core},
             {"tiles_per_row", tiles_per_row},
             {"input_tile_offset", input_tile_offset},
             {"start_id", tile_offset},
             {"low_rank_offset", tile_offset / input_tile_offset},
             {"high_rank_offset", tile_offset % input_tile_offset},
             {"flip", static_cast<uint32_t>(operation_attributes.flip)}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            node,
            {{"num_rows_per_core", num_tiles_per_core},
             {"tiles_per_row", tiles_per_row},
             {"input_tile_offset", input_tile_offset},
             {"start_id", tile_offset},
             {"low_rank_offset", tile_offset / input_tile_offset},
             {"high_rank_offset", tile_offset % input_tile_offset},
             {"flip", static_cast<uint32_t>(operation_attributes.flip)}});

        if (core_group_1.contains(node)) {
            AddRuntimeArgsForNode(
                compute_g1_run_args.runtime_arg_values,
                node,
                {{"num_rows", num_tiles_per_core}, {"tiles_per_row", tiles_per_row}});
        } else if (core_group_2.contains(node)) {
            TT_ASSERT(group_2_present);
            AddRuntimeArgsForNode(
                compute_g2_run_args.runtime_arg_values,
                node,
                {{"num_rows", num_tiles_per_core}, {"tiles_per_row", tiles_per_row}});
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        tile_offset += num_tiles_per_core;
    }

    Group<KernelSpec> kernels{std::move(reader), std::move(writer), make_compute(ACCUM_COMPUTE_G1)};
    Group<KernelRunArgs> kernel_run_args{
        std::move(reader_run_args), std::move(writer_run_args), std::move(compute_g1_run_args)};
    // Reader and writer belong to every work unit, so their node set is the union of the core
    // groups, which is the grid the legacy descriptors named directly.
    Group<WorkUnitSpec> work_units{WorkUnitSpec{
        .name = "accumulation_group_1",
        .kernels = {ACCUM_READER, ACCUM_WRITER, ACCUM_COMPUTE_G1},
        .target_nodes = core_group_1,
    }};

    if (group_2_present) {
        kernels.push_back(make_compute(ACCUM_COMPUTE_G2));
        kernel_run_args.push_back(std::move(compute_g2_run_args));
        work_units.push_back(WorkUnitSpec{
            .name = "accumulation_group_2",
            .kernels = {ACCUM_READER, ACCUM_WRITER, ACCUM_COMPUTE_G2},
            .target_nodes = core_group_2,
        });
    }

    ProgramSpec spec{
        .name = "accumulation",
        .kernels = std::move(kernels),
        .dataflow_buffers =
            {make_dfb(ACCUM_SRC, input_dataformat, in_tiles),
             make_dfb(ACCUM_DST, output_dataformat, out_tiles),
             make_dfb(ACCUM_ACC, acc_dataformat, acc_tiles)},
        .tensor_parameters =
            {TensorParameter{.unique_id = ACCUM_INPUT, .spec = input_tensor.tensor_spec()},
             TensorParameter{.unique_id = ACCUM_OUTPUT, .spec = output_tensor.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args{
        .kernel_run_args = std::move(kernel_run_args),
        .tensor_args = {{ACCUM_INPUT, input_tensor}, {ACCUM_OUTPUT, output_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
