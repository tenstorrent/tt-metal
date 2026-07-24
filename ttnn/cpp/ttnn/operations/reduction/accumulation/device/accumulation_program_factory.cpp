// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation/device/accumulation_device_operation_types.hpp"
#include "accumulation_device_operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

#include "tt-metalium/base_types.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <bit>
#include <map>
#include <string>

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

using AccumulationProgramFactory = AccumulationDeviceOperation::AccumulationProgramFactory;

namespace {

// Metal 2.0 named resource handles for the accumulation ProgramSpec.
const DFBSpecName SRC_DFB{"src"};
const DFBSpecName ACC_DFB{"acc"};
const DFBSpecName DST_DFB{"dst"};

const TensorParamName INPUT_TENSOR{"input"};
const TensorParamName OUTPUT_TENSOR{"output"};

const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName WRITER_KERNEL{"writer"};
const KernelSpecName COMPUTE_KERNEL_G1{"compute_g1"};
const KernelSpecName COMPUTE_KERNEL_G2{"compute_g2"};

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

ttnn::device_operation::ProgramArtifacts AccumulationProgramFactory::create_program_spec(
    const AccumulationParams& operation_attributes,
    const AccumulationInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

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

    // ----------------------------------------------------------------------------
    // Dataflow buffers (formerly CB_IN / CB_OUT / CB_ACC). One DFB per legacy CB.
    // entry_size / num_entries are fixed at spec construction (no per-execution override).
    // ----------------------------------------------------------------------------
    DataflowBufferSpec src_dfb_spec{
        .unique_id = SRC_DFB,
        .entry_size = tt::tile_size(input_dataformat),
        .num_entries = in_tiles,
        .data_format_metadata = input_dataformat,
    };
    DataflowBufferSpec acc_dfb_spec{
        .unique_id = ACC_DFB,
        .entry_size = tt::tile_size(acc_dataformat),
        .num_entries = acc_tiles,
        .data_format_metadata = acc_dataformat,
    };
    DataflowBufferSpec dst_dfb_spec{
        .unique_id = DST_DFB,
        .entry_size = tt::tile_size(output_dataformat),
        .num_entries = out_tiles,
        .data_format_metadata = output_dataformat,
    };

    // ----------------------------------------------------------------------------
    // Compute unpack_to_dest_mode (per-DFB). Legacy set ACC→Fp32 always, SRC→Fp32
    // unless input is Float16_b. The helper to_compute_hardware_config does NOT cover
    // unpack_to_dest_mode, so it is configured here directly (per the recipe's
    // hardware-config note).
    //
    // The Metal 2.0 spec validator (program_spec.cpp:819) rejects UnpackToDestFp32 on a
    // DFB whose data format is not Float32 — the mode is only meaningful for FP32 data.
    // The legacy code set UnpackToDestFp32 unconditionally (e.g. on the integer ACC/SRC
    // buffers for int32 cumsum/cumprod), which the legacy CB path never validated and the
    // LLK ignored for integer formats. Preserve the *meaningful* legacy behavior by
    // gating each entry on the DFB's actual format being Float32. (See PORT_REPORT.)
    // ----------------------------------------------------------------------------
    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (acc_dataformat == DataFormat::Float32) {
        unpack_to_dest_modes.insert({ACC_DFB, UnpackToDestMode::UnpackToDestFp32});
    }
    if (input_dataformat == DataFormat::Float32) {
        unpack_to_dest_modes.insert({SRC_DFB, UnpackToDestMode::UnpackToDestFp32});
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

    // ----------------------------------------------------------------------------
    // Tensor parameters: input (reader) and output (writer). Both Case 1 — the
    // legacy buffer-address RTA + TensorAccessorArgs plumbing collapse to a binding.
    // ----------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_tensor.tensor_spec()};

    // ----------------------------------------------------------------------------
    // Reader / writer kernels (single instance across all_cores).
    // ----------------------------------------------------------------------------
    const std::vector<std::string> dataflow_rta_names{
        "num_rows_per_core",
        "tiles_per_row",
        "input_tile_offset",
        "start_id",
        "low_rank_offset",
        "high_rank_offset",
        "flip"};

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{AccumulationProgramFactory::KERNEL_PATHS[0]},
        .dfb_bindings = {ProducerOf(SRC_DFB, "src")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = dataflow_rta_names},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementHardwareConfig::RoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{AccumulationProgramFactory::KERNEL_PATHS[2]},
        .dfb_bindings = {ConsumerOf(DST_DFB, "dst")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = dataflow_rta_names},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementHardwareConfig::RoleHint::WRITER},
    };

    // ----------------------------------------------------------------------------
    // Compute kernel: one KernelSpec per legacy compute KernelDescriptor (per core
    // group), preserving the work-split multiplicity. ACC is a self-loop binding
    // (PRODUCER + CONSUMER on the same KernelSpec).
    // ----------------------------------------------------------------------------
    KernelSpec::CompilerOptions::Defines compute_defines{defines_kernel_args};

    auto make_compute_spec = [&](const KernelSpecName& unique_id) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = std::filesystem::path{AccumulationProgramFactory::KERNEL_PATHS[1]},
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {ConsumerOf(SRC_DFB, "src"),
                 ProducerOf(DST_DFB, "dst"),
                 ProducerOf(ACC_DFB, "acc"),
                 ConsumerOf(ACC_DFB, "acc")},
            .compile_time_args = {{"default_acc_value", std::bit_cast<uint32_t>(default_acc_value)}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "tiles_per_row"}},
            .hw_config =
                ComputeHardwareConfig{
                    .math_fidelity = default_math_fidelity,
                    .fp32_dest_acc_en = true,
                    .dst_full_sync_en = false,
                    .math_approx_mode = false,
                    .unpack_to_dest_mode = unpack_to_dest_modes,
                },
        };
    };

    KernelSpec compute_spec_g1 = make_compute_spec(COMPUTE_KERNEL_G1);
    const bool group_2_present = !core_group_2.ranges().empty();
    KernelSpec compute_spec_g2 = group_2_present ? make_compute_spec(COMPUTE_KERNEL_G2) : KernelSpec{};

    // ----------------------------------------------------------------------------
    // Per-node runtime arg values (paired with the schemas above).
    // ----------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs compute_run_g1{.kernel = COMPUTE_KERNEL_G1};
    KernelRunArgs compute_run_g2{.kernel = COMPUTE_KERNEL_G2};

    const uint32_t flip = static_cast<uint32_t>(operation_attributes.flip);

    for (uint32_t i{0}, tile_offset = 0; i < num_cores; ++i) {
        NodeCoord core{i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        const uint32_t low_rank_offset = tile_offset / input_tile_offset;
        const uint32_t high_rank_offset = tile_offset % input_tile_offset;

        const KernelRunArgs::RuntimeArgValues dataflow_args{
            {"num_rows_per_core", num_tiles_per_core},
            {"tiles_per_row", tiles_per_row},
            {"input_tile_offset", input_tile_offset},
            {"start_id", tile_offset},
            {"low_rank_offset", low_rank_offset},
            {"high_rank_offset", high_rank_offset},
            {"flip", flip}};

        reader_run.runtime_arg_values.push_back({core, dataflow_args});
        writer_run.runtime_arg_values.push_back({core, dataflow_args});

        const KernelRunArgs::RuntimeArgValues compute_args{
            {"num_rows", num_tiles_per_core}, {"tiles_per_row", tiles_per_row}};

        if (core_group_1.contains(core)) {
            compute_run_g1.runtime_arg_values.push_back({core, compute_args});
        } else if (core_group_2.contains(core)) {
            TT_ASSERT(group_2_present);
            compute_run_g2.runtime_arg_values.push_back({core, compute_args});
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        tile_offset += num_tiles_per_core;
    }

    // ----------------------------------------------------------------------------
    // Work units: one per core group, each running reader + writer + the group's
    // compute kernel. Reader/writer are members of both WUs (single KernelSpec).
    // ----------------------------------------------------------------------------
    WorkUnitSpec wu_g1{
        .name = "accumulation_g1",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G1},
        .target_nodes = core_group_1,
    };

    ProgramSpec spec{
        .name = "accumulation",
        .kernels = {reader_spec, writer_spec, compute_spec_g1},
        .dataflow_buffers = {src_dfb_spec, acc_dfb_spec, dst_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu_g1},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, compute_run_g1};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor)}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor)}}};

    if (group_2_present) {
        spec.kernels.push_back(compute_spec_g2);
        spec.work_units.push_back(WorkUnitSpec{
            .name = "accumulation_g2",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G2},
            .target_nodes = core_group_2,
        });
        run_args.kernel_run_args.push_back(compute_run_g2);
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
