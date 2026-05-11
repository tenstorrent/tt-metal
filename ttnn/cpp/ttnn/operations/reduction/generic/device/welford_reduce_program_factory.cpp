// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford reduction program factory, migrated to the Metal 2.0 host API.
//
// Welford handles three reduce_dim variants (W, H, HW) with shared per-dim
// machinery: reader + writer + a Welford-specific compute kernel + dim-specific
// scratch DFBs. Each variant produces a single ProgramSpec.
//
// Sharded inputs are not supported on this path; sharded Welford reductions
// still go through the Gen1 pipeline upstream. Tensor base addresses are
// resolved via Metal 2.0 TensorAccessor bindings (see
// ProgramSpec::tensor_parameters and KernelSpec::tensor_bindings); is_dram
// and aligned_page_size are no longer needed as kernel-side arguments.

#include "welford_reduce_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "reduce_metal2_factory_helpers.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "welford_reduce_device_operation.hpp"

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

constexpr const char* WELFORD_READER_KERNEL = "welford_reader";
constexpr const char* WELFORD_WRITER_KERNEL = "welford_writer";
constexpr const char* WELFORD_COMPUTE_KERNEL = "welford_compute";
constexpr const char* WELFORD_WORK_UNIT = "all_workers";
constexpr const char* WELFORD_INPUT_TENSOR = "input_tensor";
constexpr const char* WELFORD_OUTPUT_TENSOR = "output_tensor";

// Welford-specific DFB ids (in addition to INPUT_DFB / SCALER_DFB / OUTPUT_DFB
// from the shared header).
constexpr const char* VAR_DFB = "var";            // W-reduce only — variance scratch tile
constexpr const char* SCALED_DFB = "scaled";      // W-reduce only — scaled input tile (always bound)
constexpr const char* PARTIAL_DFB = "partial";    // HW-reduce only — per-column mean/var partials
constexpr const char* COMBINED_DFB = "combined";  // HW-reduce only — combined scalar tile

struct WelfordWorkDistribution {
    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
    uint32_t num_work_units_per_core_group_1 = 0;
    uint32_t num_work_units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

WelfordWorkDistribution ComputeWelfordWorkDistribution(
    const WelfordReduceParams& attrs, const tt::tt_metal::Tensor& input, uint32_t num_work_units) {
    using namespace tt::tt_metal;

    auto* device = input.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    WelfordWorkDistribution wd;
    if (attrs.sub_core_grids.has_value()) {
        std::tie(
            wd.num_cores,
            wd.all_cores,
            wd.core_group_1,
            wd.core_group_2,
            wd.num_work_units_per_core_group_1,
            wd.num_work_units_per_core_group_2) = split_work_to_cores(*attrs.sub_core_grids, num_work_units);
        for (const auto& range : wd.all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    wd.cores.emplace_back(x, y);
                }
            }
        }
    } else {
        std::tie(
            wd.num_cores,
            wd.all_cores,
            wd.core_group_1,
            wd.core_group_2,
            wd.num_work_units_per_core_group_1,
            wd.num_work_units_per_core_group_2) = split_work_to_cores(compute_with_storage_grid_size, num_work_units);
        wd.cores =
            grid_to_cores(wd.num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }
    return wd;
}

m2::ProgramRunParams BuildRunParams(
    const WelfordReduceSharedVariables& shared,
    const tt::tt_metal::MeshTensor& input_mt,
    const tt::tt_metal::MeshTensor& output_mt) {
    using tt::tt_metal::ReduceOpDim;

    m2::ProgramRunParams params;

    m2::ProgramRunParams::KernelRunParams reader_params;
    reader_params.kernel_spec_name = WELFORD_READER_KERNEL;

    m2::ProgramRunParams::KernelRunParams writer_params;
    writer_params.kernel_spec_name = WELFORD_WRITER_KERNEL;

    m2::ProgramRunParams::KernelRunParams compute_params;
    compute_params.kernel_spec_name = WELFORD_COMPUTE_KERNEL;

    const bool reduce_w = (shared.reduce_dim == ReduceOpDim::W);
    const bool reduce_hw = (shared.reduce_dim == ReduceOpDim::HW);
    const uint32_t Wt = shared.Wt;
    const uint32_t HtWt = shared.HtWt;

    if (reduce_w) {
        // W-reduce: each work unit is one row of Wt tiles.
        uint32_t input_tiles_offset = 0;
        uint32_t output_tiles_offset = 0;
        for (const auto& core : shared.cores) {
            uint32_t num_work_units_per_core = 0;
            if (shared.core_group_1.contains(core)) {
                num_work_units_per_core = shared.num_work_units_per_core_group_1;
            } else if (shared.core_group_2.contains(core)) {
                num_work_units_per_core = shared.num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            const uint32_t num_input_tiles_per_core = num_work_units_per_core * Wt;
            const uint32_t num_output_tiles_per_core = num_work_units_per_core;

            reader_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args =
                    {
                        {"num_tiles", num_input_tiles_per_core},
                        {"start_id", input_tiles_offset},
                    },
            });
            writer_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args =
                    {
                        {"num_pages", num_output_tiles_per_core},
                        {"start_id", output_tiles_offset},
                    },
            });
            compute_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"NCHt", num_work_units_per_core}},
            });

            input_tiles_offset += num_input_tiles_per_core;
            output_tiles_offset += num_output_tiles_per_core;
        }
    } else if (reduce_hw) {
        // HW-reduce: each work unit is one output element from reduce_batch_size NC slices.
        uint32_t nc_slice_offset = 0;
        uint32_t output_offset = 0;
        for (const auto& core : shared.cores) {
            uint32_t num_outputs_per_core = 0;
            if (shared.core_group_1.contains(core)) {
                num_outputs_per_core = shared.num_work_units_per_core_group_1;
            } else if (shared.core_group_2.contains(core)) {
                num_outputs_per_core = shared.num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            const uint32_t nc_slices_per_core = num_outputs_per_core * shared.reduce_batch_size;
            const uint32_t num_cols = Wt * nc_slices_per_core;
            const uint32_t col_start_tile_id = nc_slice_offset * HtWt;

            reader_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args =
                    {
                        {"col_start_tile_id", col_start_tile_id},
                        {"curr_col_in_batch", 0u},
                        {"num_cols", num_cols},
                    },
            });
            // HW writer is welford-specific (writer_welford_hw.cpp): args are
            // {NC_per_core, output_tile_start_id} (tensor binding supplies dst_addr).
            writer_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args =
                    {
                        {"NC_per_core", nc_slices_per_core},
                        {"output_tile_start_id", output_offset},
                    },
            });
            compute_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"NC_per_core", nc_slices_per_core}},
            });

            nc_slice_offset += nc_slices_per_core;
            output_offset += num_outputs_per_core;
        }
    } else {
        // H-reduce: each work unit is one column of Ht tiles.
        uint32_t num_cols_read = 0;
        for (const auto& core : shared.cores) {
            uint32_t num_cols_per_core = 0;
            if (shared.core_group_1.contains(core)) {
                num_cols_per_core = shared.num_work_units_per_core_group_1;
            } else if (shared.core_group_2.contains(core)) {
                num_cols_per_core = shared.num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            const uint32_t col_start_tile_id = (num_cols_read / Wt) * HtWt + (num_cols_read % Wt);
            const uint32_t curr_col_in_batch = num_cols_read % Wt;

            reader_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args =
                    {
                        {"col_start_tile_id", col_start_tile_id},
                        {"curr_col_in_batch", curr_col_in_batch},
                        {"num_cols", num_cols_per_core},
                    },
            });
            writer_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args =
                    {
                        {"num_pages", num_cols_per_core},
                        {"start_id", num_cols_read},
                    },
            });
            compute_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"NCWt", num_cols_per_core}},
            });

            num_cols_read += num_cols_per_core;
        }
    }

    params.kernel_run_params = {std::move(reader_params), std::move(writer_params), std::move(compute_params)};
    params.tensor_args = {
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = WELFORD_INPUT_TENSOR, .tensor = std::cref(input_mt)},
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = WELFORD_OUTPUT_TENSOR, .tensor = std::cref(output_mt)},
    };
    return params;
}

}  // namespace

WelfordReduceProgramFactory::cached_program_t WelfordReduceProgramFactory::create(
    const WelfordReduceParams& operation_attributes,
    const tt::tt_metal::Tensor& tensor_args,
    tt::tt_metal::Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args;
    auto& output = tensor_return_value;

    const Shape& padded_shape = a.padded_shape();
    const Shape& logical_shape = a.logical_shape();

    const uint32_t W = logical_shape[-1];
    const uint32_t H = logical_shape[-2];
    const uint32_t W_padded = padded_shape[-1];
    const uint32_t H_padded = padded_shape[-2];
    TT_FATAL(
        H_padded > 0 && W_padded > 0,
        "Padded H and W dimensions must be non-zero, got H_padded={}, W_padded={}",
        H_padded,
        W_padded);
    const uint32_t NC = a.physical_volume() / (H_padded * W_padded);
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    const uint32_t Wt = W_padded / tile_width;
    const uint32_t Ht = H_padded / tile_height;
    const uint32_t HtWt = Ht * Wt;

    const ReduceOpDim reduce_dim = operation_attributes.reduce_dim;
    const bool reduce_w = (reduce_dim == ReduceOpDim::W);
    const bool reduce_h = (reduce_dim == ReduceOpDim::H);
    const bool reduce_hw = (reduce_dim == ReduceOpDim::HW);
    const uint32_t reduce_batch_size = operation_attributes.reduce_batch_size;

    if (reduce_hw) {
        if (operation_attributes.correction) {
            TT_FATAL(
                H * W * reduce_batch_size >= 2,
                "Bessel's correction requires at least 2 elements across all reduction dimensions, got {}",
                H * W * reduce_batch_size);
        }
    } else {
        if (operation_attributes.correction) {
            const uint32_t reduce_size = reduce_w ? W : H;
            TT_FATAL(
                reduce_size >= 2,
                "Bessel's correction requires at least 2 elements along the reduction dimension, got {}",
                reduce_size);
        }
    }

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);
    (void)math_approx_mode;
    (void)packer_l1_acc;

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_single_tile_size = tile_size(input_cb_data_format);

    // Scalar datatype is hardcoded bfloat16 due to tile creation in reader.
    const tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    const uint32_t scalar_single_tile_size = tile_size(scalar_cb_data_format);
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t dst_single_tile_size = tile_size(dst_cb_data_format);

    const uint32_t num_work_units = reduce_w ? (NC * Ht) : (reduce_hw ? (NC / reduce_batch_size) : (NC * Wt));
    if (reduce_hw) {
        TT_FATAL(
            NC % reduce_batch_size == 0, "NC ({}) must be divisible by reduce_batch_size ({})", NC, reduce_batch_size);
    }
    const WelfordWorkDistribution wd = ComputeWelfordWorkDistribution(operation_attributes, a, num_work_units);

    const uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scalar);
    const bool do_scale = (operation_attributes.scalar != 1.0f);
    const bool is_std = (operation_attributes.math_op == ReduceOpMath::STD);

    // ---- DFBs ----
    constexpr uint32_t kNumInputEntries = 2;
    constexpr uint32_t kNumScalerEntries = 2;
    constexpr uint32_t kNumOutputEntries = 2;
    constexpr uint32_t kNumScratchEntries = 2;
    constexpr uint32_t kNumPartialEntries = 4;  // HW: 4-tile depth for double-buffered (mean,var) pair pushes

    std::vector<m2::DataflowBufferSpec> dataflow_buffers = {
        MakeDFB(INPUT_DFB, input_single_tile_size, kNumInputEntries, input_cb_data_format, a.tensor_spec().tile()),
        MakeDFB(SCALER_DFB, scalar_single_tile_size, kNumScalerEntries, scalar_cb_data_format, a.tensor_spec().tile()),
        MakeDFB(OUTPUT_DFB, dst_single_tile_size, kNumOutputEntries, dst_cb_data_format, output.tensor_spec().tile()),
    };

    if (reduce_w) {
        const tt::DataFormat var_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        const uint32_t var_single_tile_size = tile_size(var_data_format);
        dataflow_buffers.insert(
            dataflow_buffers.end(),
            {
                MakeIntraDFB(
                    VAR_DFB, var_single_tile_size, kNumScratchEntries, var_data_format, a.tensor_spec().tile()),
                MakeIntraDFB(
                    SCALED_DFB,
                    input_single_tile_size,
                    kNumScratchEntries,
                    input_cb_data_format,
                    a.tensor_spec().tile()),
            });
    }
    if (reduce_hw) {
        constexpr tt::DataFormat partial_data_format = tt::DataFormat::Float32;
        const uint32_t partial_single_tile_size = tile_size(partial_data_format);
        dataflow_buffers.insert(
            dataflow_buffers.end(),
            {
                MakeDFB(
                    PARTIAL_DFB,
                    partial_single_tile_size,
                    kNumPartialEntries,
                    partial_data_format,
                    a.tensor_spec().tile()),
                MakeDFB(
                    COMBINED_DFB,
                    partial_single_tile_size,
                    kNumScratchEntries,
                    partial_data_format,
                    a.tensor_spec().tile()),
            });
    }

    std::map<std::string, std::string> reduce_defines_map =
        reduce_op_utils::get_defines(operation_attributes.math_op, reduce_dim);
    reduce_defines_map["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    reduce_defines_map["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
    const auto reduce_defines = DefinesFromMap(reduce_defines_map);

    // ---- Reader ----
    m2::KernelSpec reader;
    reader.unique_id = WELFORD_READER_KERNEL;
    reader.compiler_options.defines = reduce_defines;
    reader.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
            },
        .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{},
    };
    if (reduce_w) {
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_reduce_universal_start_id.cpp"};
        reader.compile_time_arg_bindings = {
            {"scaler_bits", scaler_bits},
        };
        reader.runtime_arguments_schema.named_runtime_args = {"num_tiles", "start_id"};
    } else {
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp"};
        reader.compile_time_arg_bindings = {
            {"Ht", Ht},
            {"Wt", Wt},
            {"HtWt", HtWt},
            {"scaler_bits", scaler_bits},
            {"use_welford", 1u},
        };
        reader.runtime_arguments_schema.named_runtime_args = {"col_start_tile_id", "curr_col_in_batch", "num_cols"};
    }
    BindDFB(reader, INPUT_DFB, "input", m2::KernelSpec::DFBEndpointType::PRODUCER);
    BindDFB(reader, SCALER_DFB, "scaler", m2::KernelSpec::DFBEndpointType::PRODUCER);
    reader.tensor_bindings = {
        m2::KernelSpec::TensorBinding{.tensor_parameter_name = WELFORD_INPUT_TENSOR, .accessor_name = "input_tensor"},
    };

    // ---- Writer ----
    m2::KernelSpec writer;
    writer.unique_id = WELFORD_WRITER_KERNEL;
    writer.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
            },
        .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{},
    };
    if (reduce_hw) {
        writer.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_welford_hw.cpp"};
        writer.compile_time_arg_bindings = {
            {"Wt", Wt},
            {"W", W},
            {"tile_width", tile_width},
            {"H", H},
            {"correction", static_cast<uint32_t>(operation_attributes.correction)},
            {"reduce_batch_size", reduce_batch_size},
        };
        writer.runtime_arguments_schema.named_runtime_args = {"NC_per_core", "output_tile_start_id"};
        // HW writer doesn't propagate reduce_defines (matches Gen1 behavior).
        BindDFB(writer, PARTIAL_DFB, "partial", m2::KernelSpec::DFBEndpointType::CONSUMER);
        BindDFB(writer, COMBINED_DFB, "combined", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(writer, OUTPUT_DFB, "output", m2::KernelSpec::DFBEndpointType::CONSUMER);
    } else {
        writer.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_unary_interleaved.cpp"};
        writer.runtime_arguments_schema.named_runtime_args = {"num_pages", "start_id"};
        writer.compiler_options.defines = reduce_defines;
        BindDFB(writer, OUTPUT_DFB, "output", m2::KernelSpec::DFBEndpointType::CONSUMER);
    }
    writer.tensor_bindings = {
        m2::KernelSpec::TensorBinding{.tensor_parameter_name = WELFORD_OUTPUT_TENSOR, .accessor_name = "output_tensor"},
    };

    // ---- Compute ----
    std::string compute_kernel_path;
    if (reduce_w) {
        compute_kernel_path = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_w.cpp";
    } else if (reduce_h) {
        compute_kernel_path = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_h.cpp";
    } else {
        compute_kernel_path = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_hw.cpp";
    }

    m2::KernelSpec compute;
    compute.unique_id = WELFORD_COMPUTE_KERNEL;
    compute.source = m2::KernelSpec::SourceFilePath{compute_kernel_path};
    compute.compiler_options.defines = reduce_defines;
    compute.config_spec = m2::ComputeConfiguration{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };
    if (reduce_w) {
        compute.compile_time_arg_bindings = {
            {"Wt", Wt},
            {"W", W},
            {"tile_width", tile_width},
            {"do_scale", static_cast<uint32_t>(do_scale)},
            {"correction", static_cast<uint32_t>(operation_attributes.correction)},
            {"is_std", static_cast<uint32_t>(is_std)},
        };
        compute.runtime_arguments_schema.named_runtime_args = {"NCHt"};
    } else if (reduce_h) {
        compute.compile_time_arg_bindings = {
            {"Ht", Ht},
            {"H", H},
            {"tile_height", tile_height},
            {"do_scale", static_cast<uint32_t>(do_scale)},
            {"correction", static_cast<uint32_t>(operation_attributes.correction)},
            {"is_std", static_cast<uint32_t>(is_std)},
        };
        compute.runtime_arguments_schema.named_runtime_args = {"NCWt"};
    } else {
        compute.compile_time_arg_bindings = {
            {"Ht", Ht},
            {"H", H},
            {"tile_height", tile_height},
            {"Wt", Wt},
            {"do_scale", static_cast<uint32_t>(do_scale)},
            {"reduce_batch_size", reduce_batch_size},
            {"is_std", static_cast<uint32_t>(is_std)},
        };
        compute.runtime_arguments_schema.named_runtime_args = {"NC_per_core"};
    }
    BindDFB(compute, INPUT_DFB, "input", m2::KernelSpec::DFBEndpointType::CONSUMER);
    BindDFB(compute, SCALER_DFB, "scaler", m2::KernelSpec::DFBEndpointType::CONSUMER);
    BindDFB(compute, OUTPUT_DFB, "output", m2::KernelSpec::DFBEndpointType::PRODUCER);
    if (reduce_w) {
        BindDFB(compute, VAR_DFB, "var_w", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(compute, VAR_DFB, "var_r", m2::KernelSpec::DFBEndpointType::CONSUMER);
        BindDFB(compute, SCALED_DFB, "scaled_w", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(compute, SCALED_DFB, "scaled_r", m2::KernelSpec::DFBEndpointType::CONSUMER);
    }
    if (reduce_hw) {
        BindDFB(compute, PARTIAL_DFB, "partial", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(compute, COMBINED_DFB, "combined", m2::KernelSpec::DFBEndpointType::CONSUMER);
    }

    // ---- Single work unit ----
    m2::WorkUnitSpec work_unit;
    work_unit.unique_id = WELFORD_WORK_UNIT;
    work_unit.kernels = {WELFORD_READER_KERNEL, WELFORD_WRITER_KERNEL, WELFORD_COMPUTE_KERNEL};
    work_unit.target_nodes = wd.all_cores;

    // ---- Assemble + parameterize ----
    m2::ProgramSpec spec;
    spec.program_id = "ttnn::welford_reduce";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = WELFORD_INPUT_TENSOR, .spec = a.mesh_tensor().tensor_spec()},
        m2::TensorParameter{.unique_id = WELFORD_OUTPUT_TENSOR, .spec = output.mesh_tensor().tensor_spec()},
    };
    spec.work_units = {std::move(work_unit)};

    Program program = m2::MakeProgramFromSpec(*a.device(), spec);

    shared_variables_t shared{
        .cores = wd.cores,
        .core_group_1 = wd.core_group_1,
        .core_group_2 = wd.core_group_2,
        .num_work_units_per_core_group_1 = wd.num_work_units_per_core_group_1,
        .num_work_units_per_core_group_2 = wd.num_work_units_per_core_group_2,
        .Wt = Wt,
        .Ht = Ht,
        .HtWt = HtWt,
        .reduce_batch_size = reduce_batch_size,
        .reduce_dim = reduce_dim,
    };

    auto run_params = BuildRunParams(shared, a.mesh_tensor(), output.mesh_tensor());
    m2::SetProgramRunParameters(program, run_params);

    return cached_program_t{std::move(program), std::move(shared)};
}

void WelfordReduceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const WelfordReduceParams& /*operation_attributes*/,
    const tt::tt_metal::Tensor& tensor_args,
    tt::tt_metal::Tensor& tensor_return_value) {
    auto run_params =
        BuildRunParams(cached_program.shared_variables, tensor_args.mesh_tensor(), tensor_return_value.mesh_tensor());
    m2::SetProgramRunParameters(cached_program.program, run_params);
}

}  // namespace ttnn::prim
