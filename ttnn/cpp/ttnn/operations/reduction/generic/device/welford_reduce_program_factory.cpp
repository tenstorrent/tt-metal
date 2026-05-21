// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford reduce program factory, Metal 2.0 host-API port.
// Multi-variant factory: branches on `reduce_dim` (W / H / HW).

#include <bit>
#include <cmath>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "welford_reduce_device_operation.hpp"

#include <map>
#include <string>
#include <vector>

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

// Unity-build hygiene: `WF_` prefix.
constexpr const char* WF_READER = "reader";
constexpr const char* WF_WRITER = "writer";
constexpr const char* WF_COMPUTE_G1 = "compute_g1";
constexpr const char* WF_COMPUTE_G2 = "compute_g2";

constexpr const char* WF_WU_G1 = "wu_g1";
constexpr const char* WF_WU_G2 = "wu_g2";

constexpr const char* WF_INPUT_DFB = "input";
constexpr const char* WF_SCALER_DFB = "scaler";
constexpr const char* WF_OUTPUT_DFB = "output";
constexpr const char* WF_SCRATCH_DFB = "scratch";    // W-reduce: c_19
constexpr const char* WF_SCALED_DFB = "scaled";      // W-reduce, do_scale: c_20
constexpr const char* WF_PARTIAL_DFB = "partial";    // HW-reduce: c_21
constexpr const char* WF_COMBINED_DFB = "combined";  // HW-reduce: c_22

constexpr const char* WF_INPUT_TENSOR = "input";
constexpr const char* WF_OUTPUT_TENSOR = "output";

m2::KernelSpec::CompilerOptions::Defines wf_defines_from_map(const std::map<std::string, std::string>& src) {
    m2::KernelSpec::CompilerOptions::Defines out;
    out.reserve(src.size());
    for (const auto& [k, v] : src) {
        out.emplace_back(k, v);
    }
    return out;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts WelfordReduceDeviceOperation::WelfordReduceProgramFactory::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_arg,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Shape& padded_shape = tensor_arg.padded_shape();
    const Shape& logical_shape = tensor_arg.logical_shape();

    const uint32_t W = logical_shape[-1];
    const uint32_t H = logical_shape[-2];
    const uint32_t W_padded = padded_shape[-1];
    const uint32_t H_padded = padded_shape[-2];
    TT_FATAL(
        H_padded > 0 && W_padded > 0,
        "Padded H and W dimensions must be non-zero, got H_padded={}, W_padded={}",
        H_padded,
        W_padded);
    const uint32_t NC = tensor_arg.physical_volume() / (H_padded * W_padded);
    const uint32_t tile_height = tensor_arg.tensor_spec().tile().get_height();
    const uint32_t tile_width = tensor_arg.tensor_spec().tile().get_width();

    const uint32_t Wt = W_padded / tile_width;
    const uint32_t Ht = H_padded / tile_height;
    const uint32_t HtWt = Ht * Wt;

    const bool reduce_w = (operation_attributes.reduce_dim == ReduceOpDim::W);
    const bool reduce_h = (operation_attributes.reduce_dim == ReduceOpDim::H);
    const bool reduce_hw = (operation_attributes.reduce_dim == ReduceOpDim::HW);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_arg.device()->arch(), operation_attributes.compute_kernel_config);

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_arg.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    // Scalar datatype is hardcoded bfloat16 due to tile creation in reader.
    const tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    const uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);
    const tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = tensor_arg.device();

    const uint32_t reduce_batch_size = operation_attributes.reduce_batch_size;
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_work_units = reduce_w ? (NC * Ht) : (reduce_hw ? (NC / reduce_batch_size) : (NC * Wt));

    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_work_units_per_core_group_1, num_work_units_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_work_units_per_core_group_1,
            num_work_units_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_work_units);
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_work_units_per_core_group_1,
            num_work_units_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_work_units);
    }

    const bool do_scale = (operation_attributes.scalar != 1.0f);
    const bool is_std = (operation_attributes.math_op == ReduceOpMath::STD);
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scalar);

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, operation_attributes.reduce_dim);
    reduce_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    reduce_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
    // DO_SCALE / IS_STD compile-time gates: used by the Welford compute kernels to
    // statically eliminate references to optionally-bound DFBs (dfb::scaled) so
    // they don't need to exist in kernel_args_generated.h when not bound.
    if (do_scale) {
        reduce_defines["DO_SCALE"] = "1";
    }

    // ----- DataflowBufferSpecs -----

    std::vector<m2::DataflowBufferSpec> dataflow_buffers;

    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = WF_INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = input_cb_data_format,
        .tile_format_metadata = tensor_arg.tensor_spec().tile(),
    });
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = WF_SCALER_DFB,
        .entry_size = scalar_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scalar_cb_data_format,
    });
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = WF_OUTPUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = dst_cb_data_format,
        .tile_format_metadata = tensor_return_value.tensor_spec().tile(),
    });

    if (reduce_w) {
        // c_19 scratch (W-reduce only). Self-loop on compute -> intra-tensix DFB
        // (implicit sync not supported for intra-tensix).
        const tt::DataFormat scratch_cb_data_format =
            fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        const uint32_t scratch_single_tile_size = tt::tile_size(scratch_cb_data_format);
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = WF_SCRATCH_DFB,
            .entry_size = scratch_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = scratch_cb_data_format,
            .disable_implicit_sync = true,
        });
    }

    if (do_scale && reduce_w) {
        // c_20 scaled (W-reduce + do_scale only). Self-loop on compute.
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = WF_SCALED_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = input_cb_data_format,
            .disable_implicit_sync = true,
        });
    }

    if (reduce_hw) {
        // c_21 partial (HW-reduce only): Float32.
        const tt::DataFormat partial_cb_data_format = tt::DataFormat::Float32;
        const uint32_t partial_single_tile_size = tt::tile_size(partial_cb_data_format);
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = WF_PARTIAL_DFB,
            .entry_size = partial_single_tile_size,
            .num_entries = 4,  // double-buffered (compute packs 2 tiles at a time)
            .data_format_metadata = partial_cb_data_format,
        });

        // c_22 combined (HW-reduce only): Float32.
        const tt::DataFormat combined_cb_data_format = tt::DataFormat::Float32;
        const uint32_t combined_single_tile_size = tt::tile_size(combined_cb_data_format);
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = WF_COMBINED_DFB,
            .entry_size = combined_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = combined_cb_data_format,
        });
    }

    // ----- KernelSpecs -----

    m2::KernelSpec reader{
        .unique_id = WF_READER,
        .compiler_options =
            {
                .defines = wf_defines_from_map(reduce_defines),
            },
        .config_spec =
            m2::DataMovementConfiguration{
                .gen1_data_movement_config =
                    m2::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                    },
            },
    };

    if (reduce_h || reduce_hw) {
        // Column-partitioned reader (Welford H / HW).
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp"};
        reader.dfb_bindings = {
            {.dfb_spec_name = WF_INPUT_DFB,
             .local_accessor_name = "input",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
            {.dfb_spec_name = WF_SCALER_DFB,
             .local_accessor_name = "scaler",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
        };
        reader.tensor_bindings = {
            {.tensor_parameter_name = WF_INPUT_TENSOR, .accessor_name = "input"},
        };
        reader.compile_time_arg_bindings = {
            {"Ht", Ht}, {"Wt", Wt}, {"HtWt", HtWt}, {"scaler_bits", scaler_bits}, {"use_welford", 1u}};
        reader.runtime_arguments_schema = {
            .named_runtime_args = {"col_start_tile_id", "curr_col_in_batch", "num_cols"}};
    } else {
        // Sequential reader (Welford W).
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_reduce_universal_start_id.cpp"};
        reader.dfb_bindings = {
            {.dfb_spec_name = WF_INPUT_DFB,
             .local_accessor_name = "input",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
            {.dfb_spec_name = WF_SCALER_DFB,
             .local_accessor_name = "scaler",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
        };
        reader.tensor_bindings = {
            {.tensor_parameter_name = WF_INPUT_TENSOR, .accessor_name = "input"},
        };
        reader.compile_time_arg_bindings = {{"scaler_bits", scaler_bits}};
        reader.runtime_arguments_schema = {.named_runtime_args = {"num_tiles", "start_id"}};
    }

    m2::KernelSpec writer{
        .unique_id = WF_WRITER,
        .config_spec =
            m2::DataMovementConfiguration{
                .gen1_data_movement_config =
                    m2::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };

    if (reduce_hw) {
        if (operation_attributes.correction) {
            TT_FATAL(
                H * W * reduce_batch_size >= 2,
                "Bessel's correction requires at least 2 elements across all reduction dimensions, got {}",
                H * W * reduce_batch_size);
        }
        // Op-local HW writer.
        writer.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_welford_hw.cpp"};
        writer.dfb_bindings = {
            {.dfb_spec_name = WF_OUTPUT_DFB,
             .local_accessor_name = "output",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
            {.dfb_spec_name = WF_PARTIAL_DFB,
             .local_accessor_name = "partial",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
            {.dfb_spec_name = WF_COMBINED_DFB,
             .local_accessor_name = "combined",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
        };
        writer.tensor_bindings = {
            {.tensor_parameter_name = WF_OUTPUT_TENSOR, .accessor_name = "output"},
        };
        writer.compile_time_arg_bindings = {
            {"Wt", Wt},
            {"W", W},
            {"tile_width", tile_width},
            {"H", H},
            {"correction", static_cast<uint32_t>(operation_attributes.correction)},
            {"reduce_batch_size", reduce_batch_size},
        };
        writer.runtime_arguments_schema = {.named_runtime_args = {"NC_per_core", "output_tile_start_id"}};
        // Note: HW writer does not pass reduce_defines (matches original).
    } else {
        // W-reduce / H-reduce: generic Metal 2.0 tile writer (cross-op fork).
        writer.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp"};
        writer.compiler_options.defines = wf_defines_from_map(reduce_defines);
        writer.dfb_bindings = {
            {.dfb_spec_name = WF_OUTPUT_DFB,
             .local_accessor_name = "output",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
        };
        writer.tensor_bindings = {
            {.tensor_parameter_name = WF_OUTPUT_TENSOR, .accessor_name = "output"},
        };
        writer.runtime_arguments_schema = {.named_runtime_args = {"num_pages", "start_id"}};
    }

    // Compute kernel(s).
    std::string compute_kernel_source;
    std::vector<std::pair<std::string, uint32_t>> base_compute_ctas;

    if (reduce_hw) {
        compute_kernel_source =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_hw.cpp";
        base_compute_ctas = {
            {"Ht", Ht},
            {"H", H},
            {"tile_height", tile_height},
            {"Wt", Wt},
            {"do_scale", static_cast<uint32_t>(do_scale)},
            {"reduce_batch_size", reduce_batch_size},
            {"is_std", static_cast<uint32_t>(is_std)},
        };
    } else {
        if (operation_attributes.correction) {
            uint32_t reduce_size = reduce_w ? W : H;
            TT_FATAL(
                reduce_size >= 2,
                "Bessel's correction requires at least 2 elements along the reduction dimension, got {}",
                reduce_size);
        }
        compute_kernel_source =
            reduce_w ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_w.cpp"
                     : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_h.cpp";
        base_compute_ctas = {
            {reduce_w ? std::string("Wt") : std::string("Ht"), reduce_w ? Wt : Ht},
            {reduce_w ? std::string("W") : std::string("H"), reduce_w ? W : H},
            {reduce_w ? std::string("tile_width") : std::string("tile_height"), reduce_w ? tile_width : tile_height},
            {"do_scale", static_cast<uint32_t>(do_scale)},
            {"correction", static_cast<uint32_t>(operation_attributes.correction)},
            {"is_std", static_cast<uint32_t>(is_std)},
        };
    }

    auto make_compute = [&](const char* unique_id) {
        m2::KernelSpec spec{
            .unique_id = unique_id,
            .source = m2::KernelSpec::SourceFilePath{compute_kernel_source},
            .compiler_options =
                {
                    .defines = wf_defines_from_map(reduce_defines),
                },
            .dfb_bindings =
                {
                    {.dfb_spec_name = WF_INPUT_DFB,
                     .local_accessor_name = "input",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
                    {.dfb_spec_name = WF_OUTPUT_DFB,
                     .local_accessor_name = "output",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
                    // Always bind scaler as CONSUMER on compute: the reader unconditionally
                    // produces a scaler tile via prepare_reduce_scaler, so the DFB needs a
                    // matching consumer. The compute kernel waits on it unconditionally and
                    // uses the value only when do_scale.
                    {.dfb_spec_name = WF_SCALER_DFB,
                     .local_accessor_name = "scaler",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
                },
            .compile_time_arg_bindings = base_compute_ctas,
            .runtime_arguments_schema =
                {.named_runtime_args =
                     {reduce_w ? std::string("NCHt") : (reduce_hw ? std::string("NC_per_core") : std::string("NCWt"))}},
        };
        if (reduce_w) {
            // Scratch DFB self-loop (welford_reduce_w writes then transposes back).
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = WF_SCRATCH_DFB,
                 .local_accessor_name = "scratch",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER});
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = WF_SCRATCH_DFB,
                 .local_accessor_name = "scratch",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER});
            if (do_scale) {
                // Scaled DFB self-loop (conditional).
                spec.dfb_bindings.push_back(
                    {.dfb_spec_name = WF_SCALED_DFB,
                     .local_accessor_name = "scaled",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER});
                spec.dfb_bindings.push_back(
                    {.dfb_spec_name = WF_SCALED_DFB,
                     .local_accessor_name = "scaled",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER});
            }
        }
        if (reduce_hw) {
            // 2-way data flow: c_21 (compute -> writer), c_22 (writer -> compute).
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = WF_PARTIAL_DFB,
                 .local_accessor_name = "partial",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER});
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = WF_COMBINED_DFB,
                 .local_accessor_name = "combined",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER});
        }
        // Build ComputeConfiguration with required unpack_to_dest_mode entries for FP32 DFBs
        // consumed by the compute kernel (when fp32_dest_acc_en is true).
        m2::ComputeConfiguration cc{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        if (fp32_dest_acc_en) {
            if (input_cb_data_format == tt::DataFormat::Float32) {
                cc.unpack_to_dest_mode.push_back({WF_INPUT_DFB, tt::tt_metal::UnpackToDestMode::Default});
            }
            // scaler_cb_data_format is Float16_b (hardcoded) — never Float32, no entry needed.
            if (reduce_w) {
                // scratch DFB format is Float32 when fp32_dest_acc_en is true.
                cc.unpack_to_dest_mode.push_back({WF_SCRATCH_DFB, tt::tt_metal::UnpackToDestMode::Default});
                if (do_scale && input_cb_data_format == tt::DataFormat::Float32) {
                    cc.unpack_to_dest_mode.push_back({WF_SCALED_DFB, tt::tt_metal::UnpackToDestMode::Default});
                }
            }
            if (reduce_hw) {
                // combined DFB is Float32 (always); partial is Float32 but PRODUCER-only on
                // compute (no unpack needed for PRODUCER).
                cc.unpack_to_dest_mode.push_back({WF_COMBINED_DFB, tt::tt_metal::UnpackToDestMode::Default});
            }
        }
        spec.config_spec = cc;
        return spec;
    };

    m2::KernelSpec compute_g1 = make_compute(WF_COMPUTE_G1);
    std::optional<m2::KernelSpec> compute_g2;
    const bool g2_present = !core_group_2.ranges().empty();
    if (g2_present) {
        compute_g2 = make_compute(WF_COMPUTE_G2);
    }

    // ----- WorkUnitSpecs -----
    std::vector<m2::WorkUnitSpec> work_units;
    work_units.push_back(m2::WorkUnitSpec{
        .unique_id = WF_WU_G1,
        .kernels = {WF_READER, WF_WRITER, WF_COMPUTE_G1},
        .target_nodes = core_group_1,
    });
    if (g2_present) {
        work_units.push_back(m2::WorkUnitSpec{
            .unique_id = WF_WU_G2,
            .kernels = {WF_READER, WF_WRITER, WF_COMPUTE_G2},
            .target_nodes = core_group_2,
        });
    }

    // ----- ProgramSpec assembly -----

    std::vector<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(std::move(compute_g1));
    if (compute_g2.has_value()) {
        kernels.push_back(std::move(*compute_g2));
    }

    m2::ProgramSpec spec{
        .program_id = "welford_reduce",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                {.unique_id = WF_INPUT_TENSOR, .spec = tensor_arg.tensor_spec()},
                {.unique_id = WF_OUTPUT_TENSOR, .spec = tensor_return_value.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    // ----- ProgramRunParams -----
    m2::ProgramRunParams run_params;
    m2::ProgramRunParams::KernelRunParams reader_rp{.kernel_spec_name = WF_READER};
    m2::ProgramRunParams::KernelRunParams writer_rp{.kernel_spec_name = WF_WRITER};
    m2::ProgramRunParams::KernelRunParams compute_g1_rp{.kernel_spec_name = WF_COMPUTE_G1};
    std::optional<m2::ProgramRunParams::KernelRunParams> compute_g2_rp;
    if (g2_present) {
        compute_g2_rp = m2::ProgramRunParams::KernelRunParams{.kernel_spec_name = WF_COMPUTE_G2};
    }

    std::vector<CoreCoord> cores;
    if (operation_attributes.sub_core_grids.has_value()) {
        for (const auto& range : all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    cores.emplace_back(x, y);
                }
            }
        }
    } else {
        cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }

    if (reduce_w) {
        uint32_t input_tiles_offset = 0;
        uint32_t output_tiles_offset = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_work_units_per_core = 0;
            const bool in_g1 = core_group_1.contains(core);
            if (in_g1) {
                num_work_units_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_work_units_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            const uint32_t num_input_tiles_per_core = num_work_units_per_core * Wt;
            const uint32_t num_output_tiles_per_core = num_work_units_per_core;

            reader_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"num_tiles", num_input_tiles_per_core}, {"start_id", input_tiles_offset}},
            });
            auto& target_compute_rp = in_g1 ? compute_g1_rp : *compute_g2_rp;
            target_compute_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"NCHt", num_work_units_per_core}},
            });
            writer_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"num_pages", num_output_tiles_per_core}, {"start_id", output_tiles_offset}},
            });
            input_tiles_offset += num_input_tiles_per_core;
            output_tiles_offset += num_output_tiles_per_core;
        }
    } else if (reduce_hw) {
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
        TT_FATAL(
            NC % reduce_batch_size == 0, "NC ({}) must be divisible by reduce_batch_size ({})", NC, reduce_batch_size);
        uint32_t nc_slice_offset = 0;
        uint32_t output_offset = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_outputs_per_core = 0;
            const bool in_g1 = core_group_1.contains(core);
            if (in_g1) {
                num_outputs_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_outputs_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            const uint32_t nc_slices_per_core = num_outputs_per_core * reduce_batch_size;
            const uint32_t num_cols_this_core = Wt * nc_slices_per_core;
            const uint32_t col_start_tile_id = nc_slice_offset * HtWt;

            reader_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args =
                    {{"col_start_tile_id", col_start_tile_id},
                     {"curr_col_in_batch", 0u},
                     {"num_cols", num_cols_this_core}},
            });
            auto& target_compute_rp = in_g1 ? compute_g1_rp : *compute_g2_rp;
            target_compute_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"NC_per_core", nc_slices_per_core}},
            });
            writer_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"NC_per_core", nc_slices_per_core}, {"output_tile_start_id", output_offset}},
            });
            nc_slice_offset += nc_slices_per_core;
            output_offset += num_outputs_per_core;
        }
    } else {
        // reduce_h
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
        uint32_t num_cols_read = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_cols_per_core = 0;
            const bool in_g1 = core_group_1.contains(core);
            if (in_g1) {
                num_cols_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_cols_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            reader_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args =
                    {{"col_start_tile_id", (num_cols_read / Wt * HtWt) + (num_cols_read % Wt)},
                     {"curr_col_in_batch", num_cols_read % Wt},
                     {"num_cols", num_cols_per_core}},
            });
            auto& target_compute_rp = in_g1 ? compute_g1_rp : *compute_g2_rp;
            target_compute_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"NCWt", num_cols_per_core}},
            });
            writer_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"num_pages", num_cols_per_core}, {"start_id", num_cols_read}},
            });
            num_cols_read += num_cols_per_core;
        }
    }

    run_params.kernel_run_params = {std::move(reader_rp), std::move(writer_rp), std::move(compute_g1_rp)};
    if (compute_g2_rp.has_value()) {
        run_params.kernel_run_params.push_back(std::move(*compute_g2_rp));
    }

    run_params.tensor_args = {
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = WF_INPUT_TENSOR, .tensor = std::cref(tensor_arg.mesh_tensor())},
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = WF_OUTPUT_TENSOR, .tensor = std::cref(tensor_return_value.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
