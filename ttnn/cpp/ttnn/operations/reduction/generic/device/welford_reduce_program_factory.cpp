// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Welford reduction program factory, ported to the Metal 2.0 host API.
// Multi-variant: W / H / HW selected from operation_attributes.reduce_dim.

#include "welford_reduce_device_operation.hpp"
#include "reduce_metal2_factory_helpers.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include <bit>
#include <cmath>
#include <functional>
#include <map>

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

constexpr const char* WELFORD_READER_KERNEL = "welford_reader";
constexpr const char* WELFORD_WRITER_KERNEL = "welford_writer";
constexpr const char* WELFORD_COMPUTE_KERNEL_G1 = "welford_compute_g1";
constexpr const char* WELFORD_COMPUTE_KERNEL_G2 = "welford_compute_g2";
constexpr const char* WELFORD_WORK_UNIT_G1 = "welford_wu_g1";
constexpr const char* WELFORD_WORK_UNIT_G2 = "welford_wu_g2";

}  // namespace

ttnn::device_operation::ProgramArtifacts WelfordReduceDeviceOperation::WelfordReduceProgramFactory::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_arg,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Shape& padded_shape = tensor_arg.padded_shape();
    const Shape& logical_shape = tensor_arg.logical_shape();

    uint32_t W = logical_shape[-1];
    uint32_t H = logical_shape[-2];
    uint32_t W_padded = padded_shape[-1];
    uint32_t H_padded = padded_shape[-2];
    TT_FATAL(
        H_padded > 0 && W_padded > 0,
        "Padded H and W dimensions must be non-zero, got H_padded={}, W_padded={}",
        H_padded,
        W_padded);
    // Product of all dimensions except the last two (H, W).
    // Named NC by convention even though tensor may have arbitrary rank.
    uint32_t NC = tensor_arg.physical_volume() / (H_padded * W_padded);
    const uint32_t tile_height = tensor_arg.tensor_spec().tile().get_height();
    const uint32_t tile_width = tensor_arg.tensor_spec().tile().get_width();

    uint32_t Wt = W_padded / tile_width;
    uint32_t Ht = H_padded / tile_height;
    uint32_t HtWt = Ht * Wt;

    const bool reduce_w = (operation_attributes.reduce_dim == ReduceOpDim::W);
    const bool reduce_h = (operation_attributes.reduce_dim == ReduceOpDim::H);
    const bool reduce_hw = (operation_attributes.reduce_dim == ReduceOpDim::HW);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_arg.device()->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_arg.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    // Scalar datatype is hardcoded bfloat16 due to tile creation in reader.
    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = tensor_arg.device();

    const bool do_scale = (operation_attributes.scalar != 1.0f);
    const bool is_std = (operation_attributes.math_op == ReduceOpMath::STD);

    const uint32_t reduce_batch_size = operation_attributes.reduce_batch_size;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_work_units = reduce_w ? (NC * Ht) : (reduce_hw ? (NC / reduce_batch_size) : (NC * Wt));
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

    // ---- DFBs ----
    std::vector<m2::DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = input_cb_data_format,
        .tile_format_metadata = tensor_arg.tensor_spec().tile(),
    });
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = scalar_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scalar_cb_data_format,
        .tile_format_metadata = tensor_arg.tensor_spec().tile(),
    });
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = dst_cb_data_format,
        .tile_format_metadata = tensor_return_value.tensor_spec().tile(),
    });

    if (reduce_w) {
        // var scratch — Welford W needs it for the transpose between scaled and raw form.
        // It stores temporary data from the DST register, so data format is the same as DST.
        tt::DataFormat scratch_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        uint32_t scratch_single_tile_size = tt::tile_size(scratch_cb_data_format);
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = WELFORD_VAR_DFB,
            .entry_size = scratch_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = scratch_cb_data_format,
            .tile_format_metadata = tensor_arg.tensor_spec().tile(),
        });

        // cb_scaled — only used when do_scale; per the patterns catalog, we bind
        // unconditionally and let `if constexpr (do_scale)` in the kernel elide
        // the uses (or, since the legacy kernel already gates inside an
        // `if constexpr (do_scale)` block, the wrapper construction sits there
        // too — paying ~1 input-tile per core in L1 when do_scale=false is the
        // documented temporary cost).
        if (do_scale) {
            dataflow_buffers.push_back(m2::DataflowBufferSpec{
                .unique_id = WELFORD_SCALED_DFB,
                .entry_size = input_single_tile_size,
                .num_entries = 1,
                .data_format_metadata = input_cb_data_format,
                .tile_format_metadata = tensor_arg.tensor_spec().tile(),
            });
        }
    }

    if (reduce_hw) {
        // cb_partial: HW-reduce per-column mean+var tile pairs from compute,
        // consumed by writer. Uses Float32 to preserve precision from DST.
        tt::DataFormat partial_cb_data_format = tt::DataFormat::Float32;
        uint32_t partial_single_tile_size = tt::tile_size(partial_cb_data_format);
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = WELFORD_PARTIAL_DFB,
            // 4 entries for double-buffering (compute packs 2 tiles at a time).
            .entry_size = partial_single_tile_size,
            .num_entries = 4,
            .data_format_metadata = partial_cb_data_format,
            .tile_format_metadata = tensor_arg.tensor_spec().tile(),
        });
        // cb_combined: HW-reduce combined Float32 scalar from writer back to
        // compute, where compute applies sqrt (if std) and repacks to cb_out.
        tt::DataFormat combined_cb_data_format = tt::DataFormat::Float32;
        uint32_t combined_single_tile_size = tt::tile_size(combined_cb_data_format);
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = WELFORD_COMBINED_DFB,
            .entry_size = combined_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = combined_cb_data_format,
            .tile_format_metadata = tensor_arg.tensor_spec().tile(),
        });
    }

    std::map<std::string, std::string> reduce_defines_map =
        reduce_op_utils::get_defines(operation_attributes.math_op, operation_attributes.reduce_dim);
    reduce_defines_map["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    reduce_defines_map["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
    const auto reduce_defines = DefinesFromMap(reduce_defines_map);

    // ---- Reader ----
    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scalar);
    m2::KernelSpec reader;
    reader.unique_id = WELFORD_READER_KERNEL;
    reader.compiler_options.defines = reduce_defines;
    reader.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
            },
    };
    reader.dfb_bindings = {
        m2::KernelSpec::DFBBinding{
            .dfb_spec_name = IN_DFB,
            .local_accessor_name = "in_dfb",
            .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
        },
        m2::KernelSpec::DFBBinding{
            .dfb_spec_name = SCALER_DFB,
            .local_accessor_name = "scaler_dfb",
            .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
        },
    };
    reader.tensor_bindings = {
        m2::KernelSpec::TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"},
    };
    if (reduce_h || reduce_hw) {
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp"};
        reader.compile_time_arg_bindings = {
            {"Ht", Ht}, {"Wt", Wt}, {"HtWt", HtWt}, {"scaler_bits", scaler_bits}, {"use_welford", 1u}};
        reader.runtime_arguments_schema.named_runtime_args = {
            "src_addr", "col_start_tile_id", "curr_col_in_batch", "num_cols"};
    } else {
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_reduce_universal_start_id.cpp"};
        reader.compile_time_arg_bindings = {{"scaler_bits", scaler_bits}};
        reader.runtime_arguments_schema.named_runtime_args = {"num_tiles", "start_id"};
    }

    // ---- Writer ----
    m2::KernelSpec writer;
    writer.unique_id = WELFORD_WRITER_KERNEL;
    writer.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
            },
    };
    writer.tensor_bindings = {
        m2::KernelSpec::TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"},
    };
    if (reduce_hw) {
        if (operation_attributes.correction) {
            TT_FATAL(
                H * W * reduce_batch_size >= 2,
                "Bessel's correction requires at least 2 elements across all reduction dimensions, got {}",
                H * W * reduce_batch_size);
        }
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
        writer.runtime_arguments_schema.named_runtime_args = {"dst_addr", "NC_per_core", "output_tile_start_id"};
        writer.dfb_bindings = {
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = WELFORD_PARTIAL_DFB,
                .local_accessor_name = "partial_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            },
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = WELFORD_COMBINED_DFB,
                .local_accessor_name = "combined_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            },
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .local_accessor_name = "out_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            },
        };
        // Note: HW writer does not carry reduce_defines (matches original behavior).
    } else {
        writer.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp"};
        writer.runtime_arguments_schema.named_runtime_args = {"num_pages", "start_id"};
        writer.compiler_options.defines = reduce_defines;
        writer.dfb_bindings = {
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .local_accessor_name = "out_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            },
        };
    }

    // ---- Compute ----
    std::string compute_kernel_path;
    m2::KernelSpec::CompileTimeArgBindings compute_ctas;
    if (reduce_hw) {
        if (operation_attributes.correction) {
            // already checked above
        }
        compute_kernel_path = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_hw.cpp";
        compute_ctas = {
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
        compute_kernel_path =
            reduce_w ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_w.cpp"
                     : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_h.cpp";
        compute_ctas = {
            {reduce_w ? "Wt" : "Ht", reduce_w ? Wt : Ht},
            {reduce_w ? "W" : "H", reduce_w ? W : H},
            {reduce_w ? "tile_width" : "tile_height", reduce_w ? tile_width : tile_height},
            {"do_scale", static_cast<uint32_t>(do_scale)},
            {"correction", static_cast<uint32_t>(operation_attributes.correction)},
            {"is_std", static_cast<uint32_t>(is_std)},
        };
    }

    auto make_compute = [&](const char* unique_id) {
        m2::KernelSpec compute;
        compute.unique_id = unique_id;
        compute.source = m2::KernelSpec::SourceFilePath{compute_kernel_path};
        compute.compile_time_arg_bindings = compute_ctas;
        compute.compiler_options.defines = reduce_defines;
        compute.config_spec = m2::ComputeConfiguration{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        compute.runtime_arguments_schema.named_runtime_args = {"work_units_per_core"};
        compute.dfb_bindings = {
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = IN_DFB,
                .local_accessor_name = "in_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            },
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = SCALER_DFB,
                .local_accessor_name = "scaler_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            },
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .local_accessor_name = "out_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            },
        };
        if (reduce_w) {
            // var scratch — compute writes (PRODUCER) and reads (CONSUMER) via transpose.
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = WELFORD_VAR_DFB,
                .local_accessor_name = "var_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            });
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = WELFORD_VAR_DFB,
                .local_accessor_name = "var_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            });
            if (do_scale) {
                // scaled_dfb is also a self-loop on the compute kernel (Welford W
                // packs into it from the FPU mul step, then re-reads via
                // transpose_wh_tile).
                compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                    .dfb_spec_name = WELFORD_SCALED_DFB,
                    .local_accessor_name = "scaled_dfb",
                    .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
                });
                compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                    .dfb_spec_name = WELFORD_SCALED_DFB,
                    .local_accessor_name = "scaled_dfb",
                    .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
                });
            }
        }
        if (reduce_hw) {
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = WELFORD_PARTIAL_DFB,
                .local_accessor_name = "partial_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            });
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = WELFORD_COMBINED_DFB,
                .local_accessor_name = "combined_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            });
        }
        return compute;
    };

    std::vector<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute(WELFORD_COMPUTE_KERNEL_G1));
    const bool g2_present = !core_group_2.ranges().empty();
    if (g2_present) {
        kernels.push_back(make_compute(WELFORD_COMPUTE_KERNEL_G2));
    }

    std::vector<m2::WorkUnitSpec> work_units;
    work_units.push_back(m2::WorkUnitSpec{
        .unique_id = WELFORD_WORK_UNIT_G1,
        .kernels = {WELFORD_READER_KERNEL, WELFORD_WRITER_KERNEL, WELFORD_COMPUTE_KERNEL_G1},
        .target_nodes = core_group_1,
    });
    if (g2_present) {
        work_units.push_back(m2::WorkUnitSpec{
            .unique_id = WELFORD_WORK_UNIT_G2,
            .kernels = {WELFORD_READER_KERNEL, WELFORD_WRITER_KERNEL, WELFORD_COMPUTE_KERNEL_G2},
            .target_nodes = core_group_2,
        });
    }

    // ---- Assemble spec ----
    m2::ProgramSpec spec;
    spec.program_id = "ttnn::welford_reduce";
    spec.kernels = std::move(kernels);
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = INPUT_TENSOR, .spec = tensor_arg.tensor_spec()},
        m2::TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = tensor_return_value.tensor_spec()},
    };
    spec.work_units = std::move(work_units);

    // ---- Run params: per-core RTAs ----
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

    m2::ProgramRunParams::KernelRunParams reader_params{.kernel_spec_name = WELFORD_READER_KERNEL};
    m2::ProgramRunParams::KernelRunParams writer_params{.kernel_spec_name = WELFORD_WRITER_KERNEL};
    m2::ProgramRunParams::KernelRunParams compute_g1_params{.kernel_spec_name = WELFORD_COMPUTE_KERNEL_G1};
    m2::ProgramRunParams::KernelRunParams compute_g2_params{.kernel_spec_name = WELFORD_COMPUTE_KERNEL_G2};

    if (reduce_w) {
        uint32_t input_tiles_offset = 0;
        uint32_t output_tiles_offset = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_work_units_per_core = 0;
            bool in_g1 = core_group_1.contains(core);
            if (in_g1) {
                num_work_units_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_work_units_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            uint32_t num_input_tiles_per_core = num_work_units_per_core * Wt;
            uint32_t num_output_tiles_per_core = num_work_units_per_core;
            reader_params.named_runtime_args.push_back({
                .node = core,
                .args = {{"num_tiles", num_input_tiles_per_core}, {"start_id", input_tiles_offset}},
            });
            writer_params.named_runtime_args.push_back({
                .node = core,
                .args = {{"num_pages", num_output_tiles_per_core}, {"start_id", output_tiles_offset}},
            });
            auto& compute_params = in_g1 ? compute_g1_params : compute_g2_params;
            compute_params.named_runtime_args.push_back({
                .node = core,
                .args = {{"work_units_per_core", num_work_units_per_core}},
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
            bool in_g1 = core_group_1.contains(core);
            if (in_g1) {
                num_outputs_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_outputs_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            uint32_t nc_slices_per_core = num_outputs_per_core * reduce_batch_size;
            uint32_t num_cols_for_reader = Wt * nc_slices_per_core;
            uint32_t col_start_tile_id = nc_slice_offset * HtWt;
            reader_params.named_runtime_args.push_back({
                .node = core,
                .args =
                    {{"src_addr", 0u},
                     {"col_start_tile_id", col_start_tile_id},
                     {"curr_col_in_batch", 0u},
                     {"num_cols", num_cols_for_reader}},
            });
            writer_params.named_runtime_args.push_back({
                .node = core,
                .args =
                    {{"dst_addr", 0u}, {"NC_per_core", nc_slices_per_core}, {"output_tile_start_id", output_offset}},
            });
            auto& compute_params = in_g1 ? compute_g1_params : compute_g2_params;
            compute_params.named_runtime_args.push_back({
                .node = core,
                .args = {{"work_units_per_core", nc_slices_per_core}},
            });
            nc_slice_offset += nc_slices_per_core;
            output_offset += num_outputs_per_core;
        }
    } else {
        // H-reduce
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
        uint32_t num_cols_read = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_cols_per_core = 0;
            bool in_g1 = core_group_1.contains(core);
            if (in_g1) {
                num_cols_per_core = num_work_units_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_cols_per_core = num_work_units_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            reader_params.named_runtime_args.push_back({
                .node = core,
                .args =
                    {{"src_addr", 0u},
                     {"col_start_tile_id", (num_cols_read / Wt * HtWt) + (num_cols_read % Wt)},
                     {"curr_col_in_batch", num_cols_read % Wt},
                     {"num_cols", num_cols_per_core}},
            });
            writer_params.named_runtime_args.push_back({
                .node = core,
                .args = {{"num_pages", num_cols_per_core}, {"start_id", num_cols_read}},
            });
            auto& compute_params = in_g1 ? compute_g1_params : compute_g2_params;
            compute_params.named_runtime_args.push_back({
                .node = core,
                .args = {{"work_units_per_core", num_cols_per_core}},
            });
            num_cols_read += num_cols_per_core;
        }
    }

    m2::ProgramRunParams run_params;
    run_params.kernel_run_params.push_back(std::move(reader_params));
    run_params.kernel_run_params.push_back(std::move(writer_params));
    run_params.kernel_run_params.push_back(std::move(compute_g1_params));
    if (g2_present) {
        run_params.kernel_run_params.push_back(std::move(compute_g2_params));
    }
    run_params.tensor_args = {
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = INPUT_TENSOR, .tensor = std::cref(tensor_arg.mesh_tensor())},
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = OUTPUT_TENSOR, .tensor = std::cref(tensor_return_value.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
}

}  // namespace ttnn::prim
