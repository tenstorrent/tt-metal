// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core W reduction program factory, ported to the Metal 2.0 host API.
// Reduces along W; work split is by rows (NC * Ht work units) across worker cores.

#include "reduce_op_device_operation.hpp"
#include "reduce_metal2_factory_helpers.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

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

#include <bit>
#include <cmath>
#include <functional>
#include <map>

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

// W-factory KernelSpec / WorkUnitSpec ids — prefixed to disambiguate under
// Unity-build anonymous-namespace merging.
constexpr const char* W_READER_KERNEL = "reduce_w_reader";
constexpr const char* W_WRITER_KERNEL = "reduce_w_writer";
constexpr const char* W_COMPUTE_KERNEL_G1 = "reduce_w_compute_g1";
constexpr const char* W_COMPUTE_KERNEL_G2 = "reduce_w_compute_g2";
constexpr const char* W_WORK_UNIT_G1 = "reduce_w_wu_g1";
constexpr const char* W_WORK_UNIT_G2 = "reduce_w_wu_g2";

}  // namespace

ttnn::device_operation::ProgramArtifacts ReduceDeviceOperation::ReduceMultiCoreWProgramFactory::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args;
    auto& output = tensor_return_value;

    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    uint32_t Wt = W / tile_width;
    uint32_t Ht = H / tile_height;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);

    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_rows = NC * Ht;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_rows_per_core_group_1, num_rows_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_rows);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);
    }
    TT_FATAL(num_cores > 0, "Reduce W requires at least one worker core");

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // ---- DFBs ----
    std::vector<m2::DataflowBufferSpec> dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = IN_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = 2,
            .data_format_metadata = src0_cb_data_format,
            .tile_format_metadata = a.tensor_spec().tile(),
        },
        m2::DataflowBufferSpec{
            .unique_id = SCALER_DFB,
            .entry_size = scaler_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = scaler_cb_data_format,
            .tile_format_metadata = a.tensor_spec().tile(),
        },
        m2::DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = 2,
            .data_format_metadata = dst_cb_data_format,
            .tile_format_metadata = output.tensor_spec().tile(),
        },
    };
    if (operation_attributes.negate) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = dst_cb_data_format,
            .tile_format_metadata = output.tensor_spec().tile(),
        });
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = INEG_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = dst_cb_data_format,
            .tile_format_metadata = output.tensor_spec().tile(),
        });
    }

    std::map<std::string, std::string> reduce_defines_map =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    if (use_post_mul) {
        reduce_defines_map["REDUCE_POST_MUL"] = "1";
    }
    const auto reduce_defines = DefinesFromMap(reduce_defines_map);

    // ---- Reader ----
    m2::KernelSpec reader;
    reader.unique_id = W_READER_KERNEL;
    reader.source = m2::KernelSpec::SourceFilePath{
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "reader_unary_reduce_universal_start_id.cpp"};
    reader.compile_time_arg_bindings = {{"scaler_bits", scaler_bits}};
    reader.runtime_arguments_schema.named_runtime_args = {"num_tiles", "start_id"};
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

    // ---- Writer (forked _metal2 copy of writer_unary_interleaved_start_id) ----
    m2::KernelSpec writer;
    writer.unique_id = W_WRITER_KERNEL;
    writer.source = m2::KernelSpec::SourceFilePath{
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_metal2.cpp"};
    writer.runtime_arguments_schema.named_runtime_args = {"num_pages", "start_id"};
    writer.compiler_options.defines = reduce_defines;
    writer.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
            },
    };
    writer.dfb_bindings = {
        m2::KernelSpec::DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "out_dfb",
            .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
        },
    };
    writer.tensor_bindings = {
        m2::KernelSpec::TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"},
    };

    // ---- Compute (per-core-group multiplicity) ----
    const std::string compute_kernel_path =
        operation_attributes.negate
            ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp"
            : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce.cpp";

    auto make_compute = [&](const char* unique_id, uint32_t Ht_for_group) {
        m2::KernelSpec compute;
        compute.unique_id = unique_id;
        compute.source = m2::KernelSpec::SourceFilePath{compute_kernel_path};
        compute.compile_time_arg_bindings = {
            {"Ht", Ht_for_group},
            {"Wt", Wt},
            {"NC", 1u},
            {"post_mul_scaler_bits", post_mul_scaler_bits},
        };
        compute.compiler_options.defines = reduce_defines;
        compute.config_spec = m2::ComputeConfiguration{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
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
        if (operation_attributes.negate) {
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = ACC_DFB,
                .local_accessor_name = "acc_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            });
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = ACC_DFB,
                .local_accessor_name = "acc_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            });
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = INEG_DFB,
                .local_accessor_name = "ineg_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            });
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = INEG_DFB,
                .local_accessor_name = "ineg_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            });
        }
        return compute;
    };

    std::vector<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute(W_COMPUTE_KERNEL_G1, num_rows_per_core_group_1));
    const bool g2_present = !core_group_2.ranges().empty();
    if (g2_present) {
        kernels.push_back(make_compute(W_COMPUTE_KERNEL_G2, num_rows_per_core_group_2));
    }

    // ---- WorkUnits ----
    std::vector<m2::WorkUnitSpec> work_units;
    work_units.push_back(m2::WorkUnitSpec{
        .unique_id = W_WORK_UNIT_G1,
        .kernels = {W_READER_KERNEL, W_WRITER_KERNEL, W_COMPUTE_KERNEL_G1},
        .target_nodes = core_group_1,
    });
    if (g2_present) {
        work_units.push_back(m2::WorkUnitSpec{
            .unique_id = W_WORK_UNIT_G2,
            .kernels = {W_READER_KERNEL, W_WRITER_KERNEL, W_COMPUTE_KERNEL_G2},
            .target_nodes = core_group_2,
        });
    }

    // ---- Assemble spec ----
    m2::ProgramSpec spec;
    spec.program_id = "ttnn::reduce_multi_core_w";
    spec.kernels = std::move(kernels);
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()},
    };
    spec.work_units = std::move(work_units);

    // ---- Run params: per-core RTAs ----
    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
    uint32_t out_dim_divider = Wt;
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
    TT_FATAL(
        cores.size() == num_cores, "Resolved core list size {} must match split num_cores {}", cores.size(), num_cores);
    TT_FATAL(num_rows == 0 || !cores.empty(), "Non-zero reduce workload requires non-empty core list");

    m2::ProgramRunParams::KernelRunParams reader_params{.kernel_spec_name = W_READER_KERNEL};
    m2::ProgramRunParams::KernelRunParams writer_params{.kernel_spec_name = W_WRITER_KERNEL};

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
        reader_params.named_runtime_args.push_back({
            .node = core,
            .args = {{"num_tiles", num_tensor_tiles_per_core}, {"start_id", num_tiles_read}},
        });
        writer_params.named_runtime_args.push_back({
            .node = core,
            .args =
                {{"num_pages", num_tensor_tiles_per_core / out_dim_divider},
                 {"start_id", num_tiles_read / out_dim_divider}},
        });
        num_tiles_read += num_tensor_tiles_per_core;
        if (i == num_cores - 1) {
            TT_FATAL(
                num_tiles_read == num_rows * Wt,
                "Reduce W assigned {} input tiles, expected {}",
                num_tiles_read,
                num_rows * Wt);
        }
    }

    m2::ProgramRunParams run_params;
    run_params.kernel_run_params.push_back(std::move(reader_params));
    run_params.kernel_run_params.push_back(std::move(writer_params));
    // Compute kernels have no RTAs (everything is CTA) — omit run-params entries.
    run_params.tensor_args = {
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = INPUT_TENSOR, .tensor = std::cref(a.mesh_tensor())},
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = OUTPUT_TENSOR, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
}

}  // namespace ttnn::prim
