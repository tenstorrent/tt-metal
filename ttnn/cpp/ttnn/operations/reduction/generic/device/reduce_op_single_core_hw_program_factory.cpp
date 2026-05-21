// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-core HW reduction program factory, ported to the Metal 2.0 host API.
// HW reduces both H and W axes on a single worker core.

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

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

// KernelSpec / WorkUnitSpec ids. Prefixed `HW_` to keep symbols disambiguated
// under Unity-build merging of anonymous-namespace declarations across the
// four reduction factory .cpp files.
constexpr const char* HW_READER_KERNEL = "reduce_hw_reader";
constexpr const char* HW_WRITER_KERNEL = "reduce_hw_writer";
constexpr const char* HW_COMPUTE_KERNEL = "reduce_hw_compute";
constexpr const char* HW_WORK_UNIT = "single_core";

}  // namespace

ttnn::device_operation::ProgramArtifacts ReduceDeviceOperation::ReduceSingleCoreHwProgramFactory::create_program_spec(
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
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    uint32_t Wt = W / tile_width;
    uint32_t Ht = H / tile_height;
    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={}, H={}, W={})", Ht, Wt, H, W);
    TT_FATAL(
        operation_attributes.dim == ReduceOpDim::HW,
        "ReduceSingleCoreHwProgramFactory supports HW dim only, got dim enum value {}",
        static_cast<int>(operation_attributes.dim));

    // The single-core HW path uses REDUCE_SCALAR mode, which applies the
    // scaler twice internally (once per dimension). Here we compensate with
    // sqrt(scaler). However, sqrt of a negative number is NaN, so negative scalers
    // must not reach this code path. Instead negative scalers are handled via the two-step
    // W-then-H path where the scaler is applied once (see the reduce function in reduce_op.cpp).
    TT_FATAL(operation_attributes.scaler >= 0, "Scalar must be non-negative");
    float scaler = std::sqrt(operation_attributes.scaler);

    TT_FATAL(
        H % tile_height == 0 && W % tile_width == 0, "Reduce HW expects tile-aligned padded shape H={}, W={}", H, W);
    uint32_t num_tensor_tiles = NC * H * W / tile_hw;
    const uint32_t num_tensor_tiles_ht_wt = NC * Ht * Wt;
    TT_FATAL(
        num_tensor_tiles == num_tensor_tiles_ht_wt,
        "Reduce HW tile count mismatch: tile_hw path={} vs Ht*Wt path={}",
        num_tensor_tiles,
        num_tensor_tiles_ht_wt);

    CoreCoord selected_core_coord = {0, 0};
    if (operation_attributes.sub_core_grids.has_value() && !operation_attributes.sub_core_grids->ranges().empty()) {
        const auto& r = operation_attributes.sub_core_grids->ranges().front();
        selected_core_coord = r.start_coord;
        TT_FATAL(
            operation_attributes.sub_core_grids->contains(selected_core_coord),
            "Selected core {} must be contained in provided sub_core_grids {}",
            selected_core_coord,
            *operation_attributes.sub_core_grids);
    }
    CoreRange core(selected_core_coord, selected_core_coord);
    CoreRangeSet core_set(core);

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);

    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device");

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(scaler);
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={}, H={}, W={})", Ht, Wt, H, W);
    uint32_t out_dim_divider = Ht * Wt;
    TT_FATAL(
        num_tensor_tiles % out_dim_divider == 0,
        "Reduce HW per-core input tiles {} must be divisible by Ht*Wt={}",
        num_tensor_tiles,
        out_dim_divider);

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
        reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::HW);
    if (use_post_mul) {
        reduce_defines_map["REDUCE_POST_MUL"] = "1";
    }
    const auto reduce_defines = DefinesFromMap(reduce_defines_map);

    // ---- Reader ----
    m2::KernelSpec reader;
    reader.unique_id = HW_READER_KERNEL;
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
    writer.unique_id = HW_WRITER_KERNEL;
    writer.source = m2::KernelSpec::SourceFilePath{
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_metal2.cpp"};
    writer.runtime_arguments_schema.named_runtime_args = {"num_pages", "start_id"};
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

    // ---- Compute ----
    const std::string compute_kernel_path =
        operation_attributes.negate
            ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_hw_neg.cpp"
            : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce.cpp";

    m2::KernelSpec compute;
    compute.unique_id = HW_COMPUTE_KERNEL;
    compute.source = m2::KernelSpec::SourceFilePath{compute_kernel_path};
    compute.compile_time_arg_bindings = {
        {"Ht", Ht},
        {"Wt", Wt},
        {"NC", NC},
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
        // Self-loop DFBs: compute kernel both produces and consumes acc / ineg.
        // Shared accessor name (one dfb::acc_dfb handle drives both directions).
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

    // ---- WorkUnit ----
    m2::WorkUnitSpec work_unit;
    work_unit.unique_id = HW_WORK_UNIT;
    work_unit.kernels = {HW_READER_KERNEL, HW_WRITER_KERNEL, HW_COMPUTE_KERNEL};
    work_unit.target_nodes = core_set;

    // ---- Assemble ----
    m2::ProgramSpec spec;
    spec.program_id = "ttnn::reduce_single_core_hw";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()},
    };
    spec.work_units = {std::move(work_unit)};

    // ---- Run params ----
    m2::ProgramRunParams run_params;
    run_params.kernel_run_params = {
        m2::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = HW_READER_KERNEL,
            .named_runtime_args = {{
                .node = selected_core_coord,
                .args = {{"num_tiles", num_tensor_tiles}, {"start_id", 0u}},
            }},
        },
        m2::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = HW_WRITER_KERNEL,
            .named_runtime_args = {{
                .node = selected_core_coord,
                .args = {{"num_pages", num_tensor_tiles / out_dim_divider}, {"start_id", 0u}},
            }},
        },
    };
    run_params.tensor_args = {
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = INPUT_TENSOR, .tensor = std::cref(a.mesh_tensor())},
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = OUTPUT_TENSOR, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
}

}  // namespace ttnn::prim
