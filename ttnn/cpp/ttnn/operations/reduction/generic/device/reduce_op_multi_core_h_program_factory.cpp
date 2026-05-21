// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core H reduction program factory, Metal 2.0 host-API port.
// Two structural variants in one factory:
//   - Interleaved (input/output NOT WIDTH_SHARDED): cross-op writer fork.
//   - Width-sharded (input AND output WIDTH_SHARDED): borrowed-memory DFBs
//     for input shard (c_1) and output (c_3), and the sharded writer fork.

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/metal2_artifacts.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

// Unity-build hygiene: `H_` prefix.
constexpr const char* H_READER = "reader";
constexpr const char* H_WRITER = "writer";
constexpr const char* H_COMPUTE_G1 = "compute_g1";
constexpr const char* H_COMPUTE_G2 = "compute_g2";

constexpr const char* H_WU_G1 = "wu_g1";
constexpr const char* H_WU_G2 = "wu_g2";
constexpr const char* H_WU_MAIN = "wu_main";  // width-sharded single group

constexpr const char* H_INPUT_DFB = "input";
constexpr const char* H_INPUT_SHARDED_DFB = "input_sharded";  // borrowed (c_1)
constexpr const char* H_SCALER_DFB = "scaler";
constexpr const char* H_OUTPUT_DFB = "output";
constexpr const char* H_ACC_DFB = "acc";
constexpr const char* H_INEG_DFB = "ineg";

constexpr const char* H_INPUT_TENSOR = "input";
constexpr const char* H_OUTPUT_TENSOR = "output";

m2::KernelSpec::CompilerOptions::Defines h_defines_from_map(const std::map<std::string, std::string>& src) {
    m2::KernelSpec::CompilerOptions::Defines out;
    out.reserve(src.size());
    for (const auto& [k, v] : src) {
        out.emplace_back(k, v);
    }
    return out;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts ReduceDeviceOperation::ReduceMultiCoreHProgramFactory::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;
    const uint32_t HtWt = Ht * Wt;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    const tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    const tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = a.device();

    const bool use_width_sharding = a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                                    output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    const uint32_t chunk_size =
        use_width_sharding ? 1u : ttnn::get_dest_reg_count(operation_attributes.compute_kernel_config);

    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cols = NC * Wt;

    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cols_per_core_group_1, num_cols_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_cols);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_cols);
    }
    TT_FATAL(num_cores > 0, "Reduce H requires at least one worker core");

    if (use_width_sharding) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_cols_per_core_group_1 = NC * (a.shard_spec().value().shape[1] / tile_width);
        num_cols_per_core_group_2 = 0;
    }

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::H);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    // ----- DataflowBufferSpecs -----

    constexpr uint32_t kNumInputEntriesInterleaved_nonNegate = 2;
    constexpr uint32_t kNumScalerEntries = 1;
    constexpr uint32_t kNumOutputEntriesInterleaved_nonNegate = 2;

    std::vector<m2::DataflowBufferSpec> dataflow_buffers;

    if (use_width_sharding) {
        const uint32_t num_shard_tiles = a.shard_spec().value().numel() / tile_hw;

        // c_0 -- local input scratch
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = H_INPUT_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = 2,
            .data_format_metadata = src0_cb_data_format,
            .tile_format_metadata = a.tensor_spec().tile(),
        });
        // c_1 -- borrowed-memory DFB backed by the input tensor's shard
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = H_INPUT_SHARDED_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_shard_tiles,
            .data_format_metadata = src0_cb_data_format,
            .tile_format_metadata = a.tensor_spec().tile(),
            .borrowed_from = H_INPUT_TENSOR,
        });
    } else {
        const uint32_t num_input_tiles =
            operation_attributes.negate ? chunk_size : kNumInputEntriesInterleaved_nonNegate;
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = H_INPUT_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = src0_cb_data_format,
            .tile_format_metadata = a.tensor_spec().tile(),
        });
    }

    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = H_SCALER_DFB,
        .entry_size = scaler_single_tile_size,
        .num_entries = kNumScalerEntries,
        .data_format_metadata = scaler_cb_data_format,
    });

    if (use_width_sharding) {
        const uint32_t num_output_tiles = output.shard_spec().value().numel() / tile_hw;
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = H_OUTPUT_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
            .tile_format_metadata = output.tensor_spec().tile(),
            .borrowed_from = H_OUTPUT_TENSOR,
        });
    } else {
        const uint32_t num_output_tiles =
            operation_attributes.negate ? chunk_size : kNumOutputEntriesInterleaved_nonNegate;
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = H_OUTPUT_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
            .tile_format_metadata = output.tensor_spec().tile(),
        });
    }

    // Negate path is only valid on the interleaved branch (the legacy code did not
    // support negate on the width-sharded branch).
    if (operation_attributes.negate) {
        TT_FATAL(!use_width_sharding, "Reduce H negate is not supported on the width-sharded path");

        const uint32_t compute_Wt_g1 = num_cols_per_core_group_1;
        const uint32_t compute_Wt_g2 = num_cols_per_core_group_2;
        uint32_t per_nc_advance = 0;
        if (compute_Wt_g2 == 0) {
            per_nc_advance = compute_Wt_g1;
        } else if (compute_Wt_g1 == 0) {
            per_nc_advance = compute_Wt_g2;
        } else {
            per_nc_advance = std::lcm(compute_Wt_g1, compute_Wt_g2);
        }
        TT_FATAL(
            per_nc_advance > 0,
            "Negate H reduce: per-core Wt resolved to 0 (g1={}, g2={}, NC={})",
            compute_Wt_g1,
            compute_Wt_g2,
            NC);
        const uint64_t negate_cb_tiles = static_cast<uint64_t>(Ht) * per_nc_advance;
        const uint64_t per_cb_bytes = negate_cb_tiles * dst_single_tile_size;
        const uint64_t negate_cb_bytes = 2ull * per_cb_bytes;
        const auto lowest_address = device->lowest_occupied_compute_l1_address();
        uint64_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
        const uint64_t base_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        TT_FATAL(
            max_l1_space > base_addr,
            "Negate H reduce: L1 base allocator address {} >= lowest occupied address {}; no room for CBs",
            base_addr,
            max_l1_space);
        max_l1_space -= base_addr;
        TT_FATAL(
            negate_cb_bytes <= max_l1_space,
            "Negate H reduce: cb_acc + cb_ineg ({} B for {} tiles) would not fit in available L1 ({} B). "
            "Caller must use h_reduce_negate_fits_in_l1 to choose the external-negate fallback.",
            negate_cb_bytes,
            negate_cb_tiles,
            max_l1_space);
        TT_FATAL(
            per_cb_bytes <= std::numeric_limits<uint32_t>::max(),
            "Negate H reduce: per-CB size {} B exceeds uint32_t total_size range",
            per_cb_bytes);
        const uint32_t per_cb_num_entries = static_cast<uint32_t>(negate_cb_tiles);
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = H_ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = per_cb_num_entries,
            .data_format_metadata = dst_cb_data_format,
            .disable_implicit_sync = true,
        });
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = H_INEG_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = per_cb_num_entries,
            .data_format_metadata = dst_cb_data_format,
            .disable_implicit_sync = true,
        });
    }

    // ----- KernelSpecs -----

    m2::KernelSpec reader{
        .unique_id = H_READER,
        .compiler_options =
            {
                .defines = h_defines_from_map(reduce_defines),
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

    if (use_width_sharding) {
        // Width-sharded reader: reads from the borrowed-memory input shard,
        // pushes through cb_in0; needs REDUCE_SCALER + DEST config defines.
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp"};
        std::map<std::string, std::string> reader_defines = reduce_defines;
        reader_defines["REDUCE_SCALER"] = "1";
        reader_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader.compiler_options.defines = h_defines_from_map(reader_defines);
        reader.dfb_bindings = {
            {.dfb_spec_name = H_INPUT_DFB,
             .local_accessor_name = "input",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
            {.dfb_spec_name = H_INPUT_SHARDED_DFB,
             .local_accessor_name = "input_sharded",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
            {.dfb_spec_name = H_SCALER_DFB,
             .local_accessor_name = "scaler",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
        };
        reader.compile_time_arg_bindings = {{"scaler_bits", scaler_bits}};
        reader.runtime_arguments_schema = {
            .named_runtime_args = {"num_tiles", "Wt", "Ht", "batch", "row_size_bytes", "batch_size_bytes"}};
        // No TensorBinding (no TensorAccessor needed — reads via borrowed-memory DFB).
    } else {
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp"};
        std::map<std::string, std::string> reader_defines = reduce_defines;
        reader_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader.compiler_options.defines = h_defines_from_map(reader_defines);
        reader.dfb_bindings = {
            {.dfb_spec_name = H_INPUT_DFB,
             .local_accessor_name = "input",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
            {.dfb_spec_name = H_SCALER_DFB,
             .local_accessor_name = "scaler",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
        };
        reader.tensor_bindings = {
            {.tensor_parameter_name = H_INPUT_TENSOR, .accessor_name = "input"},
        };
        reader.compile_time_arg_bindings = {
            {"Ht", Ht}, {"Wt", Wt}, {"HtWt", HtWt}, {"scaler_bits", scaler_bits}, {"use_welford", 0u}};
        reader.runtime_arguments_schema = {
            .named_runtime_args = {"col_start_tile_id", "curr_col_in_batch", "num_cols"}};
    }

    m2::KernelSpec writer{
        .unique_id = H_WRITER,
        .config_spec =
            m2::DataMovementConfiguration{
                .gen1_data_movement_config =
                    m2::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };

    if (use_width_sharding) {
        writer.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp"};
        writer.dfb_bindings = {
            {.dfb_spec_name = H_OUTPUT_DFB,
             .local_accessor_name = "output",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
        };
        writer.runtime_arguments_schema = {.named_runtime_args = {"num_units"}};
        // No TensorBinding — borrowed-memory output DFB.
    } else {
        writer.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp"};
        writer.compiler_options.defines = h_defines_from_map(reduce_defines);
        writer.dfb_bindings = {
            {.dfb_spec_name = H_OUTPUT_DFB,
             .local_accessor_name = "output",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
        };
        writer.tensor_bindings = {
            {.tensor_parameter_name = H_OUTPUT_TENSOR, .accessor_name = "output"},
        };
        writer.runtime_arguments_schema = {.named_runtime_args = {"num_pages", "start_id"}};
    }

    // Compute kernel(s)
    const std::string compute_kernel_source =
        std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/") +
        (operation_attributes.negate ? "reduce_h_neg.cpp" : "reduce.cpp");

    auto make_compute = [&](const char* unique_id, uint32_t compute_Wt, uint32_t compute_NC) {
        m2::KernelSpec spec{
            .unique_id = unique_id,
            .source = m2::KernelSpec::SourceFilePath{compute_kernel_source},
            .compiler_options =
                {
                    .defines = h_defines_from_map(reduce_defines),
                },
            .dfb_bindings =
                {
                    {.dfb_spec_name = H_INPUT_DFB,
                     .local_accessor_name = "input",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
                    {.dfb_spec_name = H_SCALER_DFB,
                     .local_accessor_name = "scaler",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
                    {.dfb_spec_name = H_OUTPUT_DFB,
                     .local_accessor_name = "output",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
                },
            .compile_time_arg_bindings = {{"Ht", Ht}, {"Wt", compute_Wt}, {"NC", compute_NC}},
        };
        m2::ComputeConfiguration cc{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
        };
        if (fp32_dest_acc_en) {
            if (src0_cb_data_format == tt::DataFormat::Float32) {
                cc.unpack_to_dest_mode.push_back({H_INPUT_DFB, tt::tt_metal::UnpackToDestMode::Default});
            }
            if (scaler_cb_data_format == tt::DataFormat::Float32) {
                cc.unpack_to_dest_mode.push_back({H_SCALER_DFB, tt::tt_metal::UnpackToDestMode::Default});
            }
            if (operation_attributes.negate && dst_cb_data_format == tt::DataFormat::Float32) {
                cc.unpack_to_dest_mode.push_back({H_ACC_DFB, tt::tt_metal::UnpackToDestMode::Default});
                cc.unpack_to_dest_mode.push_back({H_INEG_DFB, tt::tt_metal::UnpackToDestMode::Default});
            }
        }
        spec.config_spec = cc;
        if (use_post_mul) {
            spec.compile_time_arg_bindings.push_back({"post_mul_scaler_bits", post_mul_scaler_bits});
        }
        if (operation_attributes.negate) {
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = H_ACC_DFB,
                 .local_accessor_name = "acc",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER});
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = H_ACC_DFB,
                 .local_accessor_name = "acc",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER});
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = H_INEG_DFB,
                 .local_accessor_name = "ineg",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER});
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = H_INEG_DFB,
                 .local_accessor_name = "ineg",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER});
        }
        return spec;
    };

    const uint32_t compute_Wt_g1 = use_width_sharding ? (num_cols_per_core_group_1 / NC) : num_cols_per_core_group_1;
    const uint32_t compute_NC_g1 = use_width_sharding ? NC : 1u;
    m2::KernelSpec compute_g1 = make_compute(H_COMPUTE_G1, compute_Wt_g1, compute_NC_g1);

    std::optional<m2::KernelSpec> compute_g2;
    const bool g2_present = !core_group_2.ranges().empty();
    if (g2_present) {
        const uint32_t compute_Wt_g2 =
            use_width_sharding ? (num_cols_per_core_group_2 / NC) : num_cols_per_core_group_2;
        const uint32_t compute_NC_g2 = use_width_sharding ? NC : 1u;
        compute_g2 = make_compute(H_COMPUTE_G2, compute_Wt_g2, compute_NC_g2);
    }

    // ----- WorkUnitSpecs -----

    std::vector<m2::WorkUnitSpec> work_units;
    if (use_width_sharding) {
        work_units.push_back(m2::WorkUnitSpec{
            .unique_id = H_WU_MAIN,
            .kernels = {H_READER, H_WRITER, H_COMPUTE_G1},
            .target_nodes = all_cores,
        });
    } else {
        work_units.push_back(m2::WorkUnitSpec{
            .unique_id = H_WU_G1,
            .kernels = {H_READER, H_WRITER, H_COMPUTE_G1},
            .target_nodes = core_group_1,
        });
        if (g2_present) {
            work_units.push_back(m2::WorkUnitSpec{
                .unique_id = H_WU_G2,
                .kernels = {H_READER, H_WRITER, H_COMPUTE_G2},
                .target_nodes = core_group_2,
            });
        }
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
        .program_id = "reduce_multi_core_h",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                {.unique_id = H_INPUT_TENSOR, .spec = a.tensor_spec()},
                {.unique_id = H_OUTPUT_TENSOR, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    // ----- ProgramRunParams -----

    m2::ProgramRunParams run_params;
    m2::ProgramRunParams::KernelRunParams reader_rp{.kernel_spec_name = H_READER};
    m2::ProgramRunParams::KernelRunParams writer_rp{.kernel_spec_name = H_WRITER};

    if (use_width_sharding) {
        TT_FATAL(NC != 0, "Batch size NC must be non-zero (shape[0]={}, shape[1]={})", shape[0], shape[1]);
        const uint32_t shard_Wt = num_cols_per_core_group_1 / NC;
        const uint32_t shard_row_size = shard_Wt * src0_single_tile_size;
        const uint32_t shard_batch_size = shard_row_size * Ht;
        const uint32_t num_units = num_cols_per_core_group_1;
        const uint32_t reader_num_tiles = num_cols_per_core_group_1 * Ht;
        // Width-sharded path: iterate the actual shard core set (all_cores).
        for (const auto& range : all_cores.ranges()) {
            for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    CoreCoord core{x, y};
                    reader_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                        .node = core,
                        .args =
                            {{"num_tiles", reader_num_tiles},
                             {"Wt", shard_Wt},
                             {"Ht", Ht},
                             {"batch", NC},
                             {"row_size_bytes", shard_row_size},
                             {"batch_size_bytes", shard_batch_size}},
                    });
                    writer_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                        .node = core,
                        .args = {{"num_units", num_units}},
                    });
                }
            }
        }
    } else {
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
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
            cores.size() == num_cores,
            "Resolved core list size {} must match split num_cores {}",
            cores.size(),
            num_cores);

        uint32_t num_cols_read = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_cols_per_core = 0;
            if (core_group_1.contains(core)) {
                num_cols_per_core = num_cols_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_cols_per_core = num_cols_per_core_group_2;
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
            writer_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
                .node = core,
                .args = {{"num_pages", num_cols_per_core}, {"start_id", num_cols_read}},
            });
            num_cols_read += num_cols_per_core;
        }
    }

    run_params.kernel_run_params = {std::move(reader_rp), std::move(writer_rp)};

    run_params.tensor_args = {
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = H_INPUT_TENSOR, .tensor = std::cref(a.mesh_tensor())},
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = H_OUTPUT_TENSOR, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
