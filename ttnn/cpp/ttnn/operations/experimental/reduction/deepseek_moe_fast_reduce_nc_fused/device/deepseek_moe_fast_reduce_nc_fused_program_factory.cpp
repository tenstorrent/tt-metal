// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_program_factory.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

// Matches compile-time arg order in deepseek_moe_fast_reduce_nc_fused_reader.cpp (get_compile_time_arg_val 0..26).
struct DeepseekMoeFastReduceNcFusedReaderCtArgs {
    uint32_t cb_in_act_id{};
    uint32_t cb_scores_id{};
    uint32_t cb_scores_rm_id{};
    uint32_t act_page_size{};
    uint32_t scores_buf_page_size{};
    uint32_t scores_tile_size{};
    uint32_t num_cores{};
    uint32_t input_granularity{};
    uint32_t reduction_dim{};
    uint32_t reduction_dim_size{};
    uint32_t inner_num_tiles{};
    uint32_t reduction_num_tiles{};
    uint32_t num_tokens{};
    uint32_t num_tokens_x32{};
    uint32_t scores_cb_rm_page_size{};
    uint32_t expert_indices_page_size{};
    uint32_t expert_mapping_page_size{};
    uint32_t cluster_axis{};
    uint32_t cb_expert_indices_id{};
    uint32_t cb_expert_mapping_id{};
    uint32_t expert_indices_cb_page_size{};
    uint32_t expert_mapping_cb_page_size{};
    uint32_t expert_indices_num_pages{};
    uint32_t expert_mapping_num_pages{};
    uint32_t mesh_cols{};  // mesh_shape[1]; lets the kernel decode linearized device ids into (row, col)
    uint32_t num_shared_experts{};
    uint32_t shared_expert_scale_bf16{};  // BF16 bit pattern in low 16 bits (upper 16 bits of float32)
};

std::vector<uint32_t> to_reader_ct_arg_vector(const DeepseekMoeFastReduceNcFusedReaderCtArgs& ct) {
    return {
        ct.cb_in_act_id,
        ct.cb_scores_id,
        ct.cb_scores_rm_id,
        ct.act_page_size,
        ct.scores_buf_page_size,  // DRAM page size
        ct.scores_tile_size,
        ct.num_cores,
        ct.input_granularity,
        ct.reduction_dim,
        ct.reduction_dim_size,   // experts_k
        ct.inner_num_tiles,      // the number of tiles in the inner dimensions: [reduction_dim+1,...,rank-1]
        ct.reduction_num_tiles,  // the number of tiles in the reduction dimension: inner_num_tiles * reduction_dim_size
        ct.num_tokens,
        ct.num_tokens_x32,
        ct.scores_cb_rm_page_size,    // cb_scores_rm_id page size: one page = one token row
        ct.expert_indices_page_size,  // DRAM page size for expert_indices_tensor
        ct.expert_mapping_page_size,  // DRAM page size for expert_mapping_tensor
        ct.cluster_axis,
        ct.cb_expert_indices_id,
        ct.cb_expert_mapping_id,
        ct.expert_indices_cb_page_size,  // L1-aligned CB stride for expert_indices
        ct.expert_mapping_cb_page_size,  // L1-aligned CB stride for expert_mapping
        ct.expert_indices_num_pages,
        ct.expert_mapping_num_pages,
        ct.mesh_cols,
        ct.num_shared_experts,
        ct.shared_expert_scale_bf16,
    };
}

// Builds the per-mesh-coord ProgramDescriptor.  The mesh coordinate enters via
// reader runtime args at positions [4]/[5] (row/col), so each coord needs its
// own descriptor.
tt::tt_metal::ProgramDescriptor build_program_descriptor(
    const ttnn::experimental::prim::DeepseekMoEFastReduceNCFusedParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const ttnn::experimental::prim::DeepseekMoEFastReduceNCFusedInputs& tensor_args,
    const std::vector<ttnn::Tensor>& output_tensors) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = tensor_args.input_tensor.device();
    const uint32_t mesh_cols = device->get_view().num_cols();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const ttnn::Tensor& scores_tensor = tensor_args.scores_tensor;
    const ttnn::Tensor& expert_indices_tensor = tensor_args.expert_indices_tensor;
    const ttnn::Tensor& expert_mapping_tensor = tensor_args.expert_mapping_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();

    const uint32_t reduction_dim = operation_attributes.reduce_dim;

    const uint32_t num_tile_elements = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;
    const uint32_t input_tensor_Wt = input_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t slice_Wt = input_tensor_Wt / output_tensors.size();

    uint32_t inner_dims_product = 1;
    for (uint32_t dim = reduction_dim + 1; dim < input_rank; ++dim) {
        inner_dims_product *= input_shape[dim];
    }

    const uint32_t reduction_dim_size = input_shape[reduction_dim];  // experts_k
    const uint32_t inner_num_tiles = inner_dims_product / num_tile_elements;
    const uint32_t reduction_num_tiles = inner_num_tiles * reduction_dim_size;

    // scores shape: [tokens, 1, seq, experts_k] (ROW_MAJOR)
    const uint32_t num_tokens = scores_tensor.logical_shape()[0];  // tokens_per_device
    const uint32_t num_tokens_x32 = round_up(num_tokens, 32);

    // Choose granularity as the largest factor of num_reduce_input_tile that is less than or equal to 8.
    // Helps with locality and increases work unit for better performance.
    uint32_t input_granularity;
    for (input_granularity = 8; input_granularity > 1; --input_granularity) {
        if (reduction_dim_size % input_granularity == 0) {
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t num_output_tiles = input_tensor.physical_volume() / num_tile_elements / reduction_dim_size;

    auto grid = device->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto scores_data_format = datatype_to_dataformat_converter(scores_tensor.dtype());
    const auto output_data_format = datatype_to_dataformat_converter(output_tensors.at(0).dtype());

    const uint32_t input_page_size = input_tensor.buffer()->page_size();
    const uint32_t output_page_size = output_tensors.at(0).buffer()->page_size();
    const uint32_t scores_page_size = scores_tensor.buffer()->page_size();
    const uint32_t scores_cb_rm_page_size =
        round_up(scores_tensor.buffer()->aligned_page_size(), hal::get_l1_alignment());
    const uint32_t expert_indices_page_size = expert_indices_tensor.buffer()->page_size();
    const uint32_t expert_mapping_page_size = expert_mapping_tensor.buffer()->page_size();
    const uint32_t expert_indices_cb_page_size =
        round_up(expert_indices_tensor.buffer()->aligned_page_size(), hal::get_l1_alignment());
    const uint32_t expert_mapping_cb_page_size =
        round_up(expert_mapping_tensor.buffer()->aligned_page_size(), hal::get_l1_alignment());
    const uint32_t expert_indices_num_pages = expert_indices_tensor.buffer()->num_pages();
    const uint32_t expert_mapping_num_pages = expert_mapping_tensor.buffer()->num_pages();

    const auto expert_indices_data_format = datatype_to_dataformat_converter(expert_indices_tensor.dtype());
    const auto expert_mapping_data_format = datatype_to_dataformat_converter(expert_mapping_tensor.dtype());

    const uint32_t scores_rm_cb_num_pages = num_tokens;
    const uint32_t scores_tile_size = tt::tile_size(scores_data_format);

    const uint32_t input_tensor_buffer_factor = input_granularity * 2;
    const uint32_t output_tensor_buffer_factor = 2;

    tt::tt_metal::ProgramDescriptor desc;

    // CB c_0: activation tiles (double-buffered)
    const uint32_t cb_in_act_id = tt::CBIndex::c_0;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = input_tensor_buffer_factor * input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in_act_id),
            .data_format = input_data_format,
            .page_size = input_page_size,
        }}},
    });

    // CB c_1: pre-processed score tiles (one per expert), held resident during compute
    const uint32_t cb_scores_id = tt::CBIndex::c_1;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = reduction_dim_size * scores_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_scores_id),
            .data_format = scores_data_format,
            .page_size = scores_tile_size,
        }}},
    });

    // CB c_2: scratch for reading raw RM scores from DRAM (one page = one token row)
    const uint32_t cb_scores_rm_id = tt::CBIndex::c_2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = scores_rm_cb_num_pages * scores_cb_rm_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_scores_rm_id),
            .data_format = scores_data_format,
            .page_size = scores_cb_rm_page_size,
        }}},
    });

    // CB c_3: full expert_indices_tensor mirrored in L1. Sized to hold every page once;
    // the reader loads all pages up front before the score-tilization prologue.
    const uint32_t cb_expert_indices_id = tt::CBIndex::c_3;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = expert_indices_num_pages * expert_indices_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_expert_indices_id),
            .data_format = expert_indices_data_format,
            .page_size = expert_indices_cb_page_size,
        }}},
    });

    // CB c_4: full expert_mapping_tensor mirrored in L1, same loading pattern as c_3.
    const uint32_t cb_expert_mapping_id = tt::CBIndex::c_4;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = expert_mapping_num_pages * expert_mapping_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_expert_mapping_id),
            .data_format = expert_mapping_data_format,
            .page_size = expert_mapping_cb_page_size,
        }}},
    });

    // CB c_16: output tiles (double-buffered)
    const uint32_t cb_out_id = tt::CBIndex::c_16;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = output_tensor_buffer_factor * output_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_out_id),
            .data_format = output_data_format,
            .page_size = output_page_size,
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Reader CT args: fields 0..26 = DeepseekMoeFastReduceNcFusedReaderCtArgs; then four
    // TensorAccessorArgs (input, scores, expert_indices, expert_mapping) appended in that order.
    // Convert the host-side float32 scale to BF16 by taking the upper 16 bits of its IEEE-754
    // representation (truncation, matching the BF16 layout used elsewhere on device).
    uint32_t shared_expert_scale_f32_bits = 0;
    const float shared_expert_scale_value = operation_attributes.shared_expert_scale;
    std::memcpy(&shared_expert_scale_f32_bits, &shared_expert_scale_value, sizeof(uint32_t));
    const uint32_t shared_expert_scale_bf16 = shared_expert_scale_f32_bits >> 16;

    const DeepseekMoeFastReduceNcFusedReaderCtArgs reader_ct_named{
        .cb_in_act_id = cb_in_act_id,
        .cb_scores_id = cb_scores_id,
        .cb_scores_rm_id = cb_scores_rm_id,
        .act_page_size = input_page_size,
        .scores_buf_page_size = scores_page_size,
        .scores_tile_size = scores_tile_size,
        .num_cores = num_cores,
        .input_granularity = input_granularity,
        .reduction_dim = reduction_dim,
        .reduction_dim_size = reduction_dim_size,
        .inner_num_tiles = inner_num_tiles,
        .reduction_num_tiles = reduction_num_tiles,
        .num_tokens = num_tokens,
        .num_tokens_x32 = num_tokens_x32,
        .scores_cb_rm_page_size = scores_cb_rm_page_size,
        .expert_indices_page_size = expert_indices_page_size,
        .expert_mapping_page_size = expert_mapping_page_size,
        .cluster_axis = operation_attributes.cluster_axis,
        .cb_expert_indices_id = cb_expert_indices_id,
        .cb_expert_mapping_id = cb_expert_mapping_id,
        .expert_indices_cb_page_size = expert_indices_cb_page_size,
        .expert_mapping_cb_page_size = expert_mapping_cb_page_size,
        .expert_indices_num_pages = expert_indices_num_pages,
        .expert_mapping_num_pages = expert_mapping_num_pages,
        .mesh_cols = mesh_cols,
        .num_shared_experts = operation_attributes.num_shared_experts,
        .shared_expert_scale_bf16 = shared_expert_scale_bf16,
    };
    std::vector<uint32_t> reader_ct_args = to_reader_ct_arg_vector(reader_ct_named);
    TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(scores_tensor.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(expert_indices_tensor.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(expert_mapping_tensor.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        cb_out_id,
        output_page_size,
        num_cores,
        input_tensor_Wt,
        slice_Wt,
        static_cast<uint32_t>(output_tensors.size())};
    for (uint32_t i = 0; i < output_tensors.size(); ++i) {
        writer_ct_args.push_back(output_page_size);
    }
    for (const ttnn::Tensor& output_tensor : output_tensors) {
        TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);
    }

    constexpr const char* reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/"
        "deepseek_moe_fast_reduce_nc_fused_reader.cpp";
    constexpr const char* writer_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/"
        "deepseek_moe_fast_reduce_nc_writer.cpp";

    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: packer_l1_acc and dst_full_sync_en from get_compute_kernel_config_args are intentionally
    // not destructured here; the legacy factory did not propagate them either. dst_full_sync_en is
    // also available on ComputeConfigDescriptor but is left at its default to match prior behavior.
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, _packer_l1_acc, _dst_full_sync_en] =
        get_compute_kernel_config_args(input_tensor.device()->arch(), operation_attributes.compute_kernel_config);
    (void)_packer_l1_acc;
    (void)_dst_full_sync_en;

    tt::tt_metal::KernelDescriptor::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    constexpr const char* compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/"
        "deepseek_moe_fast_reduce_nc_fused_compute.cpp";

    tt::tt_metal::KernelDescriptor compute_desc_group_1;
    compute_desc_group_1.kernel_source = compute_kernel_file;
    compute_desc_group_1.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_group_1.core_ranges = core_group_1;
    compute_desc_group_1.compile_time_args = {
        num_cols_per_core_group_1,
        reduction_dim_size,
        input_granularity,
        cb_in_act_id,
        cb_scores_id,
        cb_out_id,
    };
    compute_desc_group_1.defines = compute_defines;
    compute_desc_group_1.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    std::optional<tt::tt_metal::KernelDescriptor> compute_desc_group_2;
    if (!core_group_2.ranges().empty()) {
        compute_desc_group_2.emplace();
        compute_desc_group_2->kernel_source = compute_kernel_file;
        compute_desc_group_2->source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_group_2->core_ranges = core_group_2;
        compute_desc_group_2->compile_time_args = {
            num_cols_per_core_group_2,
            reduction_dim_size,
            input_granularity,
            cb_in_act_id,
            cb_scores_id,
            cb_out_id,
        };
        compute_desc_group_2->defines = compute_defines;
        compute_desc_group_2->config = tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
        };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Each core is assigned an output work unit in a row wise round robin
    // fashion. For a given core, the first index is i, and all subsequent
    // indices are increments of num_cores. The total number of
    // units is num_tiles_per_group times num_cores.
    // For example, with 130 output tiles to be processed on an 8x8 grid
    // - the increment is 64
    // - the first 2 cores will have num_tiles_per_core 3 and the rest 2
    // - core x=0,y=0 will process output tiles 0, 64, and 128
    // - core x=1,y=0 will process output tiles 1, 65, and 129
    // - core x=2,y=0 will process output tiles 2 and 66
    // - core x=3,y=0 will process output tiles 3 and 67
    // - etc
    // The first tile that needs to be reduced has the same as the output tile.
    // That is the starting point for the reader, which then processes all
    // subsequent tiles to be reduced. The increment for the input indices is
    // the size of the inner dimensions in tiles (inner_num_tiles). The number
    // of tiles to process is the size of the reduce dimension in tiles
    // (reduction_num_tiles).
    TT_FATAL(
        core_group_2.ranges().empty() || num_cols_per_core_group_1 >= num_cols_per_core_group_2,
        "num_cols_per_core_group_1 must be greater than or equal to num_cols_per_core_group_2");

    TT_FATAL(
        mesh_coordinate.dims() == 2,
        "deepseek_moe_fast_reduce_nc_fused expects a 2D mesh coordinate, got {}D",
        mesh_coordinate.dims());
    const uint32_t mesh_coord_row = mesh_coordinate[0];
    const uint32_t mesh_coord_col = mesh_coordinate[1];

    const auto core_groups = {core_group_1, core_group_2};

    uint32_t start_tiles_read = 0;
    for (const auto& core_group : core_groups) {
        if (core_group.ranges().empty()) {
            continue;
        }

        uint32_t num_tiles_per_core =
            core_group_1.contains(core_group.ranges().at(0)) ? num_cols_per_core_group_1 : num_cols_per_core_group_2;
        uint32_t page_id_range_length = num_tiles_per_core * num_cores;

        for (const auto& core : corerange_to_cores(core_group)) {
            uint32_t start_tiles_to_read = start_tiles_read + page_id_range_length;

            uint32_t start_slice_row_offset = (start_tiles_read / input_tensor_Wt) * slice_Wt;
            uint32_t start_pages_read_in_row = start_tiles_read % input_tensor_Wt;

            // Reader RT args layout:
            //   [0] input addr
            //   [1] scores addr
            //   [2] start_tiles_read
            //   [3] start_tiles_to_read
            //   [4] mesh_coord_row
            //   [5] mesh_coord_col
            //   [6] expert_indices addr
            //   [7] expert_mapping addr
            // Buffer* entries register as buffer bindings so the framework's
            // cache-hit fast path can patch their addresses without re-running
            // create_workload_descriptor.
            reader_desc.emplace_runtime_args(
                core,
                {input_tensor.buffer(),
                 scores_tensor.buffer(),
                 start_tiles_read,
                 start_tiles_to_read,
                 mesh_coord_row,
                 mesh_coord_col,
                 expert_indices_tensor.buffer(),
                 expert_mapping_tensor.buffer()});

            tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args;
            writer_rt_args.push_back(start_tiles_read);
            writer_rt_args.push_back(start_tiles_to_read);
            writer_rt_args.push_back(start_slice_row_offset);
            writer_rt_args.push_back(start_pages_read_in_row);
            for (const ttnn::Tensor& output_tensor : output_tensors) {
                writer_rt_args.push_back(output_tensor.buffer());
            }
            writer_desc.emplace_runtime_args(core, writer_rt_args);

            start_tiles_read++;
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_group_1));
    if (compute_desc_group_2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_group_2));
    }

    return desc;
}

}  // namespace

namespace ttnn::experimental::prim {

tt::tt_metal::WorkloadDescriptor DeepseekMoEFastReduceNCFusedMeshWorkloadFactory::create_workload_descriptor(
    const DeepseekMoEFastReduceNCFusedParams& operation_attributes,
    const DeepseekMoEFastReduceNCFusedInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    // Per-coord programs: each coord encodes its own (row, col) in reader RT
    // args [4]/[5], so a single shared descriptor would not be correct.  Build
    // one descriptor per individual coordinate.
    const auto coords = tensor_coords.coords();
    workload_descriptor.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = build_program_descriptor(operation_attributes, coord, tensor_args, tensor_return_value);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::experimental::prim
