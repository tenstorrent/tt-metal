// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_op_program_factory_common.hpp"
#include <cstdint>
#include <optional>
#include <vector>
#include "tt-metalium/assert.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/hal.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/operations/cb_utils.hpp"
namespace ttnn::operations::conv {
namespace conv2d {

constexpr uint32_t l1_scratchpad_CB_size = 64;

std::vector<CBInfo> get_cb_info(
    const DeviceComputeKernelConfig& compute_kernel_config,
    const OptimizedConvBlockConfig& block_config,
    const OptimizedConvParallelizationConfig& pconfig,
    const ttnn::Shape& weights_shape,
    std::array<uint32_t, 2> kernel_size,
    const Conv2dConfig& conv_config,
    DataType input_datatype,
    DataType output_datatype,
    std::array<uint32_t, 2> conv_input_shard_shape,
    bool enable_bias,
    bool is_1d_depthwise_conv,
    bool skip_act_cb_create) {
    const uint32_t num_cbs = static_cast<uint32_t>(Conv2dCb::COUNT);
    std::vector<CBInfo> cb_info;
    cb_info.reserve(num_cbs);

    const bool untilize_out = conv_config.output_layout == Layout::ROW_MAJOR;

    // Tile dimensions and data formats

    // Output of halo op is always ROW_MAJOR, so input for convs is either DataType::FLOAT32 or DataType::BFLOAT16
    const tt::tt_metal::DataType conv_input_dtype = (input_datatype == tt::tt_metal::DataType::FLOAT32)
                                                        ? tt::tt_metal::DataType::FLOAT32
                                                        : tt::tt_metal::DataType::BFLOAT16;
    const uint32_t input_datum_size = conv_input_dtype == tt::tt_metal::DataType::FLOAT32 ? 4 : 2;
    const tt::DataFormat conv_input_df = datatype_to_dataformat_converter(conv_input_dtype);
    const uint32_t input_tile_size = tt::tile_size(datatype_to_dataformat_converter(conv_input_dtype));

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), compute_kernel_config);

    TT_FATAL(conv_config.weights_dtype.has_value(), "get_cb_info expects conv_config.weights_dtype to be already set");
    const tt::DataFormat weights_df = datatype_to_dataformat_converter(conv_config.weights_dtype.value());
    const tt::DataFormat bias_df = weights_df;
    const tt::DataFormat output_df = datatype_to_dataformat_converter(output_datatype);

    const uint32_t weights_tile_size = tt::tile_size(weights_df);
    const uint32_t bias_tile_size = weights_tile_size;
    const uint32_t output_tile_size = tt::tile_size(output_df);

    // Block dims
    const uint32_t act_block_num_tiles = block_config.act_block_h_ntiles * block_config.act_block_w_ntiles;
    const uint32_t weight_matrix_height_ntiles = weights_shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t weight_matrix_width_ntiles = weights_shape[3] / tt::constants::TILE_WIDTH;

    const uint32_t per_core_out_matrix_width_ntiles = pconfig.per_core_out_matrix_width_ntile;
    const uint32_t per_core_out_matrix_height_ntiles = pconfig.per_core_out_matrix_height_ntile;
    const uint32_t per_core_out_ntiles =
        pconfig.per_core_out_matrix_height_ntile * pconfig.per_core_out_matrix_width_ntile;

    const uint32_t num_blocks_act_h = per_core_out_matrix_height_ntiles / block_config.act_block_h_ntiles;

    const uint32_t num_blocks_act_w = weight_matrix_height_ntiles / block_config.act_block_w_ntiles;

    const TensorMemoryLayout sharding_scheme = conv_config.shard_layout.value();
    const uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    const uint32_t in0_num_blocks_w =
        sharding_scheme == TensorMemoryLayout::BLOCK_SHARDED ? num_blocks_act_w * conv_act_c_blocks : num_blocks_act_w;
    packer_l1_acc = determine_packer_l1_acc(packer_l1_acc, enable_bias, in0_num_blocks_w);
    const tt::tt_metal::DataType partial_dtype =
        packer_l1_acc ? (fp32_dest_acc_en ? DataType::FLOAT32 : DataType::BFLOAT16) : output_datatype;
    const tt::DataFormat partial_df = datatype_to_dataformat_converter(partial_dtype);
    const uint32_t partial_tile_size = tt::tile_size(partial_df);

    {
        // Weights CB
        uint32_t weight_block_num_tiles =
            per_core_out_matrix_width_ntiles *
            (is_1d_depthwise_conv ? block_config.act_block_h_ntiles : block_config.act_block_w_ntiles);
        if (sharding_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
            if (num_blocks_act_h > 1) {
                // Fully buffered weights
                weight_block_num_tiles *= kernel_size[0];
            } else if (conv_config.enable_weights_double_buffer) {
                weight_block_num_tiles *= 2;
            }
        } else if (conv_config.enable_weights_double_buffer) {
            weight_block_num_tiles *= 2;
        }

        cb_info.emplace_back(CBInfo{
            .name = Conv2dCb::WEIGHTS,
            .num_pages = weight_block_num_tiles,
            .page_size = weights_tile_size,
            .data_format = weights_df});
    }

    // Matmul partials CB
    cb_info.emplace_back(CBInfo{
        .name = Conv2dCb::MATMUL_PARTIALS,
        .num_pages = is_1d_depthwise_conv ? 1 : per_core_out_ntiles,
        .page_size = partial_tile_size,
        .is_globally_allocated = (!untilize_out && partial_dtype == output_datatype && !is_1d_depthwise_conv),
        .data_format = partial_df});

    {
        // ACT and ACT_SECOND_READER CB
        uint32_t act_cb_num_tiles = act_block_num_tiles;
        uint32_t act_block_split_num_tiles = 0;
        if (sharding_scheme == TensorMemoryLayout::HEIGHT_SHARDED && conv_config.enable_split_reader) {
            uint32_t act_block_h_nsubblocks = block_config.act_block_h_ntiles / block_config.out_subblock_h_ntiles;
            uint32_t act_block_h_nsubblocks_split_last = act_block_h_nsubblocks / 2;
            uint32_t act_block_h_nsubblocks_split = act_block_h_nsubblocks - act_block_h_nsubblocks_split_last;

            act_cb_num_tiles =
                act_block_h_nsubblocks_split * block_config.out_subblock_h_ntiles * block_config.act_block_w_ntiles;
            act_block_split_num_tiles = act_block_h_nsubblocks_split_last * block_config.out_subblock_h_ntiles *
                                        block_config.act_block_w_ntiles;
        }
        if (conv_config.enable_act_double_buffer) {
            act_cb_num_tiles *= 2;
            act_block_split_num_tiles *= 2;
        }

        const uint32_t act_cb_tile_size =
            sharding_scheme == TensorMemoryLayout::HEIGHT_SHARDED ? input_tile_size : output_tile_size;
        const tt::DataFormat act_cb_data_format =
            sharding_scheme == TensorMemoryLayout::HEIGHT_SHARDED ? conv_input_df : output_df;
        cb_info.emplace_back(CBInfo{
            .name = Conv2dCb::ACT,
            .num_pages = skip_act_cb_create ? 0 : act_cb_num_tiles,
            .page_size = act_cb_tile_size,
            .data_format = act_cb_data_format,
            .overlapped_by_cb = skip_act_cb_create ? std::optional<Conv2dCb>(Conv2dCb::ACT_TILIZED) : std::nullopt});
        cb_info.emplace_back(CBInfo{
            .name = Conv2dCb::ACT_SECOND_READER,
            .num_pages = act_block_split_num_tiles,
            .page_size = act_cb_tile_size,
            .data_format = act_cb_data_format});
    }

    // Temp sum CB (1d depthwise conv only)
    cb_info.emplace_back(CBInfo{
        .name = Conv2dCb::TEMP_SUM,
        .num_pages = is_1d_depthwise_conv ? 1 : 0,
        .page_size = output_tile_size,
        .data_format = output_df});

    // Tilized act CB
    const uint32_t tlized_act_cb_num_tiles = act_block_num_tiles;
    cb_info.emplace_back(CBInfo{
        .name = Conv2dCb::ACT_TILIZED,
        .num_pages = act_block_num_tiles,
        .page_size = output_tile_size,
        .data_format = output_df});

    // Bias CB
    cb_info.emplace_back(CBInfo{
        .name = Conv2dCb::BIAS,
        .num_pages = enable_bias ? per_core_out_matrix_width_ntiles : 0,
        .page_size = bias_tile_size,
        .data_format = bias_df});

    {
        // Act row major CB
        uint32_t row_major_act_cb_num_tiles = 0;
        if (sharding_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
            row_major_act_cb_num_tiles = block_config.act_block_w_ntiles * 2;
        } else if (sharding_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
            row_major_act_cb_num_tiles = act_block_num_tiles;
        }

        const bool overlap_act_cb =
            sharding_scheme == TensorMemoryLayout::BLOCK_SHARDED && conv_input_df == output_df && !skip_act_cb_create;
        cb_info.emplace_back(CBInfo{
            .name = Conv2dCb::ACT_ROW_MAJOR_BFLOAT16,
            .num_pages = overlap_act_cb ? 0 : row_major_act_cb_num_tiles,
            .page_size = input_tile_size,
            .data_format = conv_input_df,
            .overlapped_by_cb = overlap_act_cb ? std::optional<Conv2dCb>(Conv2dCb::ACT) : std::nullopt});
    }

    // Output CB
    cb_info.emplace_back(CBInfo{
        .name = Conv2dCb::OUT,
        .num_pages = per_core_out_ntiles,
        .page_size = output_tile_size,
        .is_globally_allocated = true,
        .data_format = output_df});

    // Reader indices CB
    cb_info.emplace_back(CBInfo{
        .name = Conv2dCb::READER_INDICES,
        .num_pages = 1,
        .page_size = pconfig.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT * 2,  // 2B per indexß
        .is_globally_allocated = true,
        .data_format = tt::DataFormat::UInt16});

    // L1 scratchpad CB
    cb_info.emplace_back(CBInfo{
        .name = Conv2dCb::L1_ARRAY,
        .num_pages = 1,
        .page_size = l1_scratchpad_CB_size,
        .data_format = tt::DataFormat::Float16_b});

    // Act sharded CB
    cb_info.emplace_back(CBInfo{
        .name = Conv2dCb::ACT_SHARDED,
        .num_pages = conv_input_shard_shape[0],
        .page_size = conv_input_shard_shape[1] * input_datum_size,
        .is_globally_allocated = true,
        .data_format = conv_input_df});

    TT_FATAL(cb_info.size() == num_cbs, "Expected info for {} cbs  by got {}!", num_cbs, cb_info.size());
    return cb_info;
}

void allocate_cbs(
    std::vector<CBInfo>& cb_info,
    tt::tt_metal::Program& program,
    const CoreRange& all_cores,
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& l1_indices_tensor) {
    uint32_t cb_index = 0;
    for (auto& cb : cb_info) {
        if (cb.num_pages == 0) {
            // Skip circular buffers with zero pages
            continue;
        }

        // cbs for sharded tensors.
        Buffer* buffer = nullptr;
        if (cb.is_globally_allocated) {
            if (cb.name == Conv2dCb::ACT_SHARDED) {
                buffer = input_tensor.buffer();
            } else if (cb.name == Conv2dCb::OUT || cb.name == Conv2dCb::MATMUL_PARTIALS) {
                buffer = output_tensor.buffer();
            } else if (cb.name == Conv2dCb::READER_INDICES) {
                buffer = l1_indices_tensor.buffer();
            } else {
                TT_THROW(
                    "Unexpected circular buffer name {}. Expected one of: SHARDED_ACT_CB, OUT0_CB, READER_INDICES_CB",
                    magic_enum::enum_name(cb.name));
            }
        }

        std::tie(cb.index, cb.handle) =
            tt::tt_metal::create_cb(cb_index++, program, all_cores, cb.page_size, cb.num_pages, cb.data_format, buffer);
        log_debug(
            tt::LogOp,
            "Allocated circular buffer {} with index {}, num pages {}, page size {}, globally allocated: {}",
            magic_enum::enum_name(cb.name),
            cb.index,
            cb.num_pages,
            cb.page_size,
            cb.is_globally_allocated);
    }

    for (auto& cb : cb_info) {
        if (cb.overlapped_by_cb.has_value()) {
            // If this CB is overlapped by another CB, set the handle to the overlapped CB's handle
            const CBInfo& overlapped_cb = get_cb_info_by_name(cb_info, cb.overlapped_by_cb.value());
            cb.handle = overlapped_cb.handle;
            cb.index = overlapped_cb.index;
        }
    }
}

const CBInfo& get_cb_info_by_name(const std::vector<CBInfo>& cb_info, Conv2dCb cb_name) {
    auto it = std::find_if(cb_info.begin(), cb_info.end(), [cb_name](const CBInfo& cb) { return cb.name == cb_name; });
    return *it;
}
CBInfo& access_cb_info_by_name(const std::vector<CBInfo>& cb_info, Conv2dCb cb_name) {
    return const_cast<CBInfo&>(get_cb_info_by_name(cb_info, cb_name));
}

}  // namespace conv2d
}  // namespace ttnn::operations::conv
