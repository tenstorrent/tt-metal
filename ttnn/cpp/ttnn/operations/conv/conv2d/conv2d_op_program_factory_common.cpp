// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_op_program_factory_common.hpp"
#include <umd/device/types/arch.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <umd/device/types/arch.hpp>
#include <unordered_map>
#include <vector>
#include "tt-metalium/assert.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/hal.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/tensor/types.hpp"
namespace ttnn::operations::conv {
namespace conv2d {

constexpr uint32_t l1_scratchpad_CB_size = 64;

// to enable activation reuse feature, we need to allocate space for input needed for
// one output image width + extra space for diff we need to add for each following output image width
uint32_t calculate_act_cb_size_with_reuse(
    const uint32_t act_block_h_tiles,
    const uint32_t act_block_w_tiles,
    const uint32_t output_image_width,
    const uint32_t padded_in_channels,
    const std::array<uint32_t, 2>& kernel_size,
    const uint32_t input_tile_size,
    DataType input_datatype) {
    const uint32_t image_width_tiles = tt::div_up(output_image_width, tt::constants::TILE_HEIGHT);
    const uint32_t reuse_loops = std::ceil(static_cast<float>(act_block_h_tiles) / image_width_tiles);
    const uint32_t image_width_mod_tile = output_image_width % tt::constants::TILE_HEIGHT;
    const uint32_t image_width_tile_leftover =
        image_width_mod_tile == 0 ? 0 : tt::constants::TILE_HEIGHT - image_width_mod_tile;
    tt::DataFormat data_format = datatype_to_dataformat_converter(input_datatype);
    const uint32_t dtype_size_bytes = datum_size(data_format);
    const uint32_t reuse_length = reuse_loops * padded_in_channels * kernel_size[1] *
                                  (1 + image_width_tile_leftover * kernel_size[0]) * dtype_size_bytes;
    const uint32_t reuse_tiles = tt::div_up(reuse_length, input_tile_size);

    return image_width_tiles * act_block_w_tiles + reuse_tiles;
}

std::vector<CBInfo> get_cb_info(
    const DeviceComputeKernelConfig& compute_kernel_config,
    const OptimizedConvBlockConfig& block_config,
    const OptimizedConvParallelizationConfig& pconfig,
    const ttnn::Shape& weights_shape,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> input_shape,
    std::array<uint32_t, 2> dilation,
    const Conv2dConfig& conv_config,
    DataType input_datatype,
    DataType output_datatype,
    std::array<uint32_t, 2> conv_input_shard_shape,
    uint32_t output_image_width,
    bool enable_bias,
    bool is_1d_depthwise_conv,
    bool skip_act_cb_create,
    uint32_t input_channels_padded) {
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

    const TensorMemoryLayout sharding_scheme = conv_config.shard_layout.value();

    const uint32_t per_core_out_matrix_height_ntiles = pconfig.per_core_out_matrix_height_ntile;
    const uint32_t per_core_out_matrix_width_ntiles = pconfig.per_core_out_matrix_width_ntile;
    const uint32_t tilized_act_block_num_tiles = block_config.act_block_h_ntiles * block_config.act_block_w_ntiles;
    uint32_t act_block_num_tiles, act_block_split_num_tiles = 0;
    const uint32_t padded_in_channels = weights_shape[2] / (kernel_size[0] * kernel_size[1]);
    const uint32_t num_blocks_act_h = per_core_out_matrix_height_ntiles / block_config.act_block_h_ntiles;

    const bool split_reader_enabled =
        is_split_reader_supported(sharding_scheme, is_1d_depthwise_conv, block_config.act_block_h_ntiles) &&
        conv_config.force_split_reader.value_or(is_split_reader_viable(
            block_config.act_block_h_ntiles,
            input_channels_padded,
            kernel_size[1],
            tt::tt_metal::hal::get_arch(),
            input_datatype,
            per_core_out_matrix_width_ntiles * block_config.act_block_w_ntiles,
            weights_tile_size,
            dilation[1],
            num_blocks_act_h,
            block_config.act_block_w_ntiles,
            fp32_dest_acc_en,
            output_datatype,
            conv_config.enable_activation_reuse));

    // Block dims
    if (sharding_scheme != TensorMemoryLayout::HEIGHT_SHARDED || !split_reader_enabled || is_1d_depthwise_conv) {
        if (!conv_config.enable_activation_reuse) {
            act_block_num_tiles = block_config.act_block_h_ntiles * block_config.act_block_w_ntiles;
        } else {
            act_block_num_tiles = calculate_act_cb_size_with_reuse(
                block_config.act_block_h_ntiles,
                block_config.act_block_w_ntiles,
                output_image_width,
                padded_in_channels,
                kernel_size,
                input_tile_size,
                input_datatype);
        }
    } else {
        // Calculate split reader parameters
        uint32_t act_block_h_nsubblocks = block_config.act_block_h_ntiles;
        uint32_t act_block_h_nsubblocks_split_last = act_block_h_nsubblocks / 2;
        uint32_t act_block_h_nsubblocks_split = act_block_h_nsubblocks - act_block_h_nsubblocks_split_last;

        if (!conv_config.enable_activation_reuse) {
            act_block_num_tiles = act_block_h_nsubblocks_split * block_config.act_block_w_ntiles;
            act_block_split_num_tiles = act_block_h_nsubblocks_split_last * block_config.act_block_w_ntiles;
        } else {
            act_block_num_tiles = calculate_act_cb_size_with_reuse(
                act_block_h_nsubblocks_split,
                block_config.act_block_w_ntiles,
                output_image_width,
                padded_in_channels,
                kernel_size,
                input_tile_size,
                input_datatype);
            act_block_split_num_tiles = calculate_act_cb_size_with_reuse(
                act_block_h_nsubblocks_split_last,
                block_config.act_block_w_ntiles,
                output_image_width,
                padded_in_channels,
                kernel_size,
                input_tile_size,
                input_datatype);
        }
    }

    const uint32_t weight_matrix_height_ntiles = weights_shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t weight_matrix_width_ntiles = weights_shape[3] / tt::constants::TILE_WIDTH;

    const uint32_t per_core_out_ntiles =
        pconfig.per_core_out_matrix_height_ntile * pconfig.per_core_out_matrix_width_ntile;

    const uint32_t num_blocks_act_w = weight_matrix_height_ntiles / block_config.act_block_w_ntiles;

    const uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    const uint32_t in0_num_blocks_w =
        sharding_scheme == TensorMemoryLayout::BLOCK_SHARDED ? num_blocks_act_w * conv_act_c_blocks : num_blocks_act_w;
    packer_l1_acc = determine_packer_l1_acc(packer_l1_acc, enable_bias, in0_num_blocks_w);
    const tt::tt_metal::DataType partial_dtype =
        packer_l1_acc ? (fp32_dest_acc_en ? DataType::FLOAT32 : DataType::BFLOAT16) : output_datatype;
    const tt::DataFormat partial_df = datatype_to_dataformat_converter(partial_dtype);
    const uint32_t partial_tile_size = tt::tile_size(partial_df);

    const bool is_1d_conv = input_shape[0] != 1 && input_shape[1] == 1;

    {
        // Weights CB
        uint32_t weight_block_num_tiles =
            per_core_out_matrix_width_ntiles *
            (is_1d_depthwise_conv ? block_config.act_block_h_ntiles : block_config.act_block_w_ntiles);
        if (sharding_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
            // If activation reuse is enabled, we already have full inner dim
            const bool enable_fully_buffered_weights = num_blocks_act_h > 1 && !conv_config.enable_activation_reuse;
            if (enable_fully_buffered_weights) {
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
        if (conv_config.enable_act_double_buffer) {
            act_block_num_tiles *= 2;
            act_block_split_num_tiles *= 2;
        }

        const uint32_t act_cb_tile_size =
            sharding_scheme == TensorMemoryLayout::HEIGHT_SHARDED ? input_tile_size : output_tile_size;
        const tt::DataFormat act_cb_data_format =
            sharding_scheme == TensorMemoryLayout::HEIGHT_SHARDED ? conv_input_df : output_df;
        const bool overlap_act_cb = sharding_scheme != TensorMemoryLayout::HEIGHT_SHARDED && skip_act_cb_create;
        cb_info.emplace_back(CBInfo{
            .name = Conv2dCb::ACT,
            .num_pages = overlap_act_cb ? 0 : act_block_num_tiles,
            .page_size = act_cb_tile_size,
            .data_format = act_cb_data_format,
            .overlapped_by_cb = overlap_act_cb ? std::optional<Conv2dCb>(Conv2dCb::ACT_TILIZED) : std::nullopt});
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
    cb_info.emplace_back(CBInfo{
        .name = Conv2dCb::ACT_TILIZED,
        .num_pages = tilized_act_block_num_tiles,
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
        .page_size = is_1d_conv && conv_config.config_tensors_in_dram
                         ? pconfig.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT *
                               6  // 3 indices per output, 2B per index
                         : pconfig.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT * 2,  // 2B per index
        .is_globally_allocated = !conv_config.config_tensors_in_dram,
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
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& all_cores,
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
                    enchantum::to_string(cb.name));
            }
        }

        std::tie(cb.index, cb.handle) =
            tt::tt_metal::create_cb(cb_index++, program, all_cores, cb.page_size, cb.num_pages, cb.data_format, buffer);
        log_trace(
            tt::LogOp,
            "Allocated circular buffer {} with index {}, num pages {}, page size {}, globally allocated: {}",
            enchantum::to_string(cb.name),
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

/**
 * Calculates NOC transfer rate for L1 local transfers using empirical data.
 *
 * The transfer rate follows a clamped linear approximation:
 * - Minimum rate for small transfers (16B)
 * - Linear growth until reaching peak performance
 * - Constant peak rate for larger transfers
 */
static float get_local_l1_noc_transfer_rate(uint32_t transfer_size_bytes, tt::ARCH arch) {
    // Minimum NOC transfer size that was benchmarked
    const uint32_t min_transfer_size_bytes = tt::tt_metal::hal::get_l1_alignment();

    // Architecture-specific performance characteristics
    struct NocPerformanceParams {
        uint32_t linear_growth_threshold_bytes;
        float min_rate_gbps;   // Transfer rate at minimum size
        float peak_rate_gbps;  // Maximum achievable transfer rate
    };

    NocPerformanceParams params = {0, 0.0f, 0.0f};
    switch (arch) {
        case tt::ARCH::BLACKHOLE: params = NocPerformanceParams{4096, 1.124f, 80.48f}; break;
        case tt::ARCH::WORMHOLE_B0: params = NocPerformanceParams{1024, 0.868f, 27.84f}; break;
        default: TT_THROW("Unsupported architecture when calculating NOC transfer rate");
    }

    // Clamp transfer size to the linear growth region
    const uint32_t effective_transfer_size =
        std::clamp(transfer_size_bytes, min_transfer_size_bytes, params.linear_growth_threshold_bytes);

    // Calculate transfer rate using linear interpolation
    const float rate_increase_per_byte =
        (params.peak_rate_gbps - params.min_rate_gbps) /
        static_cast<float>(params.linear_growth_threshold_bytes - min_transfer_size_bytes);

    return params.min_rate_gbps + rate_increase_per_byte * (effective_transfer_size - min_transfer_size_bytes);
}
/**
 * Calculates NOC transfer rate for DRAM transfers using empirical data.
 *
 * The transfer rate follows a clamped linear approximation:
 * - Minimum rate for small transfers (16B)
 * - Linear growth until reaching peak performance
 * - Constant peak rate for larger transfers
 */
static float get_all_dram_noc_transfer_rate(uint32_t transfer_size_bytes, tt::ARCH arch) {
    // Minimum NOC transfer size that was benchmarked
    const uint32_t min_transfer_size_bytes = tt::tt_metal::hal::get_l1_alignment();

    // Architecture-specific performance characteristics
    struct NocPerformanceParams {
        uint32_t linear_growth_threshold_bytes;
        float min_rate_gbps;   // Transfer rate at minimum size
        float peak_rate_gbps;  // Maximum achievable transfer rate
    };

    NocPerformanceParams params = {0, 0.0f, 0.0f};
    switch (arch) {
        case tt::ARCH::BLACKHOLE: params = NocPerformanceParams{2048, 0.671f, 80.885f}; break;
        case tt::ARCH::WORMHOLE_B0: params = NocPerformanceParams{2048, 0.436f, 28.411f}; break;
        default: TT_THROW("Unsupported architecture when calculating DRAM NOC transfer rate");
    }

    // Clamp transfer size to the linear growth region
    const uint32_t effective_transfer_size =
        std::clamp(transfer_size_bytes, min_transfer_size_bytes, params.linear_growth_threshold_bytes);

    // Calculate transfer rate using linear interpolation
    const float rate_increase_per_byte =
        (params.peak_rate_gbps - params.min_rate_gbps) /
        static_cast<float>(params.linear_growth_threshold_bytes - min_transfer_size_bytes);

    return params.min_rate_gbps + rate_increase_per_byte * (effective_transfer_size - min_transfer_size_bytes);
}
/**
 * Calculates NOC transfer rate for multicast L1-linked transfers using empirical data.
 *
 * The transfer rate follows a clamped linear approximation:
 * - Minimum rate for small transfers (16B)
 * - Linear growth until reaching peak performance
 * - Constant peak rate for larger transfers
 */
static float get_mcast_many_l1_linked_noc_transfer_rate(uint32_t transfer_size_bytes, tt::ARCH arch) {
    // Minimum NOC transfer size that was benchmarked
    const uint32_t min_transfer_size_bytes = tt::tt_metal::hal::get_l1_alignment();

    // Architecture-specific performance characteristics
    struct NocPerformanceParams {
        uint32_t linear_growth_threshold_bytes;
        float min_rate_gbps;   // Transfer rate at minimum size
        float peak_rate_gbps;  // Maximum achievable transfer rate
    };

    NocPerformanceParams params = {0, 0.0f, 0.0f};
    switch (arch) {
        case tt::ARCH::BLACKHOLE: params = NocPerformanceParams{65536, 0.182f, 57.677f}; break;
        case tt::ARCH::WORMHOLE_B0: params = NocPerformanceParams{65536, 0.318f, 25.345f}; break;
        default: TT_THROW("Unsupported architecture when calculating multicast L1-linked NOC transfer rate");
    }

    // Clamp transfer size to the linear growth region
    const uint32_t effective_transfer_size =
        std::clamp(transfer_size_bytes, min_transfer_size_bytes, params.linear_growth_threshold_bytes);

    // Calculate transfer rate using linear interpolation
    const float rate_increase_per_byte =
        (params.peak_rate_gbps - params.min_rate_gbps) /
        static_cast<float>(params.linear_growth_threshold_bytes - min_transfer_size_bytes);

    return params.min_rate_gbps + rate_increase_per_byte * (effective_transfer_size - min_transfer_size_bytes);
}
/**
 * Determines if split reader optimization is supported for the given configuration.
 *
 * Split reader requires all of the following conditions:
 * 1. Height-sharded memory layout (enables parallel reading across height dimension)
 * 2. Not a 1D depthwise convolution (incompatible memory access patterns)
 * 3. Multiple activation block height tiles (must have data to split between readers)
 */
bool is_split_reader_supported(
    TensorMemoryLayout memory_layout, bool is_1d_depthwise_conv, uint32_t act_block_h_ntiles) {
    return memory_layout == TensorMemoryLayout::HEIGHT_SHARDED && !is_1d_depthwise_conv && act_block_h_ntiles > 1;
}

static uint32_t get_tilize_cycles_per_tile(
    tt::ARCH arch, DataType input_dtype, DataType output_dtype, bool fp32_dest_acc) {
    // Tilize cycles lookup table: [arch][input_dtype][output_dtype][fp32_dest_acc]
    static const std::map<tt::ARCH, std::map<DataType, std::map<DataType, std::array<uint32_t, 2>>>> tilize_cycles = {
        {tt::ARCH::BLACKHOLE,
         {
             {DataType::FLOAT32,
              {{DataType::FLOAT32, {132, 132}},
               {DataType::BFLOAT16, {132, 80}},
               {DataType::BFLOAT8_B, {132, 77}}}},  // [non-fp32_dest_acc, fp32_dest_acc]
             {DataType::BFLOAT16,
              {{DataType::FLOAT32, {105, 128}},
               {DataType::BFLOAT16, {72, 72}},
               {DataType::BFLOAT8_B, {60, 60}}}}  // [non-fp32_dest_acc, fp32_dest_acc]
         }},
        {tt::ARCH::WORMHOLE_B0,
         {
             {DataType::FLOAT32,
              {{DataType::FLOAT32, {107, 128}},
               {DataType::BFLOAT16, {74, 74}},
               {DataType::BFLOAT8_B, {74, 70}}}},  // [non-fp32_dest_acc, fp32_dest_acc]
             {DataType::BFLOAT16,
              {{DataType::FLOAT32, {92, 117}},
               {DataType::BFLOAT16, {61, 68}},
               {DataType::BFLOAT8_B, {40, 43}}}}  // [non-fp32_dest_acc, fp32_dest_acc]
         }}};

    auto arch_it = tilize_cycles.find(arch);
    if (arch_it == tilize_cycles.end()) {
        TT_THROW("Unsupported architecture when calculating tilize cycles");
    }

    auto input_it = arch_it->second.find(input_dtype);
    if (input_it == arch_it->second.end()) {
        TT_THROW("Unsupported input data type when calculating tilize cycles");
    }

    auto output_it = input_it->second.find(output_dtype);
    if (output_it == input_it->second.end()) {
        TT_THROW("Unsupported output data type when calculating tilize cycles");
    }

    return output_it->second[fp32_dest_acc ? 1 : 0];
}
/*
    Split reader viability is determined by comparing the time required before matmul computation begins.

    NOTE: if activation reuse is enabled, we always enable split reader (for now)

    Thread organization and dependencies differ based on split reader configuration:

    Without split reader:
    - Compute thread: Waits for activations to be read, then performs tilization, then performs matmul
    - Activation reader thread: Reads activations from L1
    - Weight reader thread: Reads weights from DRAM and multicasts to cores

    With split reader enabled:
    - Compute thread: Waits for activations to be read, then performs tilization, then performs matmul
    - Activation reader thread: Reads activations from L1
    - Weight reader thread: Reads activations from L1 before reading weights from DRAM and multicasting to cores

    Two scenarios are compared:
    1. Without split reader (single activation reader):
       - Single activation reader transfers full activation block
       - Weight reader operates in parallel with activation reader
       - Compute waits for activation reading to complete, then tilizes, then does matmul
       - Time before matmul: max(activation_transfer + tilize, weight_transfer)

    2. With split reader (dual activation readers):
       - Two activation readers each transfer half the activation block in parallel
       - Weight reader is delayed and starts after completing activation reading
       - Compute waits for first activation reader to finish (activation_transfer/2), then tilizes
       - Compute waits for tilization to complete and weights to be available before matmul
       - Time before matmul: activation_transfer/2 + max(tilize, weight_transfer)

    Cost calculations (all in clock cycles):
    - Activation cost: (data_bytes / transfer_rate_gbps) * clock_frequency_ghz
    - Weight cost: data_bytes * (1/dram_rate_gbps + 1/mcast_rate_gbps) * clock_frequency_ghz
    - Tilize cost: cycles_per_tile * num_tiles (already in cycles)

    Transfer rates are measured in GB/s and converted to cycles by: (bytes / GB_per_s) * GHz = cycles
    The clock_frequency_ghz represents the actual clock frequency used in the transfer rate lookup tables,
    which is why we use it to convert transfer rates to cycles.
*/
bool is_split_reader_viable(
    uint32_t act_block_h_ntiles,
    uint32_t input_channels_padded,
    uint32_t kernel_width,
    tt::ARCH arch,
    DataType input_datatype,
    uint32_t weights_block_ntiles,
    uint32_t weights_tile_size,
    uint32_t dilation_w,
    uint32_t num_blocks_act_h,
    uint32_t act_block_w_ntiles,
    bool fp32_dest_acc,
    DataType output_datatype,
    bool act_reuse_enabled) {
    // If activation reuse is enabled, we always enable split_reader
    if (act_reuse_enabled) {
        return true;
    }
    // Clock frequency in GHz used in the transfer rate lookup tables for this architecture
    // This is the reference clock frequency that the lookup tables were measured against
    const float clock_frequency_ghz = arch == tt::ARCH::BLACKHOLE ? 1.35f : 1.0f;

    // Calculate activation transfer cost in cycles
    const uint32_t input_bytes_per_element = (input_datatype == DataType::FLOAT32) ? 4 : 2;
    const DataType halo_datatype = input_datatype == DataType::FLOAT32 ? DataType::FLOAT32 : DataType::BFLOAT16;

    // For dilated convs, kernel_width channels aren't sequential in L1, so transfer 1 channel at a time
    const uint32_t coallesced_read_channels = (dilation_w == 1 ? kernel_width : 1);
    const uint32_t noc_transfer_unit_bytes = input_bytes_per_element * input_channels_padded * coallesced_read_channels;

    // Get transfer rate in GB/s for local L1-to-L1 NoC transfers
    const float noc_local_l1_transfer_rate_gbps = get_local_l1_noc_transfer_rate(noc_transfer_unit_bytes, arch);

    // Calculate total activation data size in bytes
    const uint32_t activation_data_bytes = input_bytes_per_element * act_block_h_ntiles * tt::constants::TILE_HEIGHT *
                                           input_channels_padded * kernel_width;

    // Convert to cycles: (bytes / GB_per_s) * GHz = cycles
    const float activation_cycles =
        clock_frequency_ghz * static_cast<float>(activation_data_bytes) / noc_local_l1_transfer_rate_gbps;

    // Calculate weight transfer cost in cycles
    const uint32_t weight_data_bytes = weights_tile_size * weights_block_ntiles;

    // Get transfer rates in GB/s for DRAM-to-L1 and L1-to-L1 multicast
    const float noc_mcast_rate_gbps = get_mcast_many_l1_linked_noc_transfer_rate(weight_data_bytes, arch);
    const float noc_dram_rate_gbps = get_all_dram_noc_transfer_rate(weights_tile_size, arch);

    // Weight transfer involves both DRAM read and multicast (sequential operations)
    // Convert to cycles: bytes * (1/dram_rate + 1/mcast_rate) * GHz = cycles
    const float weight_cycles = clock_frequency_ghz * static_cast<float>(weight_data_bytes) *
                                (1.0f / noc_dram_rate_gbps + 1.0f / noc_mcast_rate_gbps);

    // Calculate tilization cost in cycles (get_tilize_cycles_per_tile already returns cycles per tile)
    const uint32_t total_tiles = act_block_w_ntiles * act_block_h_ntiles;
    const float tilize_cycles =
        get_tilize_cycles_per_tile(arch, halo_datatype, output_datatype, fp32_dest_acc) * total_tiles;

    // Compare scenarios:
    // Single reader: max(activation_cycles + tilize_cycles, weight_cycles)
    // Split reader: activation_cycles/2 + max(weight_cycles, tilize_cycles)
    const bool is_viable = activation_cycles / 2 + std::max(weight_cycles, tilize_cycles) <
                           std::max(activation_cycles + tilize_cycles, weight_cycles);

    log_debug(
        tt::LogOp,
        "Split reader viability: activation_cycles={:.3f}, weight_cycles={:.3f}, tilize_cycles={:.3f}, is_viable={}",
        activation_cycles,
        weight_cycles,
        tilize_cycles,
        is_viable);

    return is_viable;
}

}  // namespace conv2d
}  // namespace ttnn::operations::conv
