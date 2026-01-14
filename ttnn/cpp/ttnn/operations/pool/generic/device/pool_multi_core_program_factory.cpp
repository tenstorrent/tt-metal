// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "pool_op.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/buffer.hpp"
#include "ttnn/tensor/types.hpp"
#include <cstdint>
#include <optional>
#include <vector>
#include "ttnn/tensor/storage.hpp"
#include <tt-metalium/hal.hpp>
#include <algorithm>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::pool {
/**
 * Generic pool implementation that uses the new sliding window infrastructure.
 */
struct ScalarInfo {
    // Scalar Info is used to store the information abpou the scalar used in avg pool op
    // start and end refer to indices of the output stick core is calculating.
    // These are directly mapped to the for loop that can be found in reader and compute kernel of the pool op
    // for (uint32_t i = 0; i < nsticks_per_core; ++i), start is first stick which should be reduced and multiplied by
    // scalar value, end is the first stick which should not be reduced and multiplied by scalar value. So the interval
    // [start, end) is the range of sticks that should be reduced and multiplied by scalar value.
    uint16_t start;
    uint16_t value;
    uint16_t end;
};

// This function generates a vector of elements of type ScalarInfo. It is called once per core and generates the
// adequate scalars for each output element that core should produce. It should be called only for avg pool operation
// and only if the divisor_override is NOT set and the idea behind it is to generate config tesnor in cases where one
// scalar per core is not sufficient to create correct result. Those scenarios are ceil_mode == true and (ceil_pad_h > 0
// || ceil_pad_w > 0) or count_include_pad == false || (pad_h > 0 || pad_w > 0). Both of these scenarios can be
// irrelevant if the divisor_override is set, in which case we don't calculate the divisor since it is already passed as
// an argument. It only adds scalars that are different than the scalar preeceding it not to have duplicates of data,
// this is why we use start and end indices to know how many sequential output elements should be multiplied by the same
// scalar value.
std::vector<ScalarInfo> get_bf16_avg_pool_config_scalars(
    const AvgPoolConfig& config, uint32_t output_stick_x, uint32_t output_stick_y) {
    TT_FATAL(
        !is_pool_op_one_scalar_per_core(
            Pool2DType::AVG_POOL2D,
            config.ceil_mode,
            config.ceil_h,
            config.ceil_w,
            config.count_include_pad,
            config.pad_t + config.pad_b,
            config.pad_l + config.pad_r,
            config.divisor_override),
        "Avg pool scalars config should be calulated only for ceil_mode == true and "
        "(ceil_pad_h > 0 || ceil_pad_w > 0) or count_include_pad == false and (pad_h > 0 || pad_w > 0)");

    std::vector<ScalarInfo> scalars;
    bool first_scalar = true;
    uint32_t last_pool_area = 0;

    for (uint32_t i = 0; i < config.max_out_nhw_per_core; i++) {
        // Compute starting and ending indices of the pooling window
        int h_start = (output_stick_x * config.stride_h) - config.pad_t;
        int w_start = (output_stick_y * config.stride_w) - config.pad_l;
        // omit any ceiling mode related padding from end point calculations as these are not used in the
        // calculation of the pool area even when count_include_pad is on
        int h_end = std::min(h_start + static_cast<int>(config.kernel_h), static_cast<int>(config.in_h + config.pad_b));
        int w_end = std::min(w_start + static_cast<int>(config.kernel_w), static_cast<int>(config.in_w + config.pad_r));

        int pool_area = 0;
        if (config.count_include_pad) {
            pool_area = (h_end - h_start) * (w_end - w_start);
        } else {
            int valid_h_start = (h_start > 0) ? h_start : 0;
            int valid_w_start = (w_start > 0) ? w_start : 0;
            int valid_h_end = std::min(h_end, static_cast<int>(config.in_h));
            int valid_w_end = std::min(w_end, static_cast<int>(config.in_w));

            int effective_h = valid_h_end - valid_h_start;
            int effective_w = valid_w_end - valid_w_start;
            pool_area = (effective_h > 0 && effective_w > 0) ? effective_h * effective_w : 0;
        }
        float value = pool_area > 0 ? 1.f / static_cast<float>(pool_area) : 0.f;

        // Add new scalar if padding config changes
        if (first_scalar || static_cast<uint32_t>(pool_area) != last_pool_area) {
            if (!scalars.empty()) {
                scalars.back().end = i;
            }
            // TODO: #27672: Truncation should be removed once we figure a root cause of regression without it
            scalars.push_back({i, std::bit_cast<uint16_t>(bfloat16::truncate(value)), i});
            first_scalar = false;
        }
        last_pool_area = static_cast<uint32_t>(pool_area);

        // Advance output element coordinates
        output_stick_y = (output_stick_y + 1) % config.out_w;
        if (output_stick_y == 0) {
            output_stick_x = (output_stick_x + 1) % config.out_h;
        }
    }

    scalars.back().end = config.max_out_nhw_per_core;
    return scalars;
}

static void push_back_scalar_info_or_zero(
    std::vector<uint16_t>& config_vector,
    const std::vector<ScalarInfo>& scalars,
    uint32_t max_scalars_cnt,
    uint32_t repeats) {
    for (uint32_t r = 0; r < repeats; ++r) {
        for (uint32_t j = 0; j < max_scalars_cnt; ++j) {
            if (j < scalars.size()) {
                config_vector.insert(config_vector.end(), {scalars[j].start, scalars[j].value, scalars[j].end});
            } else {
                config_vector.insert(config_vector.end(), {0, 0, 0});
            }
        }
    }
}

// This function creates a scalar config tensor for the AvgPool2D operation. It is entirely made of
// a vector of ScalarInfo structs, which are used to configure the pooling operation. The config tensor
// is filled with max_out_nhw_per_core number of ScalarInfos for each core and then sharded across the cores.
// Since we don't usually have that many different scalars, we fill the rest of the config tensor with 0s.
static Tensor create_scalar_config_tensor(
    const AvgPoolConfig& config,
    TensorMemoryLayout in_memory_layout,
    uint32_t n_dim,
    uint32_t num_shards_c,
    uint32_t num_cores,
    bool config_tensor_in_dram) {
    std::vector<uint16_t> config_vector;

    size_t max_scalars_cnt = 0;
    std::vector<std::vector<ScalarInfo>> scalars_per_core = {};
    uint32_t num_iterations = 0;

    switch (in_memory_layout) {
        case TensorMemoryLayout::HEIGHT_SHARDED: num_iterations = num_cores; break;
        case TensorMemoryLayout::BLOCK_SHARDED: num_iterations = num_cores / num_shards_c; break;
        case TensorMemoryLayout::WIDTH_SHARDED: num_iterations = 1; break;
        default: break;
    }

    {
        uint32_t nhw_linear = 0;
        uint32_t output_stick_n = 0;
        uint32_t output_stick_h = 0;
        uint32_t output_stick_w = 0;
        for (uint32_t i = 0; i < num_iterations; ++i) {
            scalars_per_core.emplace_back(get_bf16_avg_pool_config_scalars(config, output_stick_h, output_stick_w));
            max_scalars_cnt = std::max(max_scalars_cnt, scalars_per_core.back().size());

            // Width sharded layout requires only one iteration, so we can break here
            if (in_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
                in_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                nhw_linear += config.max_out_nhw_per_core;
                output_stick_w = nhw_linear % config.out_w;
                output_stick_h = (nhw_linear / config.out_w) % config.out_h;
                output_stick_n = nhw_linear / (config.out_w * config.out_h);

                if (output_stick_n == n_dim) {
                    nhw_linear -= output_stick_n * config.out_w * config.out_h;
                    output_stick_n = 0;
                    output_stick_h = 0;
                    output_stick_w = 0;
                }
            }
        }
    }

    constexpr uint32_t entry_size = 3;
    const uint32_t entries_per_core = entry_size * max_scalars_cnt;

    switch (in_memory_layout) {
        case TensorMemoryLayout::HEIGHT_SHARDED:
        // With height sharded layout each scalar is unique and needs to be calculated
        case TensorMemoryLayout::BLOCK_SHARDED: {
            // With block sharded layout scalars across sequential channels shard repeat, so we use repeats variable not
            // to recalculate scalars that we can reuse
            for (const std::vector<ScalarInfo>& scalars : scalars_per_core) {
                uint32_t repeats = config_tensor_in_dram ? 1 : num_shards_c;
                push_back_scalar_info_or_zero(config_vector, scalars, max_scalars_cnt, repeats);
            }
            break;
        }
        case TensorMemoryLayout::WIDTH_SHARDED: {
            // With width sharded layout scalars should be calulated only once, so we push them back num_shards_c times
            // but have only one array of scalars
            uint32_t repeats = config_tensor_in_dram ? 1 : num_shards_c;
            push_back_scalar_info_or_zero(config_vector, scalars_per_core[0], max_scalars_cnt, repeats);
            break;
        }

        default: break;
    }

    TT_FATAL(
        config_vector.size() % entries_per_core == 0,
        "Config vector size {} should be a multiple of {}",
        config_vector.size(),
        entries_per_core);

    ttnn::Shape config_shape = ttnn::Shape({tt::div_up(config_vector.size(), entries_per_core), entries_per_core});
    tt::tt_metal::HostBuffer buffer(std::move(config_vector));
    return Tensor(std::move(buffer), config_shape, DataType::UINT16, Layout::ROW_MAJOR);
}

std::vector<uint16_t> generate_core_starting_indices(
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<sliding_window::ShardBoundary>& shard_boundaries,
    const tt::tt_metal::TensorMemoryLayout shard_scheme,
    const uint32_t num_cores_x,
    const uint32_t ncores) {
    std::vector<uint16_t> starting_indices;
    uint32_t repeat_factor = 0;
    switch (shard_scheme) {
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: repeat_factor = 1; break;
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: repeat_factor = ncores; break;
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: repeat_factor = num_cores_x; break;
        default: TT_FATAL(false, "Unsupported shard scheme");
    };
    for (const auto& item : shard_boundaries) {
        const auto& [output_shard_start, output_shard_end] = item.output_range;
        const auto& [input_shard_start, input_shard_end] = item.input_range;
        if (output_shard_start >= op_trace_metadata.size()) {
            // this core has no output
            starting_indices.push_back(0);
            continue;
        }
        TT_ASSERT(input_shard_start == op_trace_metadata[output_shard_start]);
        for (uint32_t r = 0; r < repeat_factor; r++) {
            starting_indices.push_back(op_trace_metadata[output_shard_start]);
        }
    }

    return starting_indices;
}

Pool2D::MultiCore::cached_program_t pool2d_multi_core_sharded_with_halo_v2_impl_new(
    Program& program,
    const Tensor& input,
    const Tensor& reader_indices,
    uint32_t reader_indices_size,
    std::vector<Tensor>& outputs,
    Pool2DType pool_type,
    uint32_t in_n,
    uint32_t in_c,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t out_c,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_t,
    uint32_t pad_b,
    uint32_t pad_l,
    uint32_t pad_r,
    uint32_t ceil_pad_h,
    uint32_t ceil_pad_w,
    bool ceil_mode,
    bool return_indices,
    std::vector<uint16_t> core_starting_indices,
    bool count_include_pad,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t num_shards_c,
    const MemoryConfig& /*out_mem_config*/,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<int32_t> divisor_override,
    uint32_t memory_used,
    const Layout& output_layout,
    bool config_tensor_in_dram) {
    distributed::MeshDevice* device = input.device();

    const tt::tt_metal::DeviceStorage& reader_indices_storage = reader_indices.device_storage();
    const bool is_block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    // distributing out_hw across the grid
    const auto all_cores = input.shard_spec().value().grid;
    const uint32_t ncores = all_cores.num_cores();
    const uint32_t num_cores_x = input.memory_config().shard_spec()->grid.bounding_box().grid_size().x;
    const uint32_t rectangular_x = is_block_sharded ? all_cores.ranges()[0].end_coord.x + 1 : num_cores_x;
    const uint32_t max_out_nhw_per_core = outputs[0].shard_spec()->shape[0];

    const uint32_t bf16_scalar = get_bf16_pool_scalar(pool_type, kernel_h, kernel_w, divisor_override);
    const uint32_t bf16_init_value = get_bf16_pool_init_value(pool_type);
    FactoryParameters params = get_factory_parameters(
        num_shards_c,
        input.dtype(),
        outputs[0].dtype(),
        kernel_h,
        kernel_w,
        in_c,
        pool_type,
        return_indices,
        output_layout);
    uint32_t eff_kernel_h = ((kernel_h - 1) * dilation_h) + 1;
    uint32_t eff_kernel_w = ((kernel_w - 1) * dilation_w) + 1;
    uint32_t pad_h = pad_t + pad_b;
    uint32_t pad_w = pad_l + pad_r;
    const uint32_t in_h_padded = in_h + pad_h + ceil_pad_h;
    const uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;
    const bool one_scalar_per_core = is_pool_op_one_scalar_per_core(
        pool_type, ceil_mode, ceil_pad_h, ceil_pad_w, count_include_pad, pad_h, pad_w, divisor_override);

    const auto& input_shape = input.padded_shape();
    const uint32_t shard_width = input.shard_spec()->shape[1];
    const uint32_t in_c_per_shard_ceil = in_c % shard_width != 0 && num_shards_c > 1
                                             ? (in_c - (in_c % shard_width)) / (num_shards_c - 1)
                                             : in_c / num_shards_c;
    const uint32_t in_nbytes_c = in_c_per_shard_ceil * params.nbytes;  // row of input (channels)
    const uint32_t shard_width_bytes = input_shape[3] / num_shards_c * params.nbytes;

    TT_FATAL(
        input_shape[3] % num_shards_c == 0,
        "Input channels {} should be divisible by number of shards {}",
        input_shape[3],
        num_shards_c);
    const uint32_t in_nbytes_leftover =
        params.is_wide_reduction &&
                (input_shape[3] / num_shards_c) % (params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH) != 0
            ? tt::round_up(
                  (input_shape[3] / num_shards_c) % (params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH),
                  tt::constants::TILE_WIDTH) *
                  params.nbytes
            : tt::round_up(input_shape[3] / num_shards_c, tt::constants::TILE_WIDTH) * params.nbytes;

    uint32_t next_cb_index = tt::CBIndex::c_0;
    const uint32_t in_scalar_cb_id_0 = next_cb_index++;
    const uint32_t in_scalar_cb_pagesize = tile_size(params.data_format);
    const uint32_t in_scalar_cb_npages = params.multi_buffering_factor;
    tt::tt_metal::create_cb(
        in_scalar_cb_id_0, program, all_cores, in_scalar_cb_pagesize, in_scalar_cb_npages, params.data_format);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_scalar_cb_id_0, in_scalar_cb_pagesize, in_scalar_cb_npages);

    uint32_t in_scalar_cb_id_1 = 32;
    if (params.is_avg_pool && params.split_reader && !one_scalar_per_core) {
        in_scalar_cb_id_1 = next_cb_index++;
        tt::tt_metal::create_cb(
            in_scalar_cb_id_1, program, all_cores, in_scalar_cb_pagesize, in_scalar_cb_npages, params.data_format);
        log_debug(
            tt::LogOp, "CB {} :: PS = {}, NP = {}", in_scalar_cb_id_1, in_scalar_cb_pagesize, in_scalar_cb_npages);
    }

    // CB storing just "clear value" (-inf for maxpool, 0 for avgpool)
    uint32_t clear_value_cb_id = next_cb_index++;
    tt::tt_metal::create_cb(
        clear_value_cb_id, program, all_cores, tile_size(params.data_format), 1, params.data_format);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", clear_value_cb_id, tile_size(params.data_format), 1);

    // incoming data is the input cb instead of raw l1/dram addr
    // this input shard has halo and padding inserted.
    const uint32_t raw_in_cb_npages = input.shard_spec().value().shape[0];
    const uint32_t raw_in_cb_pagesize = in_nbytes_c;
    const auto [raw_in_cb_id, raw_in_cb] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, raw_in_cb_pagesize, raw_in_cb_npages, params.data_format, input.buffer());

    log_debug(tt::LogOp, "Raw In CB {} :: PS = {}, NP = {}", raw_in_cb_id, raw_in_cb_pagesize, raw_in_cb_npages);

    // reader indices
    const uint32_t in_reader_indices_cb_id = next_cb_index++;
    const uint32_t in_reader_indices_cb_pagesize =
        tt::round_up(reader_indices_size, 4);  // pagesize needs to be multiple of 4
    constexpr uint32_t in_reader_indices_cb_npages = 1;

    const uint32_t max_reader_indices_size =
        (max_out_nhw_per_core * 3 * sizeof(uint16_t)) + 2;  // worst case of 3 indices per output element
    TT_FATAL(
        reader_indices_storage.get_buffer()->page_size() <= max_reader_indices_size,
        "Reader indices buffer page size {} exceeds max expected size {}",
        reader_indices_storage.get_buffer()->page_size(),
        max_reader_indices_size);

    tt::tt_metal::create_cb(
        in_reader_indices_cb_id,
        program,
        all_cores,
        config_tensor_in_dram ? max_reader_indices_size : in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages,
        tt::DataFormat::UInt16,
        config_tensor_in_dram ? nullptr : reader_indices_storage.get_buffer());

    log_debug(
        tt::LogOp,
        "In Reader Indices CB {} :: PS = {}, NP = {}",
        in_reader_indices_cb_id,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages);
    uint32_t in_cb_sz = 0;
    uint32_t in_nblocks_c = 1;
    if (return_indices) {
        // for return indices we use 1 whole tile per reduction to simplify logic
        in_cb_sz = params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        in_nblocks_c = std::ceil((float)params.in_ntiles_c / params.MAX_TILES_PER_REDUCTION);
    } else {
        if (params.is_wide_reduction) {
            in_cb_sz = params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * params.num_tilized_rows;
            in_nblocks_c = std::ceil((float)params.in_ntiles_c / params.MAX_TILES_PER_REDUCTION);
        } else {
            in_cb_sz = params.in_ntiles_c * tt::constants::TILE_WIDTH * params.num_tilized_rows;
        }
    }

    // reader output == input to tilize
    const uint32_t in_cb_id_0 = next_cb_index++;  // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_id_1 = 32;                     // input rows for "multiple (out_nelems)" output pixels
    const uint32_t in_cb_page_padded = tt::round_up(
        in_cb_sz,
        tt::constants::TILE_HW);  // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    const uint32_t in_cb_pagesize = params.nbytes * in_cb_page_padded;
    const uint32_t in_cb_npages = params.multi_buffering_factor;

    tt::tt_metal::create_cb(in_cb_id_0, program, all_cores, in_cb_pagesize, in_cb_npages, params.data_format);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_cb_id_0, in_cb_pagesize, in_cb_npages);

    if (params.split_reader) {
        in_cb_id_1 = next_cb_index++;
        tt::tt_metal::create_cb(in_cb_id_1, program, all_cores, in_cb_pagesize, in_cb_npages, params.data_format);
        log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_cb_id_1, in_cb_pagesize, in_cb_npages);
    }

    uint32_t in_idx_cb_id = 32;
    uint32_t pack_tmp_cb_id = 32;
    uint32_t pack_idx_tmp_cb_id = 32;
    uint32_t right_inc_cb_id = 32;
    uint32_t down_left_wrap_inc_cb_id = 32;
    uint32_t up_left_wrap_inc_cb_id = 32;
    uint16_t right_inc = 0;
    uint16_t down_left_wrap_inc = 0;
    uint16_t up_left_wrap_inc = 0;
    if (return_indices) {
        uint32_t tile_elems = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        in_idx_cb_id = next_cb_index++;
        tt::tt_metal::create_cb(in_idx_cb_id, program, all_cores, params.nbytes * tile_elems, 1, params.index_format);
        log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_idx_cb_id, params.nbytes * tile_elems, 1);
        pack_tmp_cb_id = next_cb_index++;
        tt::tt_metal::create_cb(pack_tmp_cb_id, program, all_cores, params.nbytes * tile_elems, 1, params.data_format);
        log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", pack_tmp_cb_id, params.nbytes * tile_elems, 1);
        pack_idx_tmp_cb_id = next_cb_index++;
        tt::tt_metal::create_cb(
            pack_idx_tmp_cb_id, program, all_cores, params.index_nbytes * tile_elems, 1, params.index_format);
        log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", pack_idx_tmp_cb_id, params.index_nbytes * tile_elems, 1);
        right_inc_cb_id = next_cb_index++;
        tt::tt_metal::create_cb(
            right_inc_cb_id, program, all_cores, params.index_nbytes * tile_elems, 1, params.index_format);
        log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", right_inc_cb_id, params.index_nbytes * tile_elems, 1);
        down_left_wrap_inc_cb_id = next_cb_index++;
        tt::tt_metal::create_cb(
            down_left_wrap_inc_cb_id, program, all_cores, params.index_nbytes * tile_elems, 1, params.index_format);
        log_debug(
            tt::LogOp, "CB {} :: PS = {}, NP = {}", down_left_wrap_inc_cb_id, params.index_nbytes * tile_elems, 1);
        up_left_wrap_inc_cb_id = next_cb_index++;
        tt::tt_metal::create_cb(
            up_left_wrap_inc_cb_id, program, all_cores, params.index_nbytes * tile_elems, 1, params.index_format);
        log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", up_left_wrap_inc_cb_id, params.index_nbytes * tile_elems, 1);

        // compute increments for index tile population
        right_inc = stride_w;
        down_left_wrap_inc = in_w * stride_h + (1 - out_w) * stride_w;
        up_left_wrap_inc =
            (1 - out_h) * stride_h * in_w + (1 - out_w) * stride_w;  // allow overflow for negative values
    }

    const bool is_output_tiled = output_layout == Layout::TILE;
    const bool is_output_block_format = is_block_float(outputs[0].dtype());
    const bool zero_pages = is_output_tiled && is_output_block_format;

    // Conditionally allocate temporary CB - only needed for TILED output
    uint32_t pre_tilize_cb_id = 32;  // default invalid CB ID

    if (is_output_tiled) {
        pre_tilize_cb_id = next_cb_index++;
        const uint32_t pre_tilize_cb_pagesize = tt::constants::TILE_WIDTH * params.nbytes;
        const uint32_t pre_tilize_cb_npages = tt::constants::TILE_HEIGHT * params.in_ntiles_c;
        tt::tt_metal::create_cb(
            pre_tilize_cb_id, program, all_cores, pre_tilize_cb_pagesize, pre_tilize_cb_npages, params.data_format);
        log_debug(
            tt::LogOp, "CB {} :: PS = {}, NP = {}", pre_tilize_cb_id, pre_tilize_cb_pagesize, pre_tilize_cb_npages);
    }

    uint32_t out_cb_pagesize;
    uint32_t out_cb_npages;

    if (is_output_tiled) {
        out_cb_pagesize = tt::tile_size(params.output_data_format);
        out_cb_npages =
            outputs[0].shard_spec().value().shape[0] * outputs[0].shard_spec().value().shape[1] / tt::constants::TILE_HW;
    } else {
        out_cb_pagesize =
            std::min(static_cast<uint32_t>(tt::constants::FACE_WIDTH), outputs[0].shard_spec().value().shape[1]) *
            params.nbytes;  // there is just one row of channels after each reduction (or 1
                            // block of c if its greater than 8 tiles)
        out_cb_npages = outputs[0].shard_spec().value().shape[0] * params.out_ntiles_c;
    }

    const auto [out_cb_id, out_cb] = tt::tt_metal::create_cb(
        next_cb_index++,
        program,
        all_cores,
        out_cb_pagesize,
        out_cb_npages,
        params.output_data_format,
        outputs[0].buffer());

    uint32_t out_idx_cb_id = 32;
    tt::tt_metal::CBHandle out_idx_cb = 0;
    if (return_indices) {
        TT_FATAL(
            outputs.size() == 2,
            "When return_indices is true, there should be two outputs, but got {}",
            outputs.size());
        uint32_t out_idx_cb_npages = out_cb_npages;
        uint32_t out_idx_cb_pagesize =
            std::min(static_cast<uint32_t>(tt::constants::FACE_WIDTH), outputs[0].shard_spec().value().shape[1]) *
            params.index_nbytes;
        std::tie(out_idx_cb_id, out_idx_cb) = tt::tt_metal::create_cb(
            next_cb_index++,
            program,
            all_cores,
            out_idx_cb_pagesize,
            out_idx_cb_npages,
            params.index_format,
            outputs[1].buffer());
    }
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", out_cb_id, out_cb_pagesize, out_cb_npages);

    for (const auto& output : outputs) {
        TT_FATAL(
            output.memory_config().is_sharded(),
            "Output memory config needs to be sharded, but got {}",
            output.memory_config());
    }

    /**
     * Reader Kernel: input rows -> input cb
     */
    tt::tt_metal::CBHandle config_cb;
    tt::tt_metal::DeviceStorage scalar_config_storage;
    uint32_t config_cb_id = 32;
    Tensor config_tensor;
    if (!one_scalar_per_core) {
        // create config tensor
        AvgPoolConfig avg_pool_config = {
            .kernel_h = kernel_h,
            .kernel_w = kernel_w,
            .in_h = in_h,
            .in_w = in_w,
            .out_h = out_h,
            .out_w = out_w,
            .stride_h = stride_h,
            .stride_w = stride_w,
            .ceil_mode = ceil_mode,
            .ceil_h = ceil_pad_h,
            .ceil_w = ceil_pad_w,
            .count_include_pad = count_include_pad,
            .pad_t = pad_t,
            .pad_b = pad_b,
            .pad_l = pad_l,
            .pad_r = pad_r,
            .max_out_nhw_per_core = max_out_nhw_per_core,
            .divisor_override = divisor_override};
        config_tensor = create_scalar_config_tensor(
            avg_pool_config, input.memory_config().memory_layout(), in_n, num_shards_c, ncores, config_tensor_in_dram);

        const std::array<uint32_t, 2> shard_shape =
            std::array<uint32_t, 2>({1, static_cast<uint32_t>(config_tensor.logical_shape()[-1])});
        const tt::tt_metal::ShardOrientation config_tensor_shard_orientation = input.shard_spec().value().orientation;
        const tt::tt_metal::ShardSpec config_shard_spec(
            input.shard_spec().value().grid, shard_shape, config_tensor_shard_orientation);
        const MemoryConfig l1_small_memory_config{
            TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec};

        config_tensor =
            config_tensor.to_device(device, config_tensor_in_dram ? DRAM_MEMORY_CONFIG : l1_small_memory_config);

        constexpr tt::DataFormat config_df = tt::DataFormat::RawUInt32;
        scalar_config_storage = config_tensor.device_storage();
        tt::tt_metal::Buffer* config_buffer = scalar_config_storage.get_buffer();
        const uint32_t config_buffer_page_size = config_buffer->page_size();
        uint32_t max_config_tensor_size =
            max_out_nhw_per_core * 3 * sizeof(uint16_t);  // worst case of 3 entries per output element

        TT_FATAL(
            config_buffer->page_size() <= max_config_tensor_size,
            "Config tensor buffer page size {} exceeds max expected size {}",
            config_buffer->page_size(),
            max_config_tensor_size);

        std::tie(config_cb_id, config_cb) = tt::tt_metal::create_cb(
            next_cb_index++,
            program,
            all_cores,
            config_tensor_in_dram ? max_config_tensor_size : config_buffer_page_size,
            1,
            config_df,
            config_tensor_in_dram ? nullptr : config_buffer);
    }
    std::vector<uint32_t> reader0_ct_args = {
        max_out_nhw_per_core,                                                                // 0
        kernel_h,                                                                            // 1
        kernel_w,                                                                            // 2
        pad_w,                                                                               // 3
        in_nbytes_leftover,                                                                  // 4
        in_w,                                                                                // 5
        in_c_per_shard_ceil,                                                                 // 6
        params.split_reader,                                                                 // enable split reader //7
        0,                                                                                   // split reader id //8
        bf16_scalar,                                                                         // 9
        bf16_init_value,                                                                     // 10
        in_nblocks_c,                                                                        // 11
        in_cb_sz,                                                                            // 12
        params.max_rows_for_reduction,                                                       // 13
        ceil_pad_w,                                                                          // 14
        in_cb_id_0,                                                                          // 15
        in_cb_id_1,                                                                          // 16
        raw_in_cb_id,                                                                        // 17
        in_reader_indices_cb_id,                                                             // 18
        in_scalar_cb_id_0,                                                                   // 19
        in_scalar_cb_id_1,                                                                   // 20
        in_idx_cb_id,                                                                        // 21
        pack_tmp_cb_id,                                                                      // 22
        pack_idx_tmp_cb_id,                                                                  // 23
        right_inc_cb_id,                                                                     // 24
        down_left_wrap_inc_cb_id,                                                            // 25
        up_left_wrap_inc_cb_id,                                                              // 26
        clear_value_cb_id,                                                                   // 27
        (uint32_t)pool_type,                                                                 // 28
        one_scalar_per_core,                                                                 // 29
        config_cb_id,                                                                        // 30
        in_nbytes_c,                                                                         // 31
        shard_width_bytes,                                                                   // 32
        params.multi_buffering_factor,                                                       // 33
        stride_w,                                                                            // 34
        dilation_h,                                                                          // 35
        dilation_w,                                                                          // 36
        (uint32_t)return_indices,                                                            // 37
        pad_t,                                                                               // 38
        pad_l,                                                                               // 39
        right_inc,                                                                           // 40
        down_left_wrap_inc,                                                                  // 41
        up_left_wrap_inc,                                                                    // 42
        (uint32_t)zero_pages,                                                                // 43
        out_cb_id,                                                                           // 44
        out_idx_cb_id,                                                                       // 45
        config_tensor_in_dram,                                                               // 46
        one_scalar_per_core ? 0 : config_tensor.device_storage().get_buffer()->address(),    // 47
        one_scalar_per_core ? 0 : config_tensor.device_storage().get_buffer()->page_size(),  // 48
        reader_indices_storage.get_buffer()->address(),                                      // 49
        reader_indices_storage.get_buffer()->page_size()                                     // 50
    };

    tt::tt_metal::TensorAccessorArgs(reader_indices_storage.get_buffer()).append_to(reader0_ct_args);
    if (!one_scalar_per_core) {
        tt::tt_metal::TensorAccessorArgs(config_tensor.device_storage().get_buffer()).append_to(reader0_ct_args);
    }
    std::vector<uint32_t> reader1_ct_args = reader0_ct_args;
    reader1_ct_args[8] = 1;  // split reader id for reader1

    std::string reader_kernel_fname =
        "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/"
        "reader_pool_2d.cpp";

    auto reader0_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
        .compile_args = reader0_ct_args};
    auto reader0_kernel = CreateKernel(program, reader_kernel_fname, all_cores, reader0_config);

    auto reader1_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::RISCV_1_default,
        .compile_args = reader1_ct_args};
    auto reader1_kernel =
        params.split_reader ? CreateKernel(program, reader_kernel_fname, all_cores, reader1_config) : 0;

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block ->
     * output cb
     */

    std::vector<uint32_t> compute_ct_args = {
        params.in_ntiles_c,             // 0
        kernel_h * kernel_w,            // 1
        params.split_reader,            // 2
        0,                              // 3 - max_out_nhw_per_core, used for grid sample but not for pool
        in_c_per_shard_ceil,            // 4
        in_nblocks_c,                   // 5
        params.max_rows_for_reduction,  // 6
        in_cb_id_0,                     // 7
        in_cb_id_1,                     // 8
        in_scalar_cb_id_0,              // 9
        in_scalar_cb_id_1,              // 10
        in_idx_cb_id,                   // 11
        pack_tmp_cb_id,                 // 12
        pack_idx_tmp_cb_id,             // 13
        right_inc_cb_id,                // 14
        down_left_wrap_inc_cb_id,       // 15
        up_left_wrap_inc_cb_id,         // 16
        out_cb_id,                      // 17
        out_idx_cb_id,                  // 18
        one_scalar_per_core,            // 19
        pre_tilize_cb_id,               // 20
        is_output_tiled,                // 21
        is_output_block_format,         // 22
        (uint32_t)return_indices,       // 23
        stride_h,                       // 24
        stride_w,                       // 25
        in_h_padded,                    // 26
        in_w_padded,                    // 27
        eff_kernel_h,                   // 28
        eff_kernel_w,                   // 29
        pad_l};                         // 30

    // Get device arch for compute kernel config initialization
    auto device_arch = input.device()->arch();

    // Initialize device compute kernel config with user-provided config or defaults
    auto device_compute_kernel_config = init_device_compute_kernel_config(
        device_arch,
        compute_kernel_config,
        MathFidelity::HiFi4,
        false,                                         // math_approx_mode
        params.is_avg_pool && params.is_large_kernel,  // fp32_dest_acc_en
        false,                                         // packer_l1_acc
        false                                          // dst_full_sync_en
    );

    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = get_math_fidelity(device_compute_kernel_config),
        .fp32_dest_acc_en = get_fp32_dest_acc_en(device_compute_kernel_config),
        .math_approx_mode = false,
        .compile_args = compute_ct_args,
        .defines = get_defines(pool_type)};

    std::string compute_kernel_fname =
        "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp";

    auto compute_kernel = CreateKernel(program, compute_kernel_fname, all_cores, compute_config);

    // set the starting indices for each core as runtime args
    uint32_t total_out_nhw = in_n * out_h * out_w;
    for (uint32_t core_i = 0; core_i < ncores; core_i++) {
        const uint32_t core_x_i = core_i % rectangular_x;
        const uint32_t core_y_i = core_i / rectangular_x;
        const CoreRange core(CoreCoord(core_x_i, core_y_i), CoreCoord(core_x_i, core_y_i));

        uint32_t total_out_nhw_processed;
        uint32_t core_nhw_index = 0;
        if (is_block_sharded) {
            total_out_nhw_processed = core_y_i * max_out_nhw_per_core;
            core_nhw_index = core_y_i;
        } else if (is_width_sharded) {
            total_out_nhw_processed = 0;
            core_nhw_index = 0;
        } else {
            total_out_nhw_processed = core_i * max_out_nhw_per_core;
            core_nhw_index = core_i;
        }
        uint32_t remaining_out_nhw =
            total_out_nhw_processed < total_out_nhw ? total_out_nhw - total_out_nhw_processed : 0;
        uint32_t out_nhw_this_core = std::min(max_out_nhw_per_core, remaining_out_nhw);
        std::vector<uint32_t> args = {out_nhw_this_core, core_nhw_index};

        if (return_indices) {
            TT_FATAL(core_starting_indices.size() == ncores, "core starting indices size should match number of cores");
            const uint32_t start_index = core_starting_indices[core_i];
            const uint32_t start_mod_batch = start_index % (in_w_padded * in_h_padded);
            const uint32_t start_row = start_mod_batch / in_w_padded;
            const uint32_t start_col = start_mod_batch % in_w_padded;

            args.push_back((uint32_t)(start_row));
            args.push_back((uint32_t)(start_col));
        }
        SetRuntimeArgs(program, reader0_kernel, core, args);
        SetRuntimeArgs(program, compute_kernel, core, args);
    }

    auto temporary_size = calculate_total_cb_size(program);

    uint32_t post_allocate_size =
        input.device()->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;
    uint32_t l1_usage = calculate_L1_usage(
        input.dtype(),
        in_c,
        pad_h,
        pad_w,
        ceil_pad_h,
        ceil_pad_w,
        ceil_mode,
        return_indices,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        input.memory_config(),
        outputs[0].memory_config(),
        pool_type,
        count_include_pad,
        divisor_override,
        output_layout,
        outputs[0].dtype(),
        config_tensor_in_dram);

    uint32_t output_cb_size = post_allocate_size - memory_used;

    // For now assume that if post_op_l1_allocation_size == 0 op is being run
    // in graph capture NO_DISPATCH mode.
    bool is_graph_capture_no_dispatch_mode = post_allocate_size == 0;
    TT_FATAL(
        temporary_size + output_cb_size == l1_usage || is_graph_capture_no_dispatch_mode,
        "Calculated CB size {} + {} = {} does not match with the actual CB size {}  ",
        temporary_size,
        output_cb_size,
        temporary_size + output_cb_size,
        l1_usage);

    {  // debug
        log_debug(tt::LogOp, "raw_in_cb :: PS = {}, NP = {}", raw_in_cb_pagesize, raw_in_cb_npages);
        log_debug(tt::LogOp, "in_cb :: PS = {}, NP = {}", in_cb_pagesize, in_cb_npages);
        log_debug(
            tt::LogOp,
            "in_reader_indices_cb :: PS = {}, NP = {}",
            in_reader_indices_cb_pagesize,
            in_reader_indices_cb_npages);
        log_debug(tt::LogOp, "in_scalar_cb :: PS = {}, NP = {}", in_scalar_cb_pagesize, in_scalar_cb_npages);
        log_debug(tt::LogOp, "out_cb :: PS = {}, NP = {}", out_cb_pagesize, out_cb_npages);
        log_debug(tt::LogOp, "in_reader_indices_addr: {}", reader_indices_storage.get_buffer()->address());
        if (scalar_config_storage.is_allocated()) {
            log_debug(tt::LogOp, "scalar_config_addr: {}", scalar_config_storage.get_buffer()->address());
        } else {
            log_debug(tt::LogOp, "scalar_config_addr: not set");
        }
        log_debug(tt::LogOp, "kernel_h: {}", kernel_h);
        log_debug(tt::LogOp, "kernel_w: {}", kernel_w);
        log_debug(tt::LogOp, "stride_h: {}", stride_h);
        log_debug(tt::LogOp, "stride_w: {}", stride_w);
        log_debug(tt::LogOp, "pad_h: {}", pad_h);
        log_debug(tt::LogOp, "pad_w: {}", pad_w);
        log_debug(tt::LogOp, "out_h: {}", out_h);
        log_debug(tt::LogOp, "out_w: {}", out_w);
        log_debug(tt::LogOp, "out_c: {}", out_c);
        log_debug(tt::LogOp, "in_h: {}", in_h);
        log_debug(tt::LogOp, "in_w: {}", in_w);
        log_debug(tt::LogOp, "in_c: {}", in_c);
        log_debug(tt::LogOp, "in_ntiles_c: {}", params.in_ntiles_c);
        log_debug(tt::LogOp, "out_ntiles_c: {}", params.out_ntiles_c);
        log_debug(tt::LogOp, "in_nblocks_c: {}", in_nblocks_c);
        log_debug(tt::LogOp, "in_nbytes_c: {}", in_nbytes_c);
        log_debug(tt::LogOp, "ncores: {}", ncores);
        log_debug(tt::LogOp, "max_out_nhw_per_core: {}", max_out_nhw_per_core);
        log_debug(tt::LogOp, "split_reader: {}", params.split_reader);
        log_debug(tt::LogOp, "multi_buffering_factor: {}", params.multi_buffering_factor);
        log_debug(tt::LogOp, "is_wide_reduction: {}", params.is_wide_reduction);
        log_debug(tt::LogOp, "is_in_sharded: {}", input.memory_config().is_sharded());
        log_debug(tt::LogOp, "is_out_sharded: {}", outputs[0].memory_config().is_sharded());
    }

    // Capture reader_indices_storage to cache this with the program
    return {
        std::move(program),
        {.reader0_kernel = reader0_kernel,
         .reader1_kernel = reader1_kernel,
         .compute_kernel = compute_kernel,
         .raw_in_cb = raw_in_cb,
         .out_cb = out_cb,
         .out_idx_cb = out_idx_cb,
         .in_scalar_cb_0 = in_scalar_cb_id_0,
         .in_scalar_cb_1 = in_scalar_cb_id_1,
         .clear_value_cb = clear_value_cb_id,
         .in_reader_indices_cb = in_reader_indices_cb_id,
         .in_cb_0 = in_cb_id_0,
         .in_cb_1 = in_cb_id_1,
         .pre_tilize_cb = pre_tilize_cb_id,
         .config_cb = config_cb_id,
         .in_idx_cb = in_idx_cb_id,
         .pack_tmp_cb = pack_tmp_cb_id,
         .pack_idx_tmp_cb = pack_idx_tmp_cb_id,
         .right_inc_cb = right_inc_cb_id,
         .down_left_wrap_inc_cb = down_left_wrap_inc_cb_id,
         .up_left_wrap_inc_cb = up_left_wrap_inc_cb_id,
         .ncores = ncores,
         .reader_indices_storage = reader_indices_storage,
         .scalar_config_storage = scalar_config_storage}};
}

Pool2D::MultiCore::cached_program_t Pool2D::MultiCore::create(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    const auto& input = tensor_args.input_tensor_;
    const auto& sliding_window_config = op_attr.sliding_window_config_;
    const auto& pool_type = op_attr.pool_type_;
    const auto& out_mem_config = op_attr.memory_config_;
    const auto& compute_kernel_config = op_attr.compute_kernel_config_;
    const auto& output_layout = op_attr.output_layout_;
    bool count_include_pad = op_attr.count_include_pad_;
    std::optional<int32_t> divisor_override = op_attr.divisor_override_;
    bool return_indices = op_attr.return_indices_;

    tt::tt_metal::Program program{};

    auto parallel_config = sliding_window::ParallelConfig{
        .grid = input.shard_spec().value().grid,
        .shard_scheme = input.memory_config().memory_layout(),
        .shard_orientation = input.shard_spec().value().orientation,
    };

    auto output_shape = sliding_window_config.get_output_shape();
    uint32_t out_h = output_shape[1];
    uint32_t out_w = output_shape[2];
    uint32_t out_c = output_shape[3];

    bool is_block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    auto in_n = sliding_window_config.batch_size;
    auto in_c = sliding_window_config.channels;
    auto in_h = sliding_window_config.input_hw.first;
    auto in_w = sliding_window_config.input_hw.second;
    auto kernel_h = sliding_window_config.window_hw.first;
    auto kernel_w = sliding_window_config.window_hw.second;
    auto stride_h = sliding_window_config.stride_hw.first;
    auto stride_w = sliding_window_config.stride_hw.second;
    auto pad_t = sliding_window_config.get_pad_top();
    auto pad_b = sliding_window_config.get_pad_bottom();
    auto pad_l = sliding_window_config.get_pad_left();
    auto pad_r = sliding_window_config.get_pad_right();
    auto ceil_pad_h = sliding_window_config.get_ceil_pad_h();
    auto ceil_pad_w = sliding_window_config.get_ceil_pad_w();
    auto ceil_mode = sliding_window_config.ceil_mode;
    auto dilation_h = sliding_window_config.dilation_hw.first;
    auto dilation_w = sliding_window_config.dilation_hw.second;
    auto num_shards_c = sliding_window_config.num_cores_c;

    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);
    std::vector<std::vector<uint16_t>> top_left_indices =
        sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, stride_w);
    std::vector<uint16_t> core_starting_indices;
    if (return_indices) {
        const uint32_t num_cores_x = input.memory_config().shard_spec()->grid.bounding_box().grid_size().x;
        const uint32_t ncores = input.shard_spec().value().grid.num_cores();
        const TensorMemoryLayout shard_scheme = input.memory_config().memory_layout();
        core_starting_indices =
            generate_core_starting_indices(op_trace_metadata, shard_boundaries, shard_scheme, num_cores_x, ncores);
    }

    Tensor reader_indices = sliding_window::construct_on_host_config_tensor(
        top_left_indices, parallel_config, op_attr.config_tensor_in_dram);
    Tensor reader_indices_on_device = sliding_window::move_config_tensor_to_device(
        reader_indices, parallel_config, is_block_sharded, input.device(), op_attr.config_tensor_in_dram);

    return pool2d_multi_core_sharded_with_halo_v2_impl_new(
        program,
        tensor_args.input_tensor_,
        reader_indices_on_device,
        top_left_indices[0].size(),
        output_tensors,
        pool_type,
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        out_c,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_t,
        pad_b,
        pad_l,
        pad_r,
        ceil_pad_h,
        ceil_pad_w,
        ceil_mode,
        return_indices,
        core_starting_indices,
        count_include_pad,
        dilation_h,
        dilation_w,
        num_shards_c,
        out_mem_config,
        compute_kernel_config,
        divisor_override,
        op_attr.memory_used,
        output_layout,
        op_attr.config_tensor_in_dram);
}

void Pool2D::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    auto& program = cached_program.program;
    auto& raw_in_cb = cached_program.shared_variables.raw_in_cb;
    auto& out_cb = cached_program.shared_variables.out_cb;

    const auto& input_tensor = tensor_args.input_tensor_;
    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensors[0].buffer();

    bool input_sharded = input_tensor.is_sharded();
    if (input_sharded) {
        UpdateDynamicCircularBufferAddress(program, raw_in_cb, *src_buffer);
    }
    bool out_sharded = output_tensors[0].is_sharded();
    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    }

    if (operation_attributes.return_indices_) {
        auto& out_idx_cb = cached_program.shared_variables.out_idx_cb;
        auto* dst_idx_buffer = output_tensors.size() > 1 ? output_tensors[1].buffer() : nullptr;
        if (out_sharded && dst_idx_buffer) {
            UpdateDynamicCircularBufferAddress(program, out_idx_cb, *dst_idx_buffer);
        }
    }
}

}  // namespace ttnn::operations::pool
