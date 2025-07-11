// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "pool_op.hpp"
#include "tt-metalium/circular_buffer.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/buffer.hpp"
#include <cstdint>
#include <optional>
#include <vector>
#include "ttnn/tensor/storage.hpp"
#include <tt-metalium/hal.hpp>
#include <algorithm>

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
            config.pad_h,
            config.pad_w,
            config.divisor_override),
        "Avg pool scalars config should be calulated only for ceil_mode == true and "
        "(ceil_pad_h > 0 || ceil_pad_w > 0) or count_include_pad == false and (pad_h > 0 || pad_w > 0)");

    std::vector<ScalarInfo> scalars;
    float value;
    bool first_scalar = true;
    uint32_t last_pool_area = 0;

    for (uint32_t i = 0; i < config.out_nhw_per_core; i++) {
        // Compute starting and ending indices of the pooling window
        int h_start = output_stick_x * config.stride_h - config.pad_h;
        int w_start = output_stick_y * config.stride_w - config.pad_w;
        int h_end = std::min(
            h_start + static_cast<int>(config.kernel_h), static_cast<int>(config.in_h + config.pad_h + config.ceil_h));
        int w_end = std::min(
            w_start + static_cast<int>(config.kernel_w), static_cast<int>(config.in_w + config.pad_w + config.ceil_w));

        int pool_area = 0;
        if (config.count_include_pad) {
            // Initial pool area
            pool_area = (h_end - h_start) * (w_end - w_start);

            // Calculate ceil induced padding overflow beyond input dimensions
            int pad_h_over = std::max(h_end - static_cast<int>(config.in_h) - static_cast<int>(config.pad_h), 0);
            int pad_w_over = std::max(w_end - static_cast<int>(config.in_w) - static_cast<int>(config.pad_w), 0);

            // Adjust pool area to exclude padded overflow
            pool_area -= pad_h_over * config.kernel_w;
            pool_area -= pad_w_over * config.kernel_h;

            // Re-add intersection if both directions overflowed
            if (pad_h_over > 0 && pad_w_over > 0) {
                pool_area += pad_h_over * pad_w_over;
            }
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
            scalars.push_back({i, bfloat16(value).to_packed(), i});
            first_scalar = false;
        }
        last_pool_area = static_cast<uint32_t>(pool_area);

        // Advance output element coordinates
        output_stick_y = (output_stick_y + 1) % config.out_w;
        if (output_stick_y == 0) {
            output_stick_x = (output_stick_x + 1) % config.out_h;
        }
    }

    scalars.back().end = config.out_nhw_per_core;
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
// is filled with out_nhw_per_core number of ScalarInfos for each core and then sharded across the cores.
// Since we don't usually have that many different scalars, we fill the rest of the config tensor with 0s.
static Tensor create_scalar_config_tensor(
    const AvgPoolConfig& config,
    TensorMemoryLayout in_memory_layout,
    uint32_t n_dim,
    uint32_t num_shards_c,
    uint32_t num_cores) {
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
                nhw_linear += config.out_nhw_per_core;
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
        case TensorMemoryLayout::BLOCK_SHARDED:
            // With block sharded layout scalars across sequential channels shard repeat, so we use repeats variable not
            // to recalculate scalars that we can reuse
            for (const std::vector<ScalarInfo>& scalars : scalars_per_core) {
                uint32_t repeats = (in_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) ? num_shards_c : 1;
                push_back_scalar_info_or_zero(config_vector, scalars, max_scalars_cnt, repeats);
            }
            break;
        case TensorMemoryLayout::WIDTH_SHARDED:
            // With width sharded layout scalars should be calulated only once, so we push them back num_shards_c times
            // but have only one array of scalars
            push_back_scalar_info_or_zero(config_vector, scalars_per_core[0], max_scalars_cnt, num_shards_c);
            break;

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

Pool2D::MultiCore::cached_program_t pool2d_multi_core_sharded_with_halo_v2_impl_new(
    Program& program,
    const Tensor& input,
    const Tensor& reader_indices,
    uint32_t reader_indices_size,
    Tensor& output,
    Pool2DType pool_type,
    uint32_t in_n,
    uint32_t in_c,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t kernel_size_h,
    uint32_t kernel_size_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t ceil_pad_h,
    uint32_t ceil_pad_w,
    bool ceil_mode,
    bool count_include_pad,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t num_shards_c,
    const MemoryConfig& out_mem_config,
    std::optional<int32_t> divisor_override,
    uint32_t memory_used) {
    // This should allocate a DRAM buffer on the device
    IDevice* device = input.device();
    tt::tt_metal::Buffer* src_dram_buffer = input.buffer();
    const tt::tt_metal::DeviceStorage& reader_indices_storage = reader_indices.device_storage();
    tt::tt_metal::Buffer* dst_dram_buffer = output.buffer();

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();

    tt::DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    const uint32_t in_nbytes = datum_size(in_df);
    const uint32_t out_nbytes = datum_size(out_df);

    const uint32_t in_nbytes_c = in_c / num_shards_c * in_nbytes;  // row of input (channels)
    const uint32_t in_nbytes_padded_c = input_shape[3] / num_shards_c * in_nbytes;  // row of input (channels)
    const uint32_t in_aligned_nbytes_c =
        tt::round_up(input_shape[3] / num_shards_c, tt::constants::TILE_WIDTH) * in_nbytes;
    const uint32_t out_nbytes_c = in_c / num_shards_c * out_nbytes;  // row of output (channels)

    constexpr tt::DataFormat indices_df =
        tt::DataFormat::RawUInt16;  // datatype_to_dataformat_converter(reader_indices.dtype());
    const uint32_t indices_nbytes = datum_size(indices_df);

    const uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;  // number of valid rows, to read
    const uint32_t kernel_size_hw_padded = tt::round_up(kernel_size_hw, tt::constants::TILE_HEIGHT);
    const uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / num_shards_c / tt::constants::TILE_WIDTH);
    const uint32_t out_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / num_shards_c / tt::constants::TILE_WIDTH);
    const bool last_tile_is_partial = (in_c / num_shards_c) % 32 != 0 && (in_c / num_shards_c) % 32 < 17;

    bool is_avg_pool = pool_type == Pool2DType::AVG_POOL2D;
    const bool is_large_kernel =
        is_partial_tile ? kernel_size_hw > tt::constants::TILE_HEIGHT / 2 : kernel_size_hw > tt::constants::TILE_HEIGHT;
    // For large kernel avg pool, we need to use fp32 accumulation to avoid precision error buildup over multiple
    // reduction stages, so we can only reduce 4 tiles at a time, otherwise we can reduce 8 tiles at a time.
    const uint32_t MAX_TILES_PER_REDUCTION = is_avg_pool && is_large_kernel ? 4 : 8;
    const bool is_wide_reduction = in_ntiles_c > MAX_TILES_PER_REDUCTION;

    // TODO: enable 32 sticks per tile for reduction for all cases, we can only support 16 row reductions for
    // partial tiles, and there is currently a bug forcing us to use 16 row reductions for avg pool when there
    // is 1 remainder C tile
    const uint32_t max_rows_for_reduction =
        !last_tile_is_partial ? tt::constants::TILE_HEIGHT : tt::constants::TILE_HEIGHT / 2;

    const uint32_t out_w_loop_count = std::ceil((float)out_w / nblocks);

    // distributing out_hw across the grid
    auto grid_size = device->compute_with_storage_grid_size();
    auto all_cores = input.shard_spec().value().grid;
    const uint32_t ncores = all_cores.num_cores();
    const uint32_t in_nhw_per_core = input.shard_spec()->shape[0];
    const uint32_t out_nhw_per_core = output.shard_spec()->shape[0];

    const uint32_t ncores_w = grid_size.x;

    // CBs
    const uint32_t multi_buffering_factor = 2;

    const uint32_t split_reader = 1;

    // scalar CB as coefficient of reduce
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;

    uint32_t next_cb_index = tt::CBIndex::c_0;
    const uint32_t in_scalar_cb_id_0 = next_cb_index++;
    const uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    const uint32_t in_scalar_cb_npages = 1 * multi_buffering_factor;
    TT_FATAL(in_scalar_cb_npages <= 2, "Kernel logic relys on scalar cb page number being <= 2");
    tt::tt_metal::create_cb(in_scalar_cb_id_0, program, all_cores, in_scalar_cb_pagesize, in_scalar_cb_npages, in_df);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_scalar_cb_id_0, in_scalar_cb_pagesize, in_scalar_cb_npages);

    const uint32_t bf16_scalar = get_bf16_pool_scalar(pool_type, kernel_size_h, kernel_size_w, divisor_override);
    const uint32_t bf16_init_value = get_bf16_pool_init_value(pool_type);
    bool one_scalar_per_core = is_pool_op_one_scalar_per_core(
        pool_type, ceil_mode, ceil_pad_h, ceil_pad_w, count_include_pad, pad_h, pad_w, divisor_override);

    uint32_t in_scalar_cb_id_1 = 32;
    if (is_avg_pool && split_reader && !one_scalar_per_core) {
        in_scalar_cb_id_1 = next_cb_index++;
        tt::tt_metal::create_cb(
            in_scalar_cb_id_1, program, all_cores, in_scalar_cb_pagesize, in_scalar_cb_npages, in_df);
        log_debug(
            tt::LogOp, "CB {} :: PS = {}, NP = {}", in_scalar_cb_id_1, in_scalar_cb_pagesize, in_scalar_cb_npages);
    }

    uint32_t clear_value_cb_id = 32;
    if (max_rows_for_reduction == tt::constants::TILE_HEIGHT || is_large_kernel ||
        (is_wide_reduction && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0)) {
        // CB storing just "clear value" (-inf for maxpool, 0 for avgpool)
        // is needed only if we use more then 16 sticks per tile for reduction
        // or if we use large kernel size.
        clear_value_cb_id = next_cb_index++;
        tt::tt_metal::create_cb(clear_value_cb_id, program, all_cores, tile_size(in_df), 1, in_df);
        log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", clear_value_cb_id, tile_size(in_df), 1);
    }

    // incoming data is the input cb instead of raw l1/dram addr
    // this input shard has halo and padding inserted.
    const uint32_t raw_in_cb_npages = input.shard_spec().value().shape[0];
    const uint32_t raw_in_cb_pagesize = input_shape[3] / num_shards_c * in_nbytes;
    auto [raw_in_cb_id, raw_in_cb] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, raw_in_cb_pagesize, raw_in_cb_npages, in_df, input.buffer());

    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", raw_in_cb_id, raw_in_cb_pagesize, raw_in_cb_npages);

    // reader indices
    const uint32_t in_reader_indices_cb_id = next_cb_index++;
    const uint32_t in_reader_indices_cb_pagesize =
        tt::round_up(reader_indices_size, 4);  // pagesize needs to be multiple of 4
    constexpr uint32_t in_reader_indices_cb_npages = 1;

    tt::tt_metal::create_cb(
        in_reader_indices_cb_id,
        program,
        all_cores,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages,
        indices_df,
        reader_indices_storage.get_buffer());

    log_debug(
        tt::LogOp,
        "CB {} :: PS = {}, NP = {}",
        in_reader_indices_cb_id,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages);
    uint32_t in_cb_sz = 0;
    uint32_t in_nblocks_c = 1;
    if (is_wide_reduction) {
        in_cb_sz = MAX_TILES_PER_REDUCTION * tt::constants::TILE_HW;
        in_nblocks_c = std::ceil((float)in_ntiles_c / MAX_TILES_PER_REDUCTION);
    } else {
        in_cb_sz = in_ntiles_c * tt::constants::TILE_HW;
    }

    // reader output == input to tilize
    const uint32_t in_cb_id_0 = next_cb_index++;  // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_id_1 = 32;                     // input rows for "multiple (out_nelems)" output pixels
    const uint32_t in_cb_page_padded = tt::round_up(
        in_cb_sz,
        tt::constants::TILE_HW);  // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    const uint32_t in_cb_pagesize = in_nbytes * in_cb_page_padded;
    const uint32_t in_cb_npages = multi_buffering_factor;

    tt::tt_metal::create_cb(in_cb_id_0, program, all_cores, in_cb_pagesize, in_cb_npages, in_df);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_cb_id_0, in_cb_pagesize, in_cb_npages);

    if (split_reader) {
        in_cb_id_1 = next_cb_index++;
        tt::tt_metal::create_cb(in_cb_id_1, program, all_cores, in_cb_pagesize, in_cb_npages, in_df);
        log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_cb_id_1, in_cb_pagesize, in_cb_npages);
    }

    // output of reduce == writer to write
    // output rows in RM
    // after reduction
    const uint32_t out_cb_pagesize = std::min(tt::constants::TILE_WIDTH, output.shard_spec().value().shape[1]) *
                                     out_nbytes;  // there is just one row of channels after each reduction (or 1 block
                                                  // of c if its greater than 8 tiles)
    const uint32_t out_cb_npages = output.shard_spec().value().shape[0] * out_ntiles_c;

    auto [out_cb_id, cb_out] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, out_cb_pagesize, out_cb_npages, out_df, output.buffer());
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", out_cb_id, out_cb_pagesize, out_cb_npages);

    TT_FATAL(output.memory_config().is_sharded(), "Output memory config needs to be sharded");

    /**
     * Reader Kernel: input rows -> input cb
     */
    CBHandle config_cb;
    tt::tt_metal::DeviceStorage scalar_config_storage;
    uint32_t config_cb_id = 32;
    Tensor config_tensor;
    if (!one_scalar_per_core) {
        // create config tensor
        AvgPoolConfig avg_pool_config = {
            .kernel_h = kernel_size_h,
            .kernel_w = kernel_size_w,
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
            .pad_h = pad_h / 2,  // pad_h is the total padding, so divide by 2 for each side
            .pad_w = pad_w / 2,  // pad_w is the total padding, so divide by 2 for each side
            .out_nhw_per_core = out_nhw_per_core,
            .divisor_override = divisor_override};
        config_tensor = create_scalar_config_tensor(
            avg_pool_config, input.memory_config().memory_layout(), in_n, num_shards_c, ncores);

        const std::array<uint32_t, 2> shard_shape =
            std::array<uint32_t, 2>({1, static_cast<uint32_t>(config_tensor.logical_shape()[-1])});
        const tt::tt_metal::ShardOrientation config_tensor_shard_orientation = input.shard_spec().value().orientation;
        const tt::tt_metal::ShardSpec config_shard_spec(
            input.shard_spec().value().grid, shard_shape, config_tensor_shard_orientation);
        const MemoryConfig memory_config{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec};
        const Tensor config_tensor_device = config_tensor.to_device(device, memory_config);

        constexpr tt::DataFormat config_df = tt::DataFormat::RawUInt32;
        scalar_config_storage = config_tensor_device.device_storage();
        tt::tt_metal::Buffer* config_buffer = scalar_config_storage.get_buffer();
        const uint32_t config_buffer_page_size = config_buffer->page_size();

        std::tie(config_cb_id, config_cb) = tt::tt_metal::create_cb(
            next_cb_index++, program, all_cores, config_buffer_page_size, 1, config_df, &*config_buffer);
    }
    std::vector<uint32_t> reader0_ct_args = {
        out_nhw_per_core,
        kernel_size_h,
        kernel_size_w,
        pad_w,
        in_aligned_nbytes_c,
        in_w,
        input_shape[3] / num_shards_c,
        split_reader,  // enable split reader
        0,             // split reader id
        bf16_scalar,
        bf16_init_value,
        in_nblocks_c,
        in_cb_sz,
        max_rows_for_reduction,
        ceil_pad_w,
        in_cb_id_0,
        in_cb_id_1,
        raw_in_cb_id,
        in_reader_indices_cb_id,
        in_scalar_cb_id_0,
        in_scalar_cb_id_1,
        clear_value_cb_id,
        (uint32_t)pool_type,
        one_scalar_per_core,
        config_cb_id,
        in_nbytes_c,
        in_nbytes_padded_c,
        multi_buffering_factor,
        stride_w};
    std::vector<uint32_t> reader1_ct_args = reader0_ct_args;
    reader1_ct_args[8] = 1;  // split reader id for reader1

    std::string reader_kernel_fname;
    if (is_large_kernel) {
        reader_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/"
            "reader_pool_2d_multi_core_sharded_with_halo_large_kernel_v2.cpp";
    } else {
        reader_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/"
            "reader_pool_2d_multi_core_sharded.cpp";
    }

    auto reader0_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
        .compile_args = reader0_ct_args};
    auto reader0_kernel = CreateKernel(program, reader_kernel_fname, all_cores, reader0_config);

    auto reader1_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::RISCV_1_default,
        .compile_args = reader1_ct_args};
    auto reader1_kernel = split_reader ? CreateKernel(program, reader_kernel_fname, all_cores, reader1_config) : 0;

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block ->
     * output cb
     */
    std::vector<uint32_t> compute_ct_args = {
        in_ntiles_c,
        kernel_size_hw,
        split_reader,
        out_nhw_per_core,
        input_shape[3] / num_shards_c,
        in_nblocks_c,
        max_rows_for_reduction,
        in_cb_id_0,
        in_cb_id_1,
        in_scalar_cb_id_0,
        in_scalar_cb_id_1,
        out_cb_id,
        one_scalar_per_core};

    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en =
            is_large_kernel && is_avg_pool,  // for large kernels average pool requires fp32 accumulation to avoid
                                             // precision error buildup over multiuple reduction stages
        .math_approx_mode = false,
        .compile_args = compute_ct_args,
        .defines = get_defines(pool_type)};
    std::string compute_kernel_fname;
    if (is_large_kernel) {
        compute_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/pool_2d_multi_core_large_kernel.cpp";
    } else {
        // both regular and wide reductions
        compute_kernel_fname = "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/pool_2d_multi_core.cpp";
    }

    auto compute_kernel = CreateKernel(program, compute_kernel_fname, all_cores, compute_config);

    uint32_t temporary_size = program.get_cb_memory_size();
    uint32_t post_allocate_size =
        input.device()->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;
    uint32_t l1_usage = calculate_L1_usage(
        input,
        pad_h,
        pad_w,
        ceil_pad_h,
        ceil_pad_w,
        ceil_mode,
        kernel_size_h,
        kernel_size_w,
        out_h,
        out_w,
        input.memory_config(),
        output.memory_config(),
        pool_type,
        count_include_pad,
        divisor_override);
    uint32_t output_cb_size = post_allocate_size - memory_used;

    // For now assume that if post_op_l1_allocation_size == 0 op is being run
    // in graph capture NO_DISPATCH mode.
    bool is_graph_capture_no_dispatch_mode = post_allocate_size == 0;
    TT_FATAL(
        temporary_size + output_cb_size == l1_usage || is_graph_capture_no_dispatch_mode,
        "Calculated CB size {} does not match with the actual CB size {}  ",
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
        log_debug(tt::LogOp, "in_addr: {}", src_dram_buffer->address());
        log_debug(tt::LogOp, "in_reader_indices_addr: {}", reader_indices_storage.get_buffer()->address());
        if (scalar_config_storage.is_allocated()) {
            log_debug(tt::LogOp, "scalar_config_addr: {}", scalar_config_storage.get_buffer()->address());
        } else {
            log_debug(tt::LogOp, "scalar_config_addr: not set");
        }
        log_debug(tt::LogOp, "out_addr: {}", dst_dram_buffer->address());
        log_debug(tt::LogOp, "kernel_size_h: {}", kernel_size_h);
        log_debug(tt::LogOp, "kernel_size_w: {}", kernel_size_w);
        log_debug(tt::LogOp, "kernel_size_hw: {}", kernel_size_hw);
        log_debug(tt::LogOp, "kernel_size_hw_padded: {}", kernel_size_hw_padded);
        log_debug(tt::LogOp, "stride_h: {}", stride_h);
        log_debug(tt::LogOp, "stride_w: {}", stride_w);
        log_debug(tt::LogOp, "pad_h: {}", pad_h);
        log_debug(tt::LogOp, "pad_w: {}", pad_w);
        log_debug(tt::LogOp, "out_h: {}", out_h);
        log_debug(tt::LogOp, "out_w: {}", out_w);
        log_debug(tt::LogOp, "out_c: {}", output_shape[3]);
        log_debug(tt::LogOp, "out_nbytes_c: {}", out_nbytes_c);
        log_debug(tt::LogOp, "in_h: {}", in_h);
        log_debug(tt::LogOp, "in_w: {}", in_w);
        log_debug(tt::LogOp, "in_c: {}", input_shape[3]);
        log_debug(tt::LogOp, "in_ntiles_c: {}", in_ntiles_c);
        log_debug(tt::LogOp, "in_nblocks_c: {}", in_nblocks_c);
        log_debug(tt::LogOp, "in_nbytes_c: {}", in_nbytes_c);
        log_debug(tt::LogOp, "in_aligned_nbytes_c: {}", in_aligned_nbytes_c);
        log_debug(tt::LogOp, "out_ntiles_c: {}", out_ntiles_c);
        log_debug(tt::LogOp, "ncores: {}", ncores);
        log_debug(tt::LogOp, "in_nhw_per_core: {}", in_nhw_per_core);
        log_debug(tt::LogOp, "out_nhw_per_core: {}", out_nhw_per_core);
        log_debug(tt::LogOp, "split_reader: {}", split_reader);
        log_debug(tt::LogOp, "multi_buffering_factor: {}", multi_buffering_factor);
        log_debug(tt::LogOp, "is_wide_reduction: {}", is_wide_reduction);
        log_debug(tt::LogOp, "is_large_kernel: {}", is_large_kernel);
        log_debug(tt::LogOp, "is_in_sharded: {}", input.memory_config().is_sharded());
        log_debug(tt::LogOp, "is_out_sharded: {}", output.memory_config().is_sharded());
    }

    // Capture reader_indices_storage to cache this with the program
    return {
        std::move(program),
        {.reader0_kernel = reader0_kernel,
         .reader1_kernel = reader1_kernel,
         .raw_in_cb = raw_in_cb,
         .cb_out = cb_out,
         .ncores = ncores,
         .ncores_w = ncores_w,
         .reader_indices_storage = reader_indices_storage,
         .scalar_config_storage = scalar_config_storage}};
}

Pool2D::MultiCore::cached_program_t Pool2D::MultiCore::create(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input_tensor_;
    const auto& sliding_window_config = op_attr.sliding_window_config_;
    const auto& pool_type = op_attr.pool_type_;
    const auto& out_mem_config = op_attr.memory_config_;
    bool count_include_pad = op_attr.count_include_pad_;
    std::optional<int32_t> divisor_override = op_attr.divisor_override_;

    tt::tt_metal::Program program{};

    auto parallel_config = sliding_window::ParallelConfig{
        .grid = input.shard_spec().value().grid,
        .shard_scheme = input.memory_config().memory_layout(),
        .shard_orientation = input.shard_spec().value().orientation,
    };

    auto output_shape = sliding_window_config.get_output_shape();
    uint32_t out_h = output_shape[1];
    uint32_t out_w = output_shape[2];

    bool is_block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    auto in_n = sliding_window_config.batch_size;
    auto in_c = sliding_window_config.channels;
    auto in_h = sliding_window_config.input_hw.first;
    auto in_w = sliding_window_config.input_hw.second;
    auto kernel_size_h = sliding_window_config.window_hw.first;
    auto kernel_size_w = sliding_window_config.window_hw.second;
    auto stride_h = sliding_window_config.stride_hw.first;
    auto stride_w = sliding_window_config.stride_hw.second;
    auto pad_h = sliding_window_config.get_pad_h();
    auto pad_w = sliding_window_config.get_pad_w();
    auto ceil_pad_h = sliding_window_config.get_ceil_pad_h();
    auto ceil_pad_w = sliding_window_config.get_ceil_pad_w();
    auto ceil_mode = sliding_window_config.ceil_mode;
    auto dilation_h = sliding_window_config.dilation_hw.first;
    auto dilation_w = sliding_window_config.dilation_hw.second;
    auto num_shards_c = sliding_window_config.num_cores_c;

    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config, op_trace_metadata);
    std::vector<std::vector<uint16_t>> top_left_indices =
        sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, stride_w);

    Tensor reader_indices = sliding_window::construct_on_host_config_tensor(top_left_indices, parallel_config);
    Tensor reader_indices_on_device =
        sliding_window::move_config_tensor_to_device(reader_indices, parallel_config, is_block_sharded, input.device());

    return pool2d_multi_core_sharded_with_halo_v2_impl_new(
        program,
        input,
        reader_indices_on_device,
        top_left_indices[0].size(),
        output_tensor,
        pool_type,
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        ceil_pad_h,
        ceil_pad_w,
        ceil_mode,
        count_include_pad,
        dilation_h,
        dilation_w,
        num_shards_c,
        out_mem_config,
        divisor_override,
        op_attr.memory_used);
}

void Pool2D::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& reader0_kernel = cached_program.shared_variables.reader0_kernel;
    auto& reader1_kernel = cached_program.shared_variables.reader1_kernel;
    auto& raw_in_cb = cached_program.shared_variables.raw_in_cb;
    auto& cb_out = cached_program.shared_variables.cb_out;
    auto& ncores = cached_program.shared_variables.ncores;
    auto& ncores_w = cached_program.shared_variables.ncores_w;

    const auto& input_tensor = tensor_args.input_tensor_;

    auto src_buffer = input_tensor.buffer();
    bool input_sharded = input_tensor.is_sharded();

    auto dst_buffer = output_tensor.buffer();
    bool out_sharded = output_tensor.is_sharded();

    if (input_sharded) {
        UpdateDynamicCircularBufferAddress(program, raw_in_cb, *src_buffer);
    }
    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_out, *dst_buffer);
    }
}

}  // namespace ttnn::operations::pool
