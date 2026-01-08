// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/common/common.hpp"

#include <algorithm>
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn::operations::data_movement {

ttnn::Shape squeeze_shape_to_ND(const ttnn::Shape& shape, const uint32_t n) {
    if (shape.rank() <= n) {
        return shape;
    }
    ttnn::SmallVector<uint32_t> shape_nd(n);
    std::copy(shape.view().rbegin(), shape.view().rbegin() + n, shape_nd.rbegin());
    const auto rank_diff_end = shape.rank() - n + 1;
    shape_nd[0] = std::accumulate(shape.cbegin(), shape.cbegin() + rank_diff_end, 1, std::multiplies<uint32_t>());

    return ttnn::Shape(shape_nd);
}

ttnn::Shape squeeze_shape_to_4D(const ttnn::Shape& shape) { return squeeze_shape_to_ND(shape, 4); }
ttnn::Shape squeeze_shape_to_3D(const ttnn::Shape& shape) { return squeeze_shape_to_ND(shape, 3); }

ttnn::Tensor squeeze_from_ND_to_4D(const ttnn::Tensor& tensor, const std::optional<CoreRangeSet>& sub_core_grids) {
    auto shape = tensor.logical_shape();
    auto rank = shape.rank();
    TT_FATAL(shape.rank() >= 4, "Tensor has to be of rank larger than 4! Instead is {}", shape.rank());
    if (rank == 4) {
        return tensor;
    }
    int i = 0;
    // This is a workaround for now, it will be fixed in another PR
    if (shape[i] == 1) {
        auto squeezed = tensor;
        while (rank > 4 && shape[i] == 1) {
            squeezed = ttnn::squeeze(squeezed, 0);
            rank = squeezed.logical_shape().rank();
            i++;
        }
        if (rank <= 4) {
            return squeezed;
        }
        return ttnn::reshape(
            squeezed,
            squeeze_shape_to_4D(shape),
            std::nullopt,
            std::nullopt,
            TileReshapeMapMode::CACHE,
            sub_core_grids);
    }
    return ttnn::reshape(
        tensor, squeeze_shape_to_4D(shape), std::nullopt, std::nullopt, TileReshapeMapMode::CACHE, sub_core_grids);
}

ttnn::Shape unsqueeze_shape_to_ND(const ttnn::Shape& shape, const uint32_t n) {
    ttnn::SmallVector<uint32_t> shape_vector(n, 1);
    std::copy(shape.view().rbegin(), shape.view().rend(), shape_vector.rbegin());
    return ttnn::Shape(shape_vector);
}

ttnn::Shape unsqueeze_shape_to_3D(const ttnn::Shape& shape) { return unsqueeze_shape_to_ND(shape, 3); };
ttnn::Shape unsqueeze_shape_to_4D(const ttnn::Shape& shape) { return unsqueeze_shape_to_ND(shape, 4); };

ttnn::Shape squeeze_or_unsqueeze_shape_to_ND(const ttnn::Shape& shape, const uint32_t n) {
    const auto input_rank = shape.rank();
    if (input_rank == n) {
        return shape;
    }
    if (input_rank < n) {
        return unsqueeze_shape_to_ND(shape, n);
    }
    return squeeze_shape_to_ND(shape, n);
}

enum DatumIndex { WormholeIndex = 0, BlackholeIndex = 1 };

float get_transaction_noc_bw(
    uint32_t transaction_size, const std::map<uint32_t, std::array<float, 2>>& dict, int index) {
    uint32_t lower_pow2 = std::pow(2, std::floor(std::log2(transaction_size)));

    uint32_t upper_pow2 = std::pow(2, std::ceil(std::log2(transaction_size)));

    auto lower_it = dict.lower_bound(lower_pow2);
    auto upper_it = dict.lower_bound(upper_pow2);

    // If lower bound not found, use first entry
    if (lower_it == dict.end()) {
        lower_it = dict.begin();
    }

    // If upper bound not found, use last entry
    if (upper_it == dict.end()) {
        upper_it = std::prev(dict.end());
    }
    float lower_bw = lower_it->second[index];
    float upper_bw = upper_it->second[index];

    if (transaction_size - lower_pow2 < upper_pow2 - transaction_size) {
        return lower_bw;
    }
    return upper_bw;
}

uint32_t get_effective_l1_cores(
    uint32_t transaction_size,
    int index,
    bool is_write,
    const std::map<uint32_t, std::array<float, 2>>& l1_read_bw,
    const std::map<uint32_t, std::array<float, 2>>& l1_write_bw,
    uint32_t num_nocs,
    uint32_t num_cores) {
    float max_bw = index == WormholeIndex ? 32.0f : 50.0f;
    auto aggregate_bw = max_bw * num_nocs;
    float achieved_l1_bw = get_transaction_noc_bw(transaction_size, is_write ? l1_write_bw : l1_read_bw, index);
    uint32_t effective_cores = std::ceil((float)aggregate_bw / (float)achieved_l1_bw);
    effective_cores = std::min(effective_cores, num_cores);  // Limit to available cores
    return effective_cores;
}

uint32_t get_effective_dram_cores(
    uint32_t transaction_size,
    int index,
    const std::map<uint32_t, std::array<float, 2>>& dram_bw,
    bool single_noc,
    uint32_t num_cores) {
    auto aggregate_bw = single_noc == 1 ? 190 : 265;
    float achieved_dram_bw = get_transaction_noc_bw(transaction_size, dram_bw, index);
    uint32_t effective_cores = std::ceil((float)aggregate_bw / (float)achieved_dram_bw);
    effective_cores = std::min(effective_cores, num_cores);  // Limit to available cores
    return effective_cores;
}

std::vector<uint32_t> get_cycles_for_transaction_size(
    uint32_t transaction_size,
    bool is_dram,
    bool is_local,
    uint32_t num_transactions,
    uint32_t num_cores,
    int index,
    bool is_read,
    const std::map<uint32_t, std::array<float, 2>>& l1_local_bw,
    const std::map<uint32_t, std::array<float, 2>>& l1_read_bw,
    const std::map<uint32_t, std::array<float, 2>>& l1_write_bw,
    const std::map<uint32_t, std::array<float, 2>>& dram_bw) {
    auto transaction_type = is_local ? l1_local_bw : (is_read ? l1_read_bw : l1_write_bw);
    if (is_dram) {
        transaction_type = dram_bw;
    }
    // measured initial latency based on the transaction and device types
    uint32_t latency_cyles = 1;
    if (transaction_type == l1_local_bw) {
        latency_cyles = index == WormholeIndex ? 56 : 88;
    } else if (transaction_type == l1_read_bw) {
        latency_cyles = index == WormholeIndex ? 259 : 403;
    } else if (transaction_type == l1_write_bw) {
        latency_cyles = index == WormholeIndex ? 256 : 404;
    } else if (transaction_type == dram_bw) {
        latency_cyles = index == WormholeIndex ? 358 : 529;
    }

    transaction_size = std::max(transaction_size, 16u);
    auto transaction_bw = get_transaction_noc_bw(transaction_size, transaction_type, index);
    float device_frequency_hz = index == WormholeIndex ? 1e9 : 1.2e9;
    uint32_t cycles =
        std::ceil((float)(num_transactions * transaction_size * device_frequency_hz) / (float)(transaction_bw * 1e9));
    return {cycles, latency_cyles};
}

int common_tm_bw_model(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    bool output_only,
    int compute_cycles,
    bool per_faceline,
    bool split_op,
    bool bcast_local,
    bool concat_op) {
    // the bw maps assigns a measured bandwidth per transaction size for each device architecture
    std::map<uint32_t, std::array<float, 2>> dram_bw = {
        {16, {0.436, 0.387}},
        {32, {0.868, 0.772}},
        {64, {1.736, 1.545}},
        {128, {3.489, 3.088}},
        {256, {6.975, 6.176}},
        {512, {13.889, 12.361}},
        {1024, {27.891, 24.71}},
        {2048, {28.411, 49.164}},
        {4096, {28.227, 50.238}},
        {8192, {28.537, 50.393}},
        {16384, {27.831, 50.636}},
        {32768, {27.758, 50.695}},
        {65536, {28.694, 50.626}}};

    std::map<uint32_t, std::array<float, 2>> l1_read_bw = {
        {16, {0.868, 0.671}},
        {32, {1.724, 1.336}},
        {64, {3.477, 2.673}},
        {128, {6.885, 5.354}},
        {256, {13.794, 10.691}},
        {512, {27.143, 21.382}},
        {1024, {28.976, 42.771}},
        {2048, {29.742, 47.977}},
        {4096, {29.544, 49.34}},
        {8192, {28.728, 49.961}},
        {16384, {28.7, 50.287}},
        {32768, {28.618, 50.335}},
        {65536, {28.7, 50.403}}};

    std::map<uint32_t, std::array<float, 2>> l1_write_bw = {
        {16, {0.681, 0.511}},
        {32, {1.254, 1.018}},
        {64, {2.709, 2.036}},
        {128, {5.417, 4.072}},
        {256, {10.823, 8.143}},
        {512, {21.668, 16.284}},
        {1024, {27.837, 32.593}},
        {2048, {27.811, 48.644}},
        {4096, {27.811, 49.828}},
        {8192, {27.808, 50.219}},
        {16384, {27.808, 50.578}},
        {32768, {27.811, 50.412}},
        {65536, {28.808, 50.383}}};

    std::map<uint32_t, std::array<float, 2>> l1_local_bw = {
        {16, {0.868, 0.671}},
        {32, {1.724, 1.337}},
        {64, {3.477, 2.675}},
        {128, {6.899, 5.354}},
        {256, {13.791, 10.708}},
        {512, {27.594, 21.413}},
        {1024, {27.696, 42.792}},
        {2048, {27.911, 46.455}},
        {4096, {27.811, 48.5}},
        {8192, {27.808, 49.639}},
        {16384, {27.814, 50.171}},
        {32768, {27.805, 50.123}},
        {65536, {27.84, 50.21}}};

    const auto& input_shape = concat_op ? output_tensor.padded_shape() : input_tensor.padded_shape();
    auto element_size_bytes = input_tensor.element_size();
    bool input_is_2d_sharded =
        input_tensor.memory_config().is_sharded() && input_tensor.memory_config().shard_spec().has_value();
    bool input_is_dram = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool input_is_tiled = input_tensor.layout() == Layout::TILE;
    uint32_t input_size_bytes = input_shape.volume() * element_size_bytes;

    uint32_t device_rows = input_tensor.device()->compute_with_storage_grid_size().x;
    uint32_t device_cols = input_tensor.device()->compute_with_storage_grid_size().y;
    uint32_t num_nocs = device_rows + device_cols;

    auto arch = input_tensor.device()->arch();
    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        TT_THROW("Unsupported architecture for common_bw_model: {}", arch);
    }
    uint32_t num_cores = device_rows * device_cols;

    uint32_t total_num_cores = num_cores;
    uint32_t index = (arch == tt::ARCH::WORMHOLE_B0) ? WormholeIndex : BlackholeIndex;

    uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    uint32_t single_tile_size = tile_width * tile_height * element_size_bytes;
    uint32_t input_transaction_size = input_is_tiled ? single_tile_size : input_shape[-1] * element_size_bytes;
    const uint32_t max_transaction_size = 2048u;  // size with highest bw

    if (input_is_2d_sharded) {
        const auto& input_shard_shape = input_tensor.memory_config().shard_spec().value().shape;
        input_transaction_size = input_is_tiled ? single_tile_size : input_shard_shape[1] * element_size_bytes;
        // can increase transaction size for height-sharded tensors
        if (!input_is_tiled && !input_is_dram &&
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            uint32_t row_size = input_shard_shape[1] * element_size_bytes;
            uint32_t multi_row_size = input_shard_shape[0] * row_size;
            input_transaction_size = std::min(multi_row_size, max_transaction_size);
        }
    }

    uint32_t num_read_transactions = std::ceil((float)input_size_bytes / (float)input_transaction_size);

    bool output_is_dram = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool output_is_tiled = output_tensor.layout() == Layout::TILE;
    bool output_2d_is_sharded =
        output_tensor.memory_config().is_sharded() && output_tensor.memory_config().shard_spec().has_value();

    const auto& output_shape =
        split_op ? input_shape
                 : output_tensor
                       .padded_shape();  // for split op, we need to write all the splits (equivalent to input shape)
    uint32_t output_size_bytes = output_shape.volume() * element_size_bytes;

    uint32_t output_transaction_size = output_is_tiled ? single_tile_size : output_shape[-1] * element_size_bytes;

    if (output_2d_is_sharded) {
        const auto& output_shard_shape = output_tensor.memory_config().shard_spec().value().shape;
        output_transaction_size = output_is_tiled ? single_tile_size : output_shard_shape[1] * element_size_bytes;
        if (!output_is_tiled && !output_is_dram &&
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            uint32_t out_row_size = output_shard_shape[1] * element_size_bytes;
            uint32_t out_multi_row_size = output_shard_shape[0] * out_row_size;
            output_transaction_size = std::min(out_multi_row_size, max_transaction_size);
        }
    }
    // for permute/transpose op, we need to write per faceline instead of tile in row invariant case to avoid
    // overestimations
    bool row_invariant = input_shape[-1] == output_shape[-1] && input_shape[-2] != output_shape[-2];
    bool process_per_faceline = per_faceline && output_is_tiled && row_invariant;
    output_transaction_size = process_per_faceline ? tile_width / 2 * element_size_bytes : output_transaction_size;

    if (process_per_faceline) {
        const auto& output_logical_shape = output_tensor.logical_shape();
        auto output_volume = output_logical_shape.volume();
        output_size_bytes = output_volume * element_size_bytes;
    }
    uint32_t num_write_transactions = std::ceil((float)output_size_bytes / (float)output_transaction_size);
    auto updated_input_transactions = num_read_transactions;
    auto updated_output_transactions = num_write_transactions;
    // limit number of cores to max aggregate bw to avoid congestion
    if (input_is_dram && output_is_dram) {
        uint32_t input_effective_cores =
            get_effective_dram_cores(input_transaction_size, index, dram_bw, false, total_num_cores);
        uint32_t output_effective_cores =
            get_effective_dram_cores(output_transaction_size, index, dram_bw, false, total_num_cores);
        auto actual_read_cores = std::min(input_effective_cores, num_read_transactions);
        auto actual_write_cores = std::min(output_effective_cores, num_write_transactions);
        updated_input_transactions = std::ceil((float)num_read_transactions / (float)actual_read_cores);
        updated_output_transactions = std::ceil((float)num_write_transactions / (float)actual_write_cores);
        num_cores = actual_read_cores;
        if (updated_input_transactions < updated_output_transactions) {
            num_cores = actual_write_cores;
        }

    } else if (input_is_dram) {
        num_cores = get_effective_dram_cores(input_transaction_size, index, dram_bw, false, total_num_cores);
        updated_input_transactions = std::ceil((float)num_read_transactions / (float)num_cores);
    } else if (output_is_dram) {
        num_cores = get_effective_dram_cores(output_transaction_size, index, dram_bw, false, total_num_cores);
        updated_output_transactions = std::ceil((float)num_write_transactions / (float)num_cores);
    }
    // local noc transactions for l1 sharded tensors
    bool is_local = input_is_2d_sharded && !input_is_dram && output_2d_is_sharded && !output_is_dram &&
                    (output_tensor.memory_config().shard_spec().value().grid ==
                     input_tensor.memory_config().shard_spec().value().grid);

    is_local = is_local || bcast_local;
    if (!input_is_dram && !output_is_dram) {
        uint32_t input_effective_cores = get_effective_l1_cores(
            input_transaction_size, index, false, l1_read_bw, l1_write_bw, num_nocs, total_num_cores);
        uint32_t output_effective_cores = get_effective_l1_cores(
            output_transaction_size, index, true, l1_read_bw, l1_write_bw, num_nocs, total_num_cores);
        auto actual_read_cores = std::min(input_effective_cores, num_read_transactions);
        auto actual_write_cores = std::min(output_effective_cores, num_write_transactions);
        auto updated_input_transactions = std::ceil((float)num_read_transactions / (float)actual_read_cores);
        auto updated_output_transactions = std::ceil((float)num_write_transactions / (float)actual_write_cores);
        num_cores = actual_read_cores;
        if (updated_input_transactions < updated_output_transactions) {
            num_cores = actual_write_cores;
        }
    } else if (!input_is_dram) {
        auto num_cores_ = get_effective_l1_cores(
            input_transaction_size, index, false, l1_read_bw, l1_write_bw, num_nocs, total_num_cores);
        if (updated_output_transactions < (num_read_transactions / num_cores_)) {
            num_cores = num_cores_;
        }
    } else if (!output_is_dram) {
        auto num_cores_ = get_effective_l1_cores(
            output_transaction_size, index, true, l1_read_bw, l1_write_bw, num_nocs, total_num_cores);
        if (updated_input_transactions < (num_write_transactions / num_cores_)) {
            num_cores = num_cores_;
        }
    }
    uint32_t total_read_cycles_not_local = 0x7FFFFFFF;
    uint32_t total_write_cycles_not_local = 0x7FFFFFFF;
    if (is_local) {
        // sometimes more cores (even if not local) is better
        // computes both and takes the minimum value between the two
        if (num_cores > output_tensor.memory_config().shard_spec().value().grid.num_cores()) {
            auto read_cycles_not_local = get_cycles_for_transaction_size(
                input_transaction_size,
                input_is_dram,
                false,
                std::ceil((float)num_read_transactions / (float)num_cores),
                num_cores,
                index,
                true,
                l1_local_bw,
                l1_read_bw,
                l1_write_bw,
                dram_bw);

            auto write_cycles_not_local = get_cycles_for_transaction_size(
                output_transaction_size,
                output_is_dram,
                false,
                std::ceil((float)num_write_transactions / (float)num_cores),
                num_cores,
                index,
                false,
                l1_local_bw,
                l1_read_bw,
                l1_write_bw,
                dram_bw);
            total_read_cycles_not_local = read_cycles_not_local[0] + read_cycles_not_local[1];
            total_write_cycles_not_local = write_cycles_not_local[0] + write_cycles_not_local[1];
        }
    }
    num_cores = is_local ? output_tensor.memory_config().shard_spec().value().grid.num_cores() : num_cores;
    uint32_t compute_cores = std::max(num_read_transactions, num_write_transactions);
    // parallelize work over cores
    // assume distribution of work is balanced between cores
    num_read_transactions = std::ceil((float)num_read_transactions / (float)num_cores);
    num_write_transactions = std::ceil((float)num_write_transactions / (float)num_cores);

    auto read_cycles = get_cycles_for_transaction_size(
        input_transaction_size,
        input_is_dram,
        is_local,
        num_read_transactions,
        num_cores,
        index,
        true,
        l1_local_bw,
        l1_read_bw,
        l1_write_bw,
        dram_bw);

    auto write_cycles = get_cycles_for_transaction_size(
        output_transaction_size,
        output_is_dram,
        is_local,
        num_write_transactions,
        num_cores,
        index,
        false,
        l1_local_bw,
        l1_read_bw,
        l1_write_bw,
        dram_bw);
    uint32_t total_read_cycles = read_cycles[0] + read_cycles[1];
    uint32_t total_write_cycles = write_cycles[0] + write_cycles[1];

    int ideal_dev_clock_cycles = 1;
    if ((input_is_dram && output_is_dram) || bcast_local) {
        ideal_dev_clock_cycles =
            output_only ? total_write_cycles : (int)std::ceil((float)(total_read_cycles + write_cycles[0]));
    } else {
        auto ideal_dev_clock_cycles_not_local =
            output_only ? total_write_cycles_not_local
                        : std::max(total_read_cycles_not_local, total_write_cycles_not_local);
        ideal_dev_clock_cycles = output_only ? total_write_cycles : std::max(total_read_cycles, total_write_cycles);
        ideal_dev_clock_cycles = std::min<unsigned int>(ideal_dev_clock_cycles_not_local, ideal_dev_clock_cycles);
    }
    // latency for llk compute kernels
    int total_compute_cycles = 0;
    if (compute_cycles > 0) {
        total_compute_cycles = std::ceil((float)compute_cycles / (float)(std::min(total_num_cores, compute_cores)));
    }
    return std::max(ideal_dev_clock_cycles, total_compute_cycles);
}

uint32_t get_estimated_size_of_cbs(
    const Tensor& input_tensor_a,
    const uint32_t input_single_tile_size,
    const uint32_t output_single_tile_size,
    const uint32_t num_tiles_per_row) {
    uint32_t cb_src0_size = input_single_tile_size * num_tiles_per_row;
    uint32_t cb_output_size = output_single_tile_size * num_tiles_per_row;
    return cb_src0_size + cb_output_size;
}

uint32_t get_max_l1_space(const Tensor& input_tensor_a) {
    auto* device = input_tensor_a.device();
    auto lowest_address = device->lowest_occupied_compute_l1_address();
    uint32_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
    max_l1_space = max_l1_space - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    return max_l1_space;
}

bool is_enough_space(
    const Tensor& input_tensor_a,
    const uint32_t input_single_tile_size,
    const uint32_t output_single_tile_size,
    const uint32_t num_tiles_per_row) {
    uint32_t max_l1_space = get_max_l1_space(input_tensor_a);
    uint32_t estimated_size_of_cbs =
        get_estimated_size_of_cbs(input_tensor_a, input_single_tile_size, output_single_tile_size, num_tiles_per_row);
    return max_l1_space > estimated_size_of_cbs;
}

ttnn::Tensor pad_to_tile_vol(
    const ttnn::Tensor& tensor,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config) {
    const auto& logical_shape = tensor.logical_shape();
    auto padded_shape = tensor.padded_shape();
    auto rank = logical_shape.rank();
    if (padded_shape[-1] % tt::constants::TILE_WIDTH != 0 || padded_shape[-2] % tt::constants::TILE_HEIGHT != 0) {
        TT_ASSERT(rank >= 2, "rank of tensor to pad to tile must be at least 2.");

        auto padded_height = tt::round_up(padded_shape[-2], tt::constants::TILE_HEIGHT);
        auto padded_width = tt::round_up(padded_shape[-1], tt::constants::TILE_WIDTH);
        uint32_t num_non_hw_dims = rank - 2u;
        auto padding_vec = ttnn::SmallVector<std::array<uint32_t, 2>>(num_non_hw_dims, {0, 0});
        padding_vec.reserve(rank);
        padding_vec.emplace_back(0, padded_height - padded_shape[-2]);
        padding_vec.emplace_back(0, padded_width - padded_shape[-1]);

        auto padded_output = ttnn::pad(tensor, padding_vec, value, use_multicore, memory_config);
        TT_FATAL(
            padded_output.padded_shape()[-1] % tt::constants::TILE_WIDTH == 0 &&
                padded_output.padded_shape()[-2] % tt::constants::TILE_HEIGHT == 0,
            "pad_to_tile_vol: output tensor must be divisible by tile size");
        return padded_output;
    }
    return tensor;
}
uint32_t wrap_index(int index, int size) { return index < 0 ? size + index : index; }

// not unit tested, use with caution
uint16_t float_to_uint16(float f) {
    // For positive infinity, return the maximum uint16_t value
    // For negative infinity, return the minimum uint16_t value
    if (std::isinf(f)) {
        return (f > 0) ? std::numeric_limits<uint16_t>::max() : std::numeric_limits<uint16_t>::min();
    }

    // For Not-a-Number (NaN), return 0
    if (std::isnan(f)) {
        return 0;
    }

    // Handle overflow and underflow
    const float max_uint16 = static_cast<float>(std::numeric_limits<uint16_t>::max());
    const float min_uint16 = static_cast<float>(std::numeric_limits<uint16_t>::min());
    if (f >= max_uint16) {
        return std::numeric_limits<uint16_t>::max();
    }
    if (f <= min_uint16) {
        return std::numeric_limits<uint16_t>::min();
    }

    // For all other finite values, safely cast after rounding
    return static_cast<uint16_t>(std::round(f));
}

// based off pack_two_bfloat16_into_uint32
uint32_t pack_two_uint16_into_uint32(std::pair<uint16_t, uint16_t> two_uint16s) {
    // first -> lower 16
    // second -> upper 16
    return (uint32_t)two_uint16s.first | ((uint32_t)two_uint16s.second << 16);
}

ttnn::Shape compute_padded_shape(ttnn::Shape logical_shape, const uint32_t tile_height, const uint32_t tile_width) {
    // Special case: if input tensor is 1D row-major, after tiling output tensor will have
    // 1D logical shape but 2D padded shape
    if (logical_shape.rank() == 1) {
        logical_shape = ttnn::Shape({1, logical_shape[0]});
    }

    ttnn::SmallVector<uint32_t> output_shape_vec(logical_shape.rank());
    std::copy(logical_shape.cbegin(), logical_shape.cend(), output_shape_vec.begin());

    const std::array<uint32_t, 2> tile_shape = {tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT};
    auto shapeit = tile_shape.rbegin();

    std::for_each(output_shape_vec.rbegin(), output_shape_vec.rbegin() + 2, [&shapeit](auto& x) {
        x = tt::round_up(x, *(shapeit++));
    });

    return ttnn::Shape(output_shape_vec);
}

ttnn::Shape pad_to_tile_shape(const ttnn::Shape& unpadded_shape) {
    using namespace tt::constants;
    auto rank = unpadded_shape.rank();
    TT_ASSERT(rank >= 1, "rank of shape to pad to tile shape must be at least 1.");
    SmallVector<uint32_t> padded_shape_vec(rank);

    for (auto i = 0; i < rank; ++i) {
        padded_shape_vec[i] = unpadded_shape[i];
    }
    if (rank >= 1) {
        auto w = tt::round_up(unpadded_shape[rank - 1], TILE_WIDTH);
        padded_shape_vec[rank - 1] = w;
    }
    if (rank >= 2) {
        auto h = tt::round_up(unpadded_shape[rank - 2], TILE_HEIGHT);
        padded_shape_vec[rank - 2] = h;
    }
    return Shape(padded_shape_vec);
}

std::array<uint32_t, 2> compute_block_sharded_shard_shape(
    const std::array<uint32_t, 2>& squeezed_tensor_hw,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::CoreCoord& grid_size,
    const tt::tt_metal::ShardOrientation& orientation,
    const uint32_t total_num_cores) {
    TT_FATAL(
        grid_size.y * grid_size.x == total_num_cores,
        "compute_block_sharded_shard_shape received a core grid shape that does not match the total number of cores");
    auto adjusted_grid_size = grid_size;
    if (orientation == tt::tt_metal::ShardOrientation::COL_MAJOR) {
        // for col major, we partition the width of the tensor along the height of the core grid
        std::swap(adjusted_grid_size.x, adjusted_grid_size.y);
    }

    auto [tensor_height, tensor_width] = squeezed_tensor_hw;
    auto tensor_height_padded_to_tile =
        layout == tt::tt_metal::Layout::TILE
            ? tt::round_up(tensor_height, adjusted_grid_size.y * tt::constants::TILE_HEIGHT)
            : tensor_height;
    std::array<uint32_t, 2> shard_shape = {
        tt::div_up(tensor_height_padded_to_tile, adjusted_grid_size.y), tt::div_up(tensor_width, adjusted_grid_size.x)};

    return shard_shape;
}

std::array<uint32_t, 2> compute_width_sharded_shard_shape(
    const std::array<uint32_t, 2>& squeezed_tensor_hw, const uint32_t total_num_cores) {
    return {squeezed_tensor_hw[0], tt::div_up(squeezed_tensor_hw[1], total_num_cores)};
}

std::array<uint32_t, 2> compute_height_sharded_shard_shape(
    const std::array<uint32_t, 2>& squeezed_tensor_hw,
    const tt::tt_metal::Layout& layout,
    const uint32_t total_num_cores) {
    auto [tensor_height, tensor_width] = squeezed_tensor_hw;
    auto squeezed_height_padded_to_tile =
        layout == tt::tt_metal::Layout::TILE ? tt::round_up(tensor_height, total_num_cores) : tensor_height;
    return {tt::div_up(squeezed_height_padded_to_tile, total_num_cores), tensor_width};
}

ttnn::MemoryConfig create_sharded_memory_config(
    const ttnn::Shape& logical_shape,
    const tt::tt_metal::CoreRangeSet& core_grid,
    const ShardStrategy& strategy,
    const tt::tt_metal::ShardOrientation& orientation,
    std::optional<std::array<uint32_t, 2>> shard_shape,
    const tt::tt_metal::Layout& layout) {
    auto rank = logical_shape.rank();
    TT_FATAL(rank >= 2, "rank of tensor to shard must be at least 2.");

    ttnn::TensorMemoryLayout tensor_memory_layout{};
    if (strategy == ShardStrategy::BLOCK) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::BLOCK_SHARDED;
    } else if (strategy == ShardStrategy::WIDTH) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::WIDTH_SHARDED;
    } else if (strategy == ShardStrategy::HEIGHT) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
    }

    auto height = logical_shape[-2];
    auto width = logical_shape[-1];
    std::array<uint32_t, 2> computed_shard_shape{};

    if (shard_shape.has_value()) {
        computed_shard_shape = shard_shape.value();
    } else {
        uint32_t batch_size = 1;
        for (int i = 0; i < rank - 2; i++) {
            batch_size *= logical_shape[i];
        }

        auto tensor_height = batch_size * height;
        auto tensor_width = width;
        std::array<uint32_t, 2> squeezed_tensor_hw{tensor_height, tensor_width};
        auto total_num_cores = core_grid.num_cores();
        CoreCoord grid_size = core_grid.bounding_box().grid_size();

        switch (strategy) {
            case ShardStrategy::BLOCK:
                computed_shard_shape = compute_block_sharded_shard_shape(
                    squeezed_tensor_hw, layout, grid_size, orientation, total_num_cores);
                break;
            case ShardStrategy::WIDTH:
                computed_shard_shape = compute_width_sharded_shard_shape(squeezed_tensor_hw, total_num_cores);
                break;
            case ShardStrategy::HEIGHT:
                computed_shard_shape = compute_height_sharded_shard_shape(squeezed_tensor_hw, layout, total_num_cores);
                break;
            default: TT_ASSERT(false, "Invalid shard strategy");
        }
    }

    if (layout == tt::tt_metal::Layout::TILE) {
        auto [shard_height, shard_width] = computed_shard_shape;
        auto tile_divides_shard_height = shard_height % tt::constants::TILE_HEIGHT == 0;
        auto tile_divides_shard_width = shard_width % tt::constants::TILE_WIDTH == 0;
        TT_FATAL(
            tile_divides_shard_width && tile_divides_shard_height,
            "For sharding tiled tensors, the shard shape must fit neatly into tiles but "
            "create_sharded_memory_config got shard width {} and shard height {} while "
            "on this architecture we have tile width {} and tile height {}",
            computed_shard_shape[0],
            computed_shard_shape[1],
            tt::constants::TILE_WIDTH,
            tt::constants::TILE_HEIGHT);
    }

    auto shard_spec = tt::tt_metal::ShardSpec(core_grid, computed_shard_shape, orientation);
    return ttnn::MemoryConfig(tensor_memory_layout, ttnn::BufferType::L1, shard_spec);
}

std::pair<uint32_t, std::array<uint32_t, 2>> tensor_coord_to_height_sharded_coord(
    const std::span<const uint32_t>& tensor_shape,
    const std::span<const uint32_t>& shard_shape,
    const std::span<const uint32_t>& tensor_coord) {
    std::array<uint32_t, 2> tensor_shape_2d{0, 0};
    for (size_t i = 0; i < tensor_shape.size(); i++) {
        if (i == tensor_shape.size() - 1) {
            // width dimension, goes unmodified
            tensor_shape_2d[1] = tensor_shape[i];
        } else {
            // height dimension, squeeze into 2D shape
            if (tensor_shape_2d[0] == 0) {
                // first time we've seen this dimension
                tensor_shape_2d[0] = tensor_shape[i];
            } else {
                tensor_shape_2d[0] *= tensor_shape[i];
            }
        }
    }

    std::array<uint32_t, 2> tensor_coord_2d{0, tensor_coord.back()};
    uint32_t height_2d = 0;
    for (size_t i = 0; i < tensor_coord.size() - 1; i++) {
        std::vector<uint32_t> page_shapes(tensor_shape.begin() + i + 1, tensor_shape.end() - 1);
        auto component_sum =
            tensor_coord[i] * std::accumulate(page_shapes.begin(), page_shapes.end(), 1, std::multiplies<uint32_t>());
        height_2d += component_sum;
    }
    tensor_coord_2d[0] = height_2d;

    uint32_t shard_height = shard_shape[0];
    uint32_t w_in_shard = tensor_coord_2d[1];
    uint32_t h_in_shard = height_2d % shard_height;
    uint32_t which_shard = height_2d / shard_height;

    std::array<uint32_t, 2> shard_coord{h_in_shard, w_in_shard};
    return std::make_pair(which_shard, shard_coord);
}

uint32_t get_num_pages(const ttnn::Tensor& tensor) {
    if (tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        return tt::div_up(tensor.padded_shape().volume(), tensor.padded_shape()[-1]);
    }
    const auto& tile_shape = tensor.tensor_spec().tile().get_tile_shape();
    return tt::div_up(tensor.padded_shape().volume(), tile_shape[0] * tile_shape[1]);
}

}  // namespace ttnn::operations::data_movement
