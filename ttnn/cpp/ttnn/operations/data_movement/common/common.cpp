// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

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


ttnn::Tensor squeeze_from_ND_to_4D(const ttnn::Tensor& tensor) {
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
        return ttnn::reshape(squeezed, squeeze_shape_to_4D(shape));
    }
    return ttnn::reshape(tensor, squeeze_shape_to_4D(shape));
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
    } else if (input_rank < n) {
        return unsqueeze_shape_to_ND(shape, n);
    } else {
        return squeeze_shape_to_ND(shape, n);
    }
}

float interpolate_transaction_bw(
    uint32_t transaction_size, const std::map<uint32_t, std::array<float, 2>>& dict, int index) {
    auto it = dict.lower_bound(transaction_size);
    if (it == dict.begin()) {
        return it->second[index];
    }
    if (it == dict.end()) {
        return std::prev(it)->second[index];
    }
    if (it->first == transaction_size) {
        return it->second[index];
    }
    auto upper = it;
    auto lower = std::prev(it);
    float bw = lower->second[index] + (upper->second[index] - lower->second[index]) *
                                          (float(transaction_size - lower->first) / float(upper->first - lower->first));
    return bw;
}

uint32_t get_effective_l1_cores(
    uint32_t transaction_size,
    int index,
    bool is_write,
    std::map<uint32_t, std::array<float, 2>> l1_read_bw,
    std::map<uint32_t, std::array<float, 2>> l1_write_bw) {
    const uint32_t max_l1_cores = (index == 0) ? (8 + 8) : (9 + 12);
    int num_cores = (index == 0) ? 64 : 108;
    float max_bw = index == 0 ? 28.0f : 34.0f;
    auto aggregate_bw = max_bw * max_l1_cores;
    float achieved_l1_bw = interpolate_transaction_bw(transaction_size, is_write ? l1_write_bw : l1_read_bw, index);
    uint32_t effective_cores = std::ceil((float)aggregate_bw / (float)achieved_l1_bw);
    if (effective_cores > num_cores) {
        effective_cores = num_cores;
    }
    return effective_cores;
}

uint32_t get_effective_dram_cores(
    uint32_t transaction_size,
    uint32_t num_dram_channels,
    bool is_read,
    int index,
    std::map<uint32_t, std::array<float, 2>> dram_bw,
    bool single_noc) {
    int num_cores = (index == 0) ? 64 : 108;
    auto aggregate_bw = single_noc == 1 ? 150 : 220;
    float achieved_dram_bw = interpolate_transaction_bw(transaction_size, dram_bw, index);
    uint32_t effective_cores = std::ceil((float)aggregate_bw / (float)achieved_dram_bw);
    if (effective_cores > num_cores) {
        effective_cores = num_cores;  // Limit to available cores
    }
    return effective_cores;
}

uint32_t get_cycles_for_transaction_size(
    uint32_t transaction_size,
    bool is_dram,
    bool is_local,
    uint32_t num_transactions,
    uint32_t num_cores,
    int index,
    bool is_read,
    std::map<uint32_t, std::array<float, 2>> l1_local_bw,
    std::map<uint32_t, std::array<float, 2>> l1_read_bw,
    std::map<uint32_t, std::array<float, 2>> l1_write_bw,
    std::map<uint32_t, std::array<float, 2>> dram_bw) {
    auto transaction_type = is_local ? l1_local_bw : (is_read ? l1_read_bw : l1_write_bw);
    if (is_dram) {
        transaction_type = dram_bw;
    }
    uint32_t latency_cyles = 1;
    if (transaction_type == l1_local_bw) {
        latency_cyles = index == 0 ? 56 : 52;
    } else if (transaction_type == l1_read_bw) {
        latency_cyles = index == 0 ? 259 : 278;
    } else if (transaction_type == l1_write_bw) {
        latency_cyles = index == 0 ? 256 : 279;
    } else if (transaction_type == dram_bw) {
        latency_cyles = index == 0 ? 358 : 1737;
    }

    transaction_size = std::max(transaction_size, 16u);
    auto transaction_bw = interpolate_transaction_bw(transaction_size, transaction_type, index);
    float device_frequency_hz = index == 0 ? 1e9 : 1.2e9;
    uint32_t cycles = 1;
    if (is_dram) {
        cycles = std::ceil(
            (float)(num_transactions * transaction_size * device_frequency_hz) / (float)(transaction_bw * 1e9));
    } else {
        cycles = std::ceil(
            (float)(num_transactions * transaction_size * device_frequency_hz) / (float)(transaction_bw * 1e9));
    }
    return cycles + latency_cyles;
}

int common_tm_bw_model(const Tensor& input_tensor, const Tensor& output_tensor, bool output_only, int compute_cycles) {
    std::map<uint32_t, std::array<float, 2>> dram_bw = {
        {16, {0.436, 0.651}},
        {32, {0.868, 1.295}},
        {64, {1.736, 2.591}},
        {128, {3.489, 5.182}},
        {256, {6.975, 10.366}},
        {512, {13.889, 20.723}},
        {1024, {27.891, 32.65}},
        {2048, {28.411, 33.587}},
        {4096, {28.227, 32.686}},
        {8192, {28.537, 24.456}},
        {16384, {27.831, 23.934}},
        {32768, {27.758, 23.702}},
        {65536, {28.694, 26.328}}};

    std::map<uint32_t, std::array<float, 2>> l1_read_bw = {
        {16, {0.868, 1.176}},
        {32, {1.724, 2.319}},
        {64, {3.477, 4.649}},
        {128, {6.885, 9.275}},
        {256, {13.794, 18.623}},
        {512, {27.143, 34.602}},
        {1024, {28.976, 35.935}},
        {2048, {29.742, 35.95}},
        {4096, {29.544, 35.646}},
        {8192, {28.728, 34.447}},
        {16384, {28.7, 34.456}},
        {32768, {28.618, 34.456}},
        {65536, {28.7, 34.452}}};

    std::map<uint32_t, std::array<float, 2>> l1_write_bw = {
        {16, {0.681, 0.897}},
        {32, {1.254, 1.781}},
        {64, {2.709, 3.553}},
        {128, {5.417, 7.12}},
        {256, {10.823, 14.25}},
        {512, {21.668, 28.488}},
        {1024, {27.837, 33.509}},
        {2048, {27.811, 33.505}},
        {4096, {27.811, 33.505}},
        {8192, {27.808, 33.505}},
        {16384, {27.808, 33.505}},
        {32768, {27.811, 33.501}},
        {65536, {28.808, 33.505}}};

    std::map<uint32_t, std::array<float, 2>> l1_local_bw = {
        {16, {0.868, 1.174}},
        {32, {1.724, 2.326}},
        {64, {3.477, 4.704}},
        {128, {6.899, 9.413}},
        {256, {13.791, 18.565}},
        {512, {27.594, 31.737}},
        {1024, {27.696, 33.505}},
        {2048, {27.911, 33.501}},
        {4096, {27.811, 33.484}},
        {8192, {27.808, 33.514}},
        {16384, {27.814, 33.505}},
        {32768, {27.805, 33.398}},
        {65536, {27.84, 33.497}}};

    if (input_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Input tensor not on DEVICE?!");
    }
    const auto& input_shape = input_tensor.padded_shape();
    auto element_size_bytes = input_tensor.element_size();
    bool input_is_sharded = input_tensor.memory_config().is_sharded();
    bool input_is_dram = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool input_is_tiled = input_tensor.layout() == Layout::TILE;
    uint32_t input_size_bytes = input_shape.volume() * element_size_bytes;
    if (input_shape.rank() == 4) {
        printf("Input tensor shape: %u %u %u %u\n", input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    }
    printf("input is tile %d, sharded %d, dram %d\n", input_is_tiled, input_is_sharded, input_is_dram);
    auto arch = input_tensor.device()->arch();
    int num_cores = (arch == tt::ARCH::WORMHOLE_B0) ? 64 : 108;
    int total_num_cores = num_cores;
    int index = (arch == tt::ARCH::WORMHOLE_B0) ? 0 : 1;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    uint32_t single_tile_size = tile_width * tile_height * element_size_bytes;
    uint32_t input_transaction_size = input_is_tiled ? single_tile_size : input_shape[-1] * element_size_bytes;
    printf("initial input transaction size %u\n", input_transaction_size);
    if (input_is_sharded) {
        const auto& input_shard_shape = input_tensor.memory_config().shard_spec().value().shape;
        input_transaction_size = input_is_tiled ? single_tile_size : input_shard_shape[1] * element_size_bytes;
        // can increase transaction size for height-sharded tensors
        if (!input_is_tiled && !input_is_dram &&
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            uint32_t row_size = input_shard_shape[1] * element_size_bytes;
            uint32_t multi_row_size = input_shard_shape[0] * row_size;
            input_transaction_size = std::min(multi_row_size, 65536u);
            printf("input is height sharded, transaction size %u\n", input_transaction_size);
        }
    }

    uint32_t num_read_transactions = std::ceil((float)input_size_bytes / (float)input_transaction_size);
    printf("initial number of read transactions %u\n", num_read_transactions);

    if (output_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }
    bool output_is_dram = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool output_is_tiled = output_tensor.layout() == Layout::TILE;
    bool output_is_sharded = output_tensor.memory_config().is_sharded();

    const auto& output_shape = output_tensor.padded_shape();
    uint32_t output_size_bytes = output_shape.volume() * element_size_bytes;
    if (output_shape.rank() == 4) {
        printf(
            "Output tensor shape: %u %u %u %u\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    }
    printf("output is tile %d, sharded %d, dram %d\n", output_is_tiled, output_is_sharded, output_is_dram);

    uint32_t output_transaction_size = output_is_tiled ? single_tile_size : output_shape[-1] * element_size_bytes;
    printf("initial output transaction size %u\n", output_transaction_size);
    if (output_is_sharded) {
        const auto& output_shard_shape = output_tensor.memory_config().shard_spec().value().shape;
        output_transaction_size = output_is_tiled ? single_tile_size : output_shard_shape[1] * element_size_bytes;
        if (!output_is_tiled && !output_is_dram &&
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            uint32_t out_row_size = output_shard_shape[1] * element_size_bytes;
            uint32_t out_multi_row_size = output_shard_shape[0] * out_row_size;
            output_transaction_size = std::min(out_multi_row_size, 65536u);
            printf("output is height sharded, transaction size %u\n", output_transaction_size);
        }
    }
    uint32_t num_write_transactions = std::ceil((float)output_size_bytes / (float)output_transaction_size);
    printf("initial number of write transactions %u\n", num_write_transactions);
    int num_dram_channels = (arch == tt::ARCH::WORMHOLE_B0) ? 12 : 8;
    auto updated_input_transactions = num_read_transactions;
    auto updated_output_transactions = num_write_transactions;
    // limit number of cores to max aggregate bw to avoid congestion
    if (input_is_dram && output_is_dram) {
        printf("input and output are both dram\n");

        uint32_t input_effective_cores =
            get_effective_dram_cores(input_transaction_size, num_dram_channels, true, index, dram_bw, false);
        printf("input effective cores %u\n", input_effective_cores);
        uint32_t output_effective_cores =
            get_effective_dram_cores(output_transaction_size, num_dram_channels, false, index, dram_bw, false);
        printf("output effective cores %u\n", output_effective_cores);
        updated_input_transactions = std::ceil((float)num_read_transactions / (float)input_effective_cores);
        updated_output_transactions = std::ceil((float)num_write_transactions / (float)output_effective_cores);
        num_cores = input_effective_cores;
        if (updated_input_transactions < updated_output_transactions) {  // to do permute specific
            num_cores = output_effective_cores;
        }
        printf("num cores %u\n", num_cores);

    } else if (input_is_dram) {
        printf("only input is dram\n");
        num_cores = get_effective_dram_cores(input_transaction_size, num_dram_channels, true, index, dram_bw, true);
        printf("num cores %u\n", num_cores);
        updated_input_transactions = std::ceil((float)num_read_transactions / (float)num_cores);
    } else if (output_is_dram) {
        printf("only output is dram\n");
        num_cores = get_effective_dram_cores(output_transaction_size, num_dram_channels, false, index, dram_bw, true);
        printf("num cores %u\n", num_cores);
        updated_output_transactions = std::ceil((float)num_write_transactions / (float)num_cores);
    }
    // local noc transactions for l1 sharded tensors
    bool is_local = input_is_sharded && !input_is_dram && output_is_sharded && !output_is_dram &&
                    (output_tensor.memory_config().shard_spec().value().grid ==
                     input_tensor.memory_config().shard_spec().value().grid);
    printf("is local %d\n", is_local);
    float read_util = 1.0f;
    float write_util = 1.0f;
    if (!input_is_dram && !output_is_dram) {
        printf("input and output are both l1\n");
        uint32_t input_effective_cores =
            get_effective_l1_cores(input_transaction_size, index, false, l1_read_bw, l1_write_bw);
        printf("input effective cores %u\n", input_effective_cores);
        uint32_t output_effective_cores =
            get_effective_l1_cores(output_transaction_size, index, true, l1_read_bw, l1_write_bw);
        printf("output effective cores %u\n", output_effective_cores);
        auto actual_read_cores = std::min(input_effective_cores, num_read_transactions);
        auto actual_write_cores = std::min(output_effective_cores, num_write_transactions);
        num_cores = std::min(actual_read_cores, actual_write_cores);
        read_util = (float)num_cores / (float)input_effective_cores;
        write_util = (float)num_cores / (float)output_effective_cores;
        printf("num cores %u\n", num_cores);
    } else if (!input_is_dram) {
        printf("only input is l1\n");
        auto num_cores_ = get_effective_l1_cores(input_transaction_size, index, false, l1_read_bw, l1_write_bw);
        printf("num cores _ %u\n", num_cores_);
        if (updated_output_transactions < (num_read_transactions / num_cores_)) {
            num_cores = num_cores_;
            printf("num cores %u\n", num_cores);
        }
    } else if (!output_is_dram) {
        printf("only output is l1\n");
        auto num_cores_ = get_effective_l1_cores(output_transaction_size, index, true, l1_read_bw, l1_write_bw);
        printf("num cores _ %u\n", num_cores_);
        if (updated_input_transactions < (num_write_transactions / num_cores_)) {
            num_cores = num_cores_;
            printf("num cores %u\n", num_cores);
        }
    }

    num_cores = is_local ? input_tensor.memory_config().shard_spec().value().grid.num_cores() : num_cores;
    printf("final num cores %u\n", num_cores);
    // parallelize work over cores
    // assume distribution of work is balanced between cores
    num_read_transactions = std::ceil((float)num_read_transactions / (float)num_cores);
    num_write_transactions = std::ceil((float)num_write_transactions / (float)num_cores);
    printf("final number of read transactions %u\n", num_read_transactions);
    printf("final number of write transactions %u\n", num_write_transactions);
    auto total_read_cycles = get_cycles_for_transaction_size(
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

    auto total_write_cycles = get_cycles_for_transaction_size(
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
    printf("total read cycles %u\n", total_read_cycles);
    // Use max(read, write) to account for overlap
    int ideal_dev_clock_cycles = 1;
    if (input_is_dram && output_is_dram) {
        // Using the utilization to determine overlap potential
        float overlap_factor = 1.0f;
        if (read_util < 0.3f && write_util < 0.3f) {
            // Both operations have low utilization - can overlap significantly
            overlap_factor = 0.6f;
        } else if (read_util < 0.3f || write_util < 0.3f) {
            // One operation has low utilization - can partially overlap
            overlap_factor = 0.8f;
        } else if (read_util > 0.8f && write_util > 0.8f) {
            // Both operations near peak - minimal overlap
            overlap_factor = 1.0f;
        } else {
            // Moderate utilization - some overlap possible
            overlap_factor = 0.9f;
        }
        printf("overlap factor %f\n", overlap_factor);
        ideal_dev_clock_cycles = std::ceil((float)(total_read_cycles + total_write_cycles) * (float)overlap_factor);
    } else {
        ideal_dev_clock_cycles = output_only ? total_write_cycles : std::max(total_read_cycles, total_write_cycles);
    }
    int total_compute_cycles = 0;
    if (compute_cycles > 0) {
        total_compute_cycles = std::ceil((float)compute_cycles / (float)total_num_cores);
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
    auto device = input_tensor_a.device();
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
    QueueId queue_id,
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
        auto padding_vec = ttnn::SmallVector<std::pair<uint32_t, uint32_t>>(num_non_hw_dims, {0, 0});
        padding_vec.reserve(rank);
        padding_vec.emplace_back(0, padded_height - padded_shape[-2]);
        padding_vec.emplace_back(0, padded_width - padded_shape[-1]);

        constexpr bool pad_use_multicore = true;
        auto padded_output = ttnn::pad(queue_id, tensor, padding_vec, value, use_multicore, memory_config);
        TT_FATAL(
            padded_output.padded_shape()[-1] % tt::constants::TILE_WIDTH == 0 &&
                padded_output.padded_shape()[-2] % tt::constants::TILE_HEIGHT == 0,
            "pad_to_tile_vol: output tensor must be divisible by tile size");
        return padded_output;
    }
    return tensor;
}
uint32_t wrap_index(int index, int size) { return index < 0 ? size + index : index; }

ttnn::Shape compute_padded_shape(
    const ttnn::Shape& logical_shape, const uint32_t tile_height, const uint32_t tile_width) {
    if (logical_shape.rank() == 1) {
        return ttnn::Shape{tile_height, tile_width};
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

std::array<uint32_t, 2> compute_block_sharded_shard_shape(const std::array<uint32_t, 2>& squeezed_tensor_hw,
                                                          const tt::tt_metal::Layout& layout,
                                                          const tt::tt_metal::CoreCoord& grid_size,
                                                          const tt::tt_metal::ShardOrientation& orientation,
                                                          const uint32_t total_num_cores) {
    TT_FATAL(grid_size.y * grid_size.x == total_num_cores, "compute_block_sharded_shard_shape received a core grid shape that does not match the total number of cores");
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
    std::array<uint32_t, 2> shard_shape = {tt::div_up(tensor_height_padded_to_tile, adjusted_grid_size.y),
                                           tt::div_up(tensor_width, adjusted_grid_size.x)};

    return shard_shape;
}

std::array<uint32_t, 2> compute_width_sharded_shard_shape(const std::array<uint32_t, 2>& squeezed_tensor_hw,
                                                          const uint32_t total_num_cores) {
    return {squeezed_tensor_hw[0], tt::div_up(squeezed_tensor_hw[1], total_num_cores)};
}

std::array<uint32_t, 2> compute_height_sharded_shard_shape(const std::array<uint32_t, 2>& squeezed_tensor_hw,
                                                           const tt::tt_metal::Layout& layout,
                                                           const uint32_t total_num_cores) {
    auto [tensor_height, tensor_width] = squeezed_tensor_hw;
    auto squeezed_height_padded_to_tile = layout == tt::tt_metal::Layout::TILE
                                                    ? tt::round_up(tensor_height, total_num_cores)
                                                    : tensor_height;
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
    std::array<uint32_t, 2> computed_shard_shape;

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
                computed_shard_shape = compute_block_sharded_shard_shape(squeezed_tensor_hw, layout, grid_size, orientation, total_num_cores);
                break;
            case ShardStrategy::WIDTH:
                computed_shard_shape = compute_width_sharded_shard_shape(squeezed_tensor_hw, total_num_cores);
                break;
            case ShardStrategy::HEIGHT:
                computed_shard_shape = compute_height_sharded_shard_shape(squeezed_tensor_hw, layout, total_num_cores);
                break;
            default:
                TT_ASSERT(false, "Invalid shard strategy");
        }
    }

    if (layout == tt::tt_metal::Layout::TILE) {
        auto [shard_height, shard_width] = computed_shard_shape;
        auto tile_divides_shard_height = shard_height % tt::constants::TILE_HEIGHT == 0;
        auto tile_divides_shard_width = shard_width % tt::constants::TILE_WIDTH == 0;
        TT_FATAL(tile_divides_shard_width && tile_divides_shard_height,
                 "For sharding tiled tensors, the shard shape must fit neatly into tiles but "
                 "create_sharded_memory_config got shard width {} and shard height {} while "
                 "on this architecture we have tile width {} and tile height {}",
                 computed_shard_shape[0], computed_shard_shape[1], tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT);
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

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
