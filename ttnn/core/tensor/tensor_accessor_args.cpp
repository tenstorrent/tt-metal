// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/tensor/tensor_accessor_args.hpp>

#include <tt-metalium/device.hpp>
#include <ttnn/tensor/types.hpp>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> prepare_sharded_args(
    const Buffer& buffer, tensor_accessor::ArgsConfig args_config) {
    TT_FATAL(buffer.buffer_distribution_spec(), "Buffer must have a buffer distribution spec");

    const auto& buffer_distribution_spec = buffer.buffer_distribution_spec().value();
    const auto& tensor_shape = buffer_distribution_spec.get_tensor_shape_in_pages();
    const auto& shard_shape = buffer_distribution_spec.get_shard_shape_in_pages();
    const auto& bank_coords = buffer_distribution_spec.get_cores();

    auto rank_rt = args_config.test(tensor_accessor::ArgConfig::RuntimeRank);
    auto num_banks_rt = args_config.test(tensor_accessor::ArgConfig::RuntimeNumBanks);
    auto tensor_shape_rt = args_config.test(tensor_accessor::ArgConfig::RuntimeTensorShape);
    auto shard_shape_rt = args_config.test(tensor_accessor::ArgConfig::RuntimeShardShape);
    auto bank_coords_rt = args_config.test(tensor_accessor::ArgConfig::RuntimeBankCoords);

    size_t rank = tensor_shape.size();
    size_t n_banks = bank_coords.size();
    TT_FATAL(
        rank <= tt::tt_metal::MAX_NUM_DIMENSIONS,
        "Rank must be less than or equal to {} for rank",
        tt::tt_metal::MAX_NUM_DIMENSIONS);
    TT_FATAL(
        !rank_rt || (tensor_shape_rt && shard_shape_rt),
        "If rank is runtime, tensor_shape and shard_shape must also be runtime");
    TT_FATAL(!num_banks_rt || bank_coords_rt, "If num_banks is runtime, bank_coords must also be runtime");

    size_t n_compile_time_args = 1 + !rank_rt + !num_banks_rt + rank * !tensor_shape_rt + rank * !shard_shape_rt +
                                 n_banks * !bank_coords_rt;  // +1 for the crta config
    size_t n_runtime_args =
        rank_rt + num_banks_rt + rank * tensor_shape_rt + rank * shard_shape_rt + n_banks * bank_coords_rt;
    std::vector<uint32_t> compile_time_args;
    std::vector<uint32_t> runtime_args;
    compile_time_args.reserve(n_compile_time_args);
    runtime_args.reserve(n_runtime_args);
    auto& rank_args = rank_rt ? runtime_args : compile_time_args;
    auto& num_banks_args = num_banks_rt ? runtime_args : compile_time_args;
    auto& tensor_shape_args = tensor_shape_rt ? runtime_args : compile_time_args;
    auto& shard_shape_args = shard_shape_rt ? runtime_args : compile_time_args;
    auto& bank_coords_args = bank_coords_rt ? runtime_args : compile_time_args;

    compile_time_args.push_back(args_config.raw());
    rank_args.push_back(rank);
    num_banks_args.push_back(n_banks);
    tensor_shape_args.insert(tensor_shape_args.end(), tensor_shape.cbegin(), tensor_shape.cend());
    shard_shape_args.insert(shard_shape_args.end(), shard_shape.cbegin(), shard_shape.cend());

    auto device = buffer.device();
    auto bank_type = buffer.core_type();
    for (size_t i = 0; i < n_banks; i += 2) {
        const auto virtual_coord1 = device->virtual_core_from_logical_core(bank_coords[i], bank_type);

        if (i + 1 < n_banks) {
            // Pack two coordinates into one uint32_t if we have a pair
            const auto virtual_coord2 = device->virtual_core_from_logical_core(bank_coords[i + 1], bank_type);
            bank_coords_args.push_back(
                ((virtual_coord2.x & 0xFF) << 24) | ((virtual_coord2.y & 0xFF) << 16) |
                ((virtual_coord1.x & 0xFF) << 8) | (virtual_coord1.y & 0xFF));
        } else {
            // Handle odd number of coordinates by setting the second coordinate to zero
            bank_coords_args.push_back(((virtual_coord1.x & 0xFF) << 8) | (virtual_coord1.y & 0xFF));
        }
    }

    return {std::move(compile_time_args), std::move(runtime_args)};
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TensorAccessorArgs::TensorAccessorArgs(const Buffer& buffer, tensor_accessor::ArgsConfig args_config) {
    if (is_sharded(buffer.buffer_layout())) {
        args_config.set(tensor_accessor::ArgConfig::Sharded);
        args_config.set(tensor_accessor::ArgConfig::IsDram, buffer.is_dram());
        std::tie(compile_time_args, runtime_args) = CMAKE_UNIQUE_NAMESPACE::prepare_sharded_args(buffer, args_config);
        return;
    }

    args_config = tensor_accessor::ArgConfig::None;
    args_config.set(tensor_accessor::ArgConfig::IsDram, buffer.is_dram());
    compile_time_args = {args_config.raw()};
}

}  // namespace tt::tt_metal
