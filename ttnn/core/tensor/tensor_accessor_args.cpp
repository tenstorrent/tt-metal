// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/tensor/tensor_accessor_args.hpp>

#include <tt-metalium/device.hpp>
#include <ttnn/tensor/types.hpp>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
void append_sharded_args(
    const Buffer& buffer, tensor_accessor::ArgsConfig args_config, std::vector<uint32_t>& args, bool is_runtime) {
    TT_FATAL(buffer.buffer_distribution_spec(), "Buffer must have a buffer distribution spec");

    const auto& buffer_distribution_spec = buffer.buffer_distribution_spec().value();
    const auto& tensor_shape = buffer_distribution_spec.tensor_shape_in_pages();
    const auto& shard_shape = buffer_distribution_spec.shard_shape_in_pages();
    const auto& bank_coords = buffer_distribution_spec.cores();

    auto add_rank = args_config.test(tensor_accessor::ArgConfig::RuntimeRank) == is_runtime;
    auto add_num_banks = args_config.test(tensor_accessor::ArgConfig::RuntimeNumBanks) == is_runtime;
    auto add_tensor_shape = args_config.test(tensor_accessor::ArgConfig::RuntimeTensorShape) == is_runtime;
    auto add_shard_shape = args_config.test(tensor_accessor::ArgConfig::RuntimeShardShape) == is_runtime;
    auto add_bank_coords = args_config.test(tensor_accessor::ArgConfig::RuntimeBankCoords) == is_runtime;

    size_t rank = tensor_shape.size();
    size_t n_banks = bank_coords.size();
    TT_FATAL(
        rank <= tt::tt_metal::MAX_NUM_DIMENSIONS,
        "Rank must be less than or equal to {} for rank",
        tt::tt_metal::MAX_NUM_DIMENSIONS);

    size_t n_args =
        add_rank + add_num_banks + rank * add_tensor_shape + rank * add_shard_shape + n_banks * add_bank_coords;
    if (!is_runtime) {
        n_args += 1;  // +1 for the args_config config
    }
    args.reserve(args.size() + n_args);

    if (!is_runtime) {
        args.push_back(args_config.raw());
    }

    if (add_rank) {
        args.push_back(rank);
    }
    if (add_num_banks) {
        args.push_back(n_banks);
    }
    if (add_tensor_shape) {
        args.insert(args.end(), tensor_shape.cbegin(), tensor_shape.cend());
    }
    if (add_shard_shape) {
        args.insert(args.end(), shard_shape.cbegin(), shard_shape.cend());
    }

    if (add_bank_coords) {
        auto device = buffer.device();
        auto bank_type = buffer.core_type();
        for (size_t i = 0; i < n_banks; i += 2) {
            // We don't virtualize DRAM coord, since we need logical x coord == bank_id to calculate the address
            const auto coord1 =
                buffer.is_dram() ? bank_coords[i] : device->virtual_core_from_logical_core(bank_coords[i], bank_type);

            if (i + 1 < n_banks) {
                // Pack two coordinates into one uint32_t if we have a pair
                const auto coord2 = buffer.is_dram()
                                        ? bank_coords[i + 1]
                                        : device->virtual_core_from_logical_core(bank_coords[i + 1], bank_type);
                args.push_back(
                    ((coord2.x & 0xFF) << 24) | ((coord2.y & 0xFF) << 16) | ((coord1.x & 0xFF) << 8) |
                    (coord1.y & 0xFF));
            } else {
                // Handle odd number of coordinates by setting the second coordinate to zero
                args.push_back(((coord1.x & 0xFF) << 8) | (coord1.y & 0xFF));
            }
        }
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TensorAccessorArgs::TensorAccessorArgs(const Buffer& buffer, tensor_accessor::ArgsConfig args_config) :
    buffer_(&buffer), args_config_(args_config) {
    if (is_sharded(buffer.buffer_layout())) {
        args_config_.set(tensor_accessor::ArgConfig::Sharded);
    } else {
        args_config_ = tensor_accessor::ArgConfig::None;
    }
    args_config_.set(tensor_accessor::ArgConfig::IsDram, buffer.is_dram());

    if (args_config_.test(tensor_accessor::ArgConfig::RuntimeRank)) {
        TT_FATAL(
            args_config_.test(tensor_accessor::ArgConfig::RuntimeTensorShape) &&
                args_config_.test(tensor_accessor::ArgConfig::RuntimeShardShape),
            "If rank is runtime, tensor_shape and shard_shape must also be runtime");
    }
    if (args_config_.test(tensor_accessor::ArgConfig::RuntimeNumBanks)) {
        TT_FATAL(
            args_config_.test(tensor_accessor::ArgConfig::RuntimeBankCoords),
            "If num_banks is runtime, bank_coords must also be runtime");
    }
}

void TensorAccessorArgs::append_args(
    std::vector<uint32_t>& compile_time_args, std::vector<uint32_t>& common_runtime_args) const {
    if (args_config_.test(tensor_accessor::ArgConfig::Sharded)) {
        CMAKE_UNIQUE_NAMESPACE::append_sharded_args(*buffer_, args_config_, compile_time_args, /* is_runtime */ false);
        CMAKE_UNIQUE_NAMESPACE::append_sharded_args(*buffer_, args_config_, common_runtime_args, /* is_runtime */ true);
    } else {
        compile_time_args.push_back(args_config_.raw());
    }
}

void TensorAccessorArgs::append_args(std::vector<uint32_t>& compile_time_args) const {
    TT_FATAL(
        (args_config_ & tensor_accessor::ArgConfig::Runtime).raw() == 0,
        "Common runtime arguments are required for ArgsConfig {}",
        args_config_.raw());
    if (args_config_.test(tensor_accessor::ArgConfig::Sharded)) {
        CMAKE_UNIQUE_NAMESPACE::append_sharded_args(*buffer_, args_config_, compile_time_args, /* is_runtime */ false);
    } else {
        compile_time_args.push_back(args_config_.raw());
    }
}

std::vector<uint32_t> TensorAccessorArgs::get_compile_time_args() const {
    std::vector<uint32_t> compile_time_args;
    if (args_config_.test(tensor_accessor::ArgConfig::Sharded)) {
        CMAKE_UNIQUE_NAMESPACE::append_sharded_args(*buffer_, args_config_, compile_time_args, /* is_runtime */ false);
    } else {
        compile_time_args.push_back(args_config_.raw());
    }
    return compile_time_args;
}

std::vector<uint32_t> TensorAccessorArgs::get_common_runtime_args() const {
    std::vector<uint32_t> common_runtime_args;
    if (args_config_.test(tensor_accessor::ArgConfig::Sharded)) {
        CMAKE_UNIQUE_NAMESPACE::append_sharded_args(*buffer_, args_config_, common_runtime_args, /* is_runtime */ true);
    }
    return common_runtime_args;
}

}  // namespace tt::tt_metal
