// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include "tt_metal/common/constants.hpp"

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::dram_prefetcher {

void DramPrefetcher::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() >= 1, "Must have at least one input tensor");
    // Check that all tensors are on the same device
    for (const auto& tensor : input_tensors) {
        TT_FATAL(tensor.device() == input_tensors[0].device(), "All tensors must be on the same device");
    }
    // Check that all tensors' k is divisible by 24
    for (const auto& tensor : input_tensors) {
        TT_FATAL(tensor.get_legacy_shape()[0] % 24 == 0, "All tensors' k must be divisible by 24");
    }
    TT_FATAL(global_cb.has_value(), "Global circular buffer must be provided");
    // Check that global_cb sender_receiver_core_mapping has same number of receivers for each sender core
    auto sender_receiver_core_mapping = global_cb->sender_receiver_core_mapping();
    for (const auto& [sender_core, receiver_core_range] : sender_receiver_core_mapping) {
        TT_FATAL(
            receiver_core_range.size() == sender_receiver_core_mapping.begin()->second.size(),
            "Global circular buffer must have same number of receivers for each sender core");
    }
}
std::vector<ttnn::SimpleShape> DramPrefetcher::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0).get_logical_shape()};
}
std::vector<Tensor> DramPrefetcher::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Configure L1 interleaved memory layout

    // TODO: set tensor address to be global_cb sender cores address or write out to another tensor
    // create tensor with buffer and address of receiver cores address

    MemoryConfig output_mem_config = L1_MEMORY_CONFIG;

    auto input_tensor = input_tensors.at(0);

    auto shard_shape = input_tensor.get_logical_shape();
    shard_shape[-1] = shard_shape[-1] / 24;  // width is sharded over 24 cores

    // Create core ranges
    CoreRange first_core_range(CoreCoord(1, 0), CoreCoord(2, 0));
    CoreRange second_core_range(CoreCoord(1, 4), CoreCoord(2, 4));

    // Create core range set using std::set
    std::set<CoreRange> ranges = {first_core_range, second_core_range};
    CoreRangeSet core_range_set(ranges);

    // Create shard spec with core range set and shard shape
    ShardSpec shard_spec(core_range_set, {shard_shape[0], shard_shape[1]}, ShardOrientation::ROW_MAJOR);

    // print shard spec
    std::cout << "Shard spec: " << shard_spec << std::endl;

    // Update memory config with shard spec
    output_mem_config.shard_spec = shard_spec;

    auto output_tensor_spec = TensorSpec(
        input_tensor.get_logical_shape(),
        TensorLayout(input_tensor.get_dtype(), PageConfig(Layout::TILE), output_mem_config));

    return {create_device_tensor(output_tensor_spec, input_tensor.device())};
}
operation::ProgramWithCallbacks DramPrefetcher::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& output_tensor = output_tensors.at(0);
    return dram_prefetcher_multi_core(input_tensors, global_cb, output_tensor);
}

}  // namespace ttnn::operations::dram_prefetcher
