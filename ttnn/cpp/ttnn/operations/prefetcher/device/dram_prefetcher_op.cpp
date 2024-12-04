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
}
std::vector<ttnn::SimpleShape> DramPrefetcher::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}
std::vector<Tensor> DramPrefetcher::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}
operation::ProgramWithCallbacks DramPrefetcher::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    // return dram_prefetcher_multi_core(input_tensors, global_cb, num_receivers);
    return dram_prefetcher_multi_core(input_tensors, num_receivers);
}

}  // namespace ttnn::operations::dram_prefetcher
