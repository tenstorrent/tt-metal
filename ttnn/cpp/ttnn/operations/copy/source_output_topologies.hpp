// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// The output labels of an operation whose output is distributed like `source`, such as a copy or a typecast. A fresh
// output, and a preallocated one stored on the same mesh coordinates as the source, which the operation overwrites on
// every device it is stored on, take the source's label. A preallocated output stored on other mesh coordinates gets
// only part of the source or keeps its old data on some devices; {} leaves it to the framework default.
//
// The labels are applied before the launch validates the call, so `validate_call`, the operation's own validation,
// runs before the source's label replaces another label on a preallocated output: a call it rejects leaves the
// caller's tensor as it was.
template <typename ValidateCall>
std::vector<tt::tt_metal::TensorTopology> source_output_topologies(
    const Tensor& source, const std::optional<Tensor>& preallocated_output, const ValidateCall& validate_call) {
    if (preallocated_output.has_value()) {
        const auto source_coords = source.device_storage().get_coords();
        const auto output_coords = preallocated_output->device_storage().get_coords();
        if (!std::is_permutation(
                source_coords.begin(), source_coords.end(), output_coords.begin(), output_coords.end())) {
            return {};
        }
        if (preallocated_output->tensor_topology() != source.tensor_topology()) {
            validate_call();
        }
    }
    return {source.tensor_topology()};
}

}  // namespace ttnn::prim
