// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::generic {

using OptionalAddr = std::optional<std::uint32_t>;

/// Collect buffer addresses from IO tensors (nullopt for tensors without a device buffer).
inline std::vector<OptionalAddr> collect_io_tensor_addresses(const std::vector<Tensor>& io_tensors) {
    std::vector<OptionalAddr> addrs;
    addrs.reserve(io_tensors.size());
    for (const auto& t : io_tensors) {
        auto* buf = t.buffer();
        addrs.push_back(buf ? std::optional{buf->address()} : std::nullopt);
    }
    return addrs;
}

/// Find the first IO tensor whose address matches *value*, or nullopt.
inline std::optional<std::uint32_t> find_io_tensor_index(std::uint32_t value, const std::vector<OptionalAddr>& addrs) {
    for (size_t i = 0; i < addrs.size(); ++i) {
        if (addrs[i].has_value() && addrs[i].value() == value) {
            return static_cast<std::uint32_t>(i);
        }
    }
    return std::nullopt;
}

/// For each CBDescriptor with a buffer, match its address against *tensor_addrs*
/// and return (cb_idx, io_tensor_idx) pairs.  Must be called while buffer pointers
/// are still valid.
inline std::vector<std::pair<uint32_t, uint32_t>> compute_cb_io_tensor_map(
    const tt::tt_metal::ProgramDescriptor& desc, const std::vector<OptionalAddr>& tensor_addrs) {
    std::vector<std::pair<uint32_t, uint32_t>> result;
    for (size_t ci = 0; ci < desc.cbs.size(); ++ci) {
        const auto* buf = desc.cbs[ci].buffer;
        if (buf != nullptr) {
            if (auto ti = find_io_tensor_index(buf->address(), tensor_addrs)) {
                result.emplace_back(static_cast<uint32_t>(ci), *ti);
            }
        }
    }
    return result;
}

}  // namespace ttnn::operations::experimental::generic
