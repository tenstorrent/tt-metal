// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct WidthShardedAllReduceParams {
    uint32_t num_links = 1;
    uint32_t ring_size = 0;
    std::optional<uint32_t> cluster_axis;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;

    static constexpr auto attribute_names =
        std::forward_as_tuple("num_links", "ring_size", "cluster_axis", "sub_device_id", "topology");
    auto attribute_values() const {
        return std::make_tuple(num_links, ring_size, cluster_axis, sub_device_id, topology);
    }
};

struct WidthShardedAllReduceInputs {
    Tensor input;
};

}  // namespace ttnn::prim
