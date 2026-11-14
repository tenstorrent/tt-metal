// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Sum a ROW_MAJOR WIDTH_SHARDED tensor across the mesh (or along ``cluster_axis``).
// Output layout and memory config match the input (same shard spec).
// `max_packet_bytes`: largest payload of one fabric packet. Each reducer core receives one packet per
// chip, so a larger cap sends fewer packets to fewer reducer cores, each summing more faces, at a
// ping-pong scratch of 2 * ring_size * packet bytes per reducer core. Past the default cap that scratch
// only comes out of L1_SMALL, leaving the cap's worth free; packets it cannot hold fall back to smaller
// ones. Unset keeps the scratch inside the cap (2560 B).
ttnn::Tensor width_sharded_all_reduce(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
    std::optional<uint32_t> max_packet_bytes = std::nullopt);

}  // namespace ttnn::experimental::deepseek
