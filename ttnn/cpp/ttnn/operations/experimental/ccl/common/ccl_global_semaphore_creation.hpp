// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/global_semaphore.hpp"

#include <vector>
#include <optional>

namespace ttnn::ccl::worker_detail {

std::vector<std::shared_ptr<const tt::tt_metal::GlobalSemaphore>> create_global_semaphores(
    const std::vector<Device*>& devices,
    const CoreRangeSet& worker_cores,
    std::optional<SubDeviceId> worker_subdevice_id_opt = std::nullopt);
}
