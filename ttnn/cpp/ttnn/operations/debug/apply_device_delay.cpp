// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "apply_device_delay.hpp"
#include "device/apply_device_delay_device_operation.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::operations::debug {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

void apply_device_delay(
    ttnn::MeshDevice& mesh_device,
    const std::vector<std::vector<uint32_t>>& delays,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {
    SubDeviceId sd = subdevice_id.value_or(mesh_device.get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd);

    log_info(tt::LogAlways, "Starting delay primitive");
    ttnn::prim::apply_device_delay(mesh_device, delays, subdevice_core_range_set);
    log_info(tt::LogAlways, "Ending delay primitive");
}

}  // namespace ttnn::operations::debug
