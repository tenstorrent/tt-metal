// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "datamovement_kernel_config.hpp"

#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>

namespace ttnn {

tt::tt_metal::experimental::DataMovementHardwareConfig to_datamovement_hardware_config(
    tt::ARCH arch, const tt::tt_metal::experimental::DataMovementGen1Config& gen1_config) {
    if (arch == tt::ARCH::QUASAR) {
        return tt::tt_metal::experimental::DataMovementGen2Config{};
    }
    return gen1_config;
}

tt::tt_metal::experimental::DataMovementHardwareConfig create_reader_datamovement_config(tt::ARCH arch) {
    return to_datamovement_hardware_config(arch, tt::tt_metal::experimental::create_reader_gen1_datamovement_config());
}

tt::tt_metal::experimental::DataMovementHardwareConfig create_writer_datamovement_config(tt::ARCH arch) {
    return to_datamovement_hardware_config(arch, tt::tt_metal::experimental::create_writer_gen1_datamovement_config());
}

}  // namespace ttnn
