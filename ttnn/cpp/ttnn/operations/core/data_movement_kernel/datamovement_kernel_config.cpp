// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "datamovement_kernel_config.hpp"

#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>

namespace ttnn {

tt::tt_metal::experimental::DataMovementHardwareConfig create_reader_datamovement_config(tt::ARCH arch) {
    if (arch == tt::ARCH::QUASAR) {
        return tt::tt_metal::experimental::DataMovementGen2Config{};
    }
    return tt::tt_metal::experimental::create_reader_gen1_datamovement_config();
}

tt::tt_metal::experimental::DataMovementHardwareConfig create_writer_datamovement_config(tt::ARCH arch) {
    if (arch == tt::ARCH::QUASAR) {
        return tt::tt_metal::experimental::DataMovementGen2Config{};
    }
    return tt::tt_metal::experimental::create_writer_gen1_datamovement_config();
}

}  // namespace ttnn
