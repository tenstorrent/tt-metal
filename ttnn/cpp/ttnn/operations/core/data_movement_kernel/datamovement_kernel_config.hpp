// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>

namespace ttnn {

// Generation-agnostic construction of a Metal 2.0 DataMovementHardwareConfig.
//
// DataMovementHardwareConfig is a variant over a Gen1 (WH/BH) and a Gen2 (Quasar) config, and a kernel
// must carry the alternative matching its target architecture. These helpers pick that alternative from
// `arch` so architecture-agnostic host code doesn't have to branch at every kernel spec.

// The conventional reader / writer DM placement, generation-agnostic. On Gen1 these forward to the
// metal create_reader/writer_gen1_datamovement_config() placement; on Quasar both yield a default
// DataMovementGen2Config.
inline tt::tt_metal::experimental::DataMovementHardwareConfig create_reader_datamovement_config(tt::ARCH arch) {
    if (arch == tt::ARCH::QUASAR) {
        return tt::tt_metal::experimental::DataMovementGen2Config{};
    }
    return tt::tt_metal::experimental::CreateReader1xxDataMovementConfig();
}

inline tt::tt_metal::experimental::DataMovementHardwareConfig create_writer_datamovement_config(tt::ARCH arch) {
    if (arch == tt::ARCH::QUASAR) {
        return tt::tt_metal::experimental::DataMovementGen2Config{};
    }
    return tt::tt_metal::experimental::CreateWriter1xxDataMovementConfig();
}

}  // namespace ttnn
