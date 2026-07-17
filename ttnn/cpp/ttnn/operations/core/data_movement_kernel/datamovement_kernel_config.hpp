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
//
// disable_dfb_implicit_sync_for_all opts the kernel's DFBs out of implicit-sync credit accounting so the
// kernel's explicit reserve_back/push_back (resp. wait_front/pop_front) stays authoritative. This is a
// Gen2 (Quasar) concept only — DM kernels doing many sub-tile ("stick") NOC transfers stall the implicit
// credit accounting there; it is ignored on the Gen1 (WH/BH) placement, which has no such feature.
inline tt::tt_metal::experimental::DataMovementHardwareConfig create_reader_datamovement_config(
    tt::ARCH arch, bool disable_dfb_implicit_sync_for_all = false) {
    if (arch == tt::ARCH::QUASAR) {
        return tt::tt_metal::experimental::DataMovementGen2Config{
            .disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all};
    }
    return tt::tt_metal::experimental::CreateReader1xxDataMovementConfig();
}

inline tt::tt_metal::experimental::DataMovementHardwareConfig create_writer_datamovement_config(
    tt::ARCH arch, bool disable_dfb_implicit_sync_for_all = false) {
    if (arch == tt::ARCH::QUASAR) {
        return tt::tt_metal::experimental::DataMovementGen2Config{
            .disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all};
    }
    return tt::tt_metal::experimental::CreateWriter1xxDataMovementConfig();
}

}  // namespace ttnn
