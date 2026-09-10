// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>

namespace ttnn {

// Generation-agnostic construction of a Metal 2.0 DataMovementHardwareConfig.
//
// Calls the Metal reader/writer factory (which fills config_1xx pins) and layers the
// 2xx implicit-sync opt-out on top so architecture-agnostic host code doesn't have to
// branch at every kernel spec.

// The conventional reader / writer DM placement. On 1xx these use the Metal factory's
// RISC/NOC pins; on 2xx those extras are unused and the implicit-sync flag applies.
//
// disable_dfb_implicit_sync_for_all opts the kernel's DFBs out of implicit-sync credit accounting so the
// kernel's explicit reserve_back/push_back (resp. wait_front/pop_front) stays authoritative. This is a
// 2xx (Quasar) concept only — DM kernels doing many sub-tile ("stick") NOC transfers stall the implicit
// credit accounting there; it is ignored on the 1xx (WH/BH) placement, which has no such feature.
inline tt::tt_metal::experimental::DataMovementHardwareConfig create_reader_datamovement_config(
    bool disable_dfb_implicit_sync_for_all = false) {
    auto hw = tt::tt_metal::experimental::CreateReaderDataMovementConfig();
    hw.config_2xx = tt::tt_metal::experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
        .disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all};
    return hw;
}

inline tt::tt_metal::experimental::DataMovementHardwareConfig create_writer_datamovement_config(
    bool disable_dfb_implicit_sync_for_all = false) {
    auto hw = tt::tt_metal::experimental::CreateWriterDataMovementConfig();
    hw.config_2xx = tt::tt_metal::experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
        .disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all};
    return hw;
}

}  // namespace ttnn
