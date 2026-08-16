// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>

namespace ttnn {

// Generation-agnostic construction of a Metal 2.0 DataMovementHardwareConfig.
//
// A DataMovementHardwareConfig carries a settings block per generation and only the block matching the
// target architecture is applied. These helpers populate both, so architecture-agnostic host code can
// build one kernel spec that runs on either generation.

// The conventional reader / writer DM placement. The Gen1 block gets the metal
// CreateReader/WriterGen1DataMovementConfig() placement; the Gen2 block, which has no placement to
// choose, gets the implicit-sync setting below.
//
// disable_dfb_implicit_sync_for_all opts the kernel's DFBs out of implicit-sync credit accounting so the
// kernel's explicit reserve_back/push_back (resp. wait_front/pop_front) stays authoritative. This is a
// Gen2 (Quasar) concept only — DM kernels doing many sub-tile ("stick") NOC transfers stall the implicit
// credit accounting there; it is inert on Gen1 (WH/BH), which has no such feature.
inline tt::tt_metal::experimental::DataMovementHardwareConfig create_reader_datamovement_config(
    bool disable_dfb_implicit_sync_for_all = false) {
    return tt::tt_metal::experimental::DataMovementHardwareConfig{
        .gen1 = tt::tt_metal::experimental::CreateReaderGen1DataMovementConfig(),
        .gen2 = {.disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all},
    };
}

inline tt::tt_metal::experimental::DataMovementHardwareConfig create_writer_datamovement_config(
    bool disable_dfb_implicit_sync_for_all = false) {
    return tt::tt_metal::experimental::DataMovementHardwareConfig{
        .gen1 = tt::tt_metal::experimental::CreateWriterGen1DataMovementConfig(),
        .gen2 = {.disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all},
    };
}

}  // namespace ttnn
