// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC mode of the flat expert (MIMO_FL_DRISC, kernels/se_drisc.hpp): the persistent gate/up senders on the
// Blackhole programmable DRAM cores. They run outside the flat programs (DRAM-core kernels launch by slow dispatch,
// next to fast dispatch, as the tensor prefetcher's), launched once per mesh device and left running until
// flat_drisc_stop.

#pragma once

#include <cstdint>
#include <utility>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {
class IDevice;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace ttnn::operations::bringup::flat_routed_expert {

bool flat_drisc_enabled();
// working region of the kernels' L1 (se_drisc.hpp offsets are from its base)
uint32_t flat_drisc_base();
uint32_t flat_drisc_size();
// MIMO_FL_DRISC_TILES (1, 2, or 3 = switch): DRISC tiles used per bank
uint32_t flat_drisc_tiles();
// MIMO_FL_DRISC_PURE: every slot on DRISC (switch tiles; reader cells keep only reader down columns)
bool flat_drisc_pure();
// switch mode: slot r's alternate {tile, job} (job ~0u: none)
std::pair<tt::tt_metal::CoreCoord, uint32_t> flat_drisc_alt(tt::tt_metal::IDevice* device, uint32_t r, uint32_t banks);
// reader slot r's DRISC: in bank r % banks (where its weights are); with one tile per bank the tile that is no endpoint
// (both NIUs in stream mode) serves both of the bank's slots (job r / banks), with two tile 1 + r / banks (index 0 is
// the bank's NOC0 endpoint, owned by firmware)
tt::tt_metal::CoreCoord flat_drisc_logical(tt::tt_metal::IDevice* device, uint32_t r, uint32_t banks);
// the per-call GDDR MPFE weight word (se_drisc.hpp SD_MPFE) for a plan (with reader down columns or not)
uint32_t flat_drisc_mpfe(bool reader_down);
// the job index of slot r on its DRISC
uint32_t flat_drisc_job(uint32_t r, uint32_t banks);
// NOC0 (physical) coordinates of that tile / of a worker core
tt::tt_metal::CoreCoord flat_drisc_physical(tt::tt_metal::IDevice* device, uint32_t r, uint32_t banks);
// the NoC slot r's sender writes on (0 or 1)
uint32_t flat_drisc_noc(tt::tt_metal::IDevice* device, uint32_t r, uint32_t banks, bool reader_down);
tt::tt_metal::CoreCoord flat_worker_noc0(tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord& logical);
// launches the senders of slots 0..n_slots on every chip (once; no-op while they run)
void flat_drisc_ensure(
    tt::tt_metal::distributed::MeshDevice* mesh, uint32_t n_slots, uint32_t banks, bool reader_down);
// logs every sender's words (job counters, grants, progress)
void flat_drisc_dump(tt::tt_metal::distributed::MeshDevice* mesh);
// stops them (no-op when not running)
void flat_drisc_stop(tt::tt_metal::distributed::MeshDevice* mesh);

}  // namespace ttnn::operations::bringup::flat_routed_expert
