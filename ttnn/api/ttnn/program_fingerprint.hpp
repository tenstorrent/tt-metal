// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

// What every operation call asked each device of the mesh to run, written as JSON lines when the process starts with
// TT_METAL_PROGRAM_FINGERPRINT_DIR set: one line per enqueued mesh workload in <dir>/fingerprint-<pid>.jsonl, with the
// mesh shape, the device buffers alive at the call, and per coordinate range its program
// (tt::tt_metal::experimental::program_fingerprint_json). The native migration's
// mesh scale check compares two routes of one operation with it, on mock meshes larger than the machine. Off, it
// costs one cached flag test per call.
namespace ttnn::program_fingerprint {

inline constexpr const char* ENV = "TT_METAL_PROGRAM_FINGERPRINT_DIR";

inline bool enabled() {
    static const bool on = [] {
        const char* dir = std::getenv(ENV);
        return dir != nullptr && dir[0] != '\0';
    }();
    return on;
}

void record(tt::tt_metal::distributed::MeshDevice& mesh_device, tt::tt_metal::distributed::MeshWorkload& workload);

}  // namespace ttnn::program_fingerprint
