// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides type aliases for distributed computing components used in the TTNN library.
// It imports and renames types from the tt_metal library to maintain a consistent naming convention
// within the TTNN namespace while leveraging the underlying tt_metal functionality.

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

namespace ttnn::distributed {

using MeshShape = tt::tt_metal::distributed::MeshShape;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using MeshCoordinateRange = tt::tt_metal::distributed::MeshCoordinateRange;
using MeshCoordinateRangeSet = tt::tt_metal::distributed::MeshCoordinateRangeSet;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using SystemMesh = tt::tt_metal::distributed::SystemMesh;
using MeshDeviceView = tt::tt_metal::distributed::MeshDeviceView;
using MeshDeviceConfig = tt::tt_metal::distributed::MeshDeviceConfig;

}  // namespace ttnn::distributed

namespace ttnn {

// These types are exported to the ttnn namespace for convenience.
using ttnn::distributed::MeshCoordinate;
using ttnn::distributed::MeshCoordinateRange;
using ttnn::distributed::MeshCoordinateRangeSet;
using ttnn::distributed::MeshDevice;
using ttnn::distributed::MeshDeviceConfig;
using ttnn::distributed::MeshDeviceView;
using ttnn::distributed::MeshShape;
using ttnn::distributed::SystemMesh;

}  // namespace ttnn
