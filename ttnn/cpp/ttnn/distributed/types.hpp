// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides type aliases for distributed computing components used in the TTNN library.
// It imports and renames types from the tt_metal library to maintain a consistent naming convention
// within the TTNN namespace while leveraging the underlying tt_metal functionality.

#include "tt_metal/distributed/mesh_device.hpp"

namespace ttnn::distributed {

using MeshShape = tt::tt_metal::distributed::MeshShape;
using DeviceIds = tt::tt_metal::distributed::DeviceIds;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using MeshDeviceView = tt::tt_metal::distributed::MeshDeviceView;
using MeshType = tt::tt_metal::distributed::MeshType;
using MeshDeviceConfig = tt::tt_metal::distributed::MeshDeviceConfig;

}  // namespace ttnn::distributed

namespace ttnn {

// These types are exported to the ttnn namespace for convenience.
using ttnn::distributed::DeviceIds;
using ttnn::distributed::MeshDevice;
using ttnn::distributed::MeshDeviceConfig;
using ttnn::distributed::MeshDeviceView;
using ttnn::distributed::MeshShape;
using ttnn::distributed::MeshType;

}  // namespace ttnn
