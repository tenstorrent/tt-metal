// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/detail/tt_metal.hpp"
#include "types.hpp"

namespace ttnn::device {

using Device = ttnn::Device;

namespace device_pool {

extern std::vector<Device *> devices;

} // namespace device_pool

} // namespace ttnn::device
