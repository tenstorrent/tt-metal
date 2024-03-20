// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/detail/tt_metal.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device_pool.hpp"

namespace ttnn {

namespace device {

Device &open_device(int device_id);
void close_device(Device &device);
void enable_program_cache(Device &device);
void disable_and_clear_program_cache(Device &device);

}  // namespace device

using namespace device;

}  // namespace ttnn

namespace tt {
namespace tt_metal {
    using DeviceMesh = ttnn::multi_device::DeviceMesh;
} // namespace tt_metal
} // namespace tt
