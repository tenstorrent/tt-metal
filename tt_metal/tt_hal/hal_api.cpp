// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "tt_metal/tt_hal/hal_api.hpp"
#include "tt_metal/llrt/hal.hpp"

using tt::tt_metal::HalL1MemAddrType;
using tt::tt_metal::HalProgrammableCoreType;
using tt::tt_metal::HalSingleton;

namespace tt::tt_metal::tt_hal {

uint32_t get_l1_size() {
    return HalSingleton::getInstance().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
}

}  // namespace tt::tt_metal::tt_hal
