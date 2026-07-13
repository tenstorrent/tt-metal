// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/bh_dram_read_device_operation.hpp"

namespace ttnn {

// bh_dram_read: a minimal, read-only op. Places one worker core per DRAM bank;
// each core reads the input tensor's pages that live in its assigned bank and
// discards them. Returns nothing.
void bh_dram_read(const Tensor& input_tensor);

}  // namespace ttnn
