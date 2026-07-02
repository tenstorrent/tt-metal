// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bh_dram_read.hpp"

namespace ttnn {

void bh_dram_read(const Tensor& input_tensor) {
    // The primitive returns the input tensor aliased unchanged (read-only op);
    // discard it to present a void API.
    ttnn::prim::bh_dram_read(input_tensor);
}

}  // namespace ttnn
