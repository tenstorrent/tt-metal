// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt{
    namespace tt_metal{
        Tensor read_tensor_from_L1(
            uint64_t addr,
            CoreCoord core,
            uint64_t size,
            DataType dtype,
            Device* device
        );
        void print_tensor_info(Tensor &tensor);

    }
}
