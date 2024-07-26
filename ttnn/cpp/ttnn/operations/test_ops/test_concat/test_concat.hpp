// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/concat_device_operation.hpp"

namespace ttnn {
namespace operations::test_ops {

struct ExecuteConcat {
    static Tensor execute_on_worker_thread (const std::vector<Tensor> &input_tensors, uint8_t dim) {
        uint8_t queue_id = 0;
        return ttnn::device_operation::run<Concat>(
            queue_id,
            Concat::operation_attributes_t{dim},
            Concat::tensor_args_t{input_tensors}
        );
    }
};

}

constexpr auto test_concat = ttnn::register_operation<operations::test_ops::Concat>("ttnn::test_concat");

}
