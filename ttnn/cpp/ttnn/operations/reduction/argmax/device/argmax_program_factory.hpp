// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction::detail {

using namespace tt::constants;

operation::ProgramWithCallbacks argmax_single_core(
    const Tensor &input, const Tensor &output, const std::optional<uint32_t> dim);

}
