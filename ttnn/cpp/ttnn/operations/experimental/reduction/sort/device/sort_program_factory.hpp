// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_log.h>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::reduction::detail {

operation::ProgramWithCallbacks sort_single_core_interleaved(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool descending,
    const bool stable,
    Tensor& value_tensor,
    Tensor& index_tensor) {
    tt::tt_metal::Program program{};

    return {std::move(program), {}};
}

operation::ProgramWithCallbacks sort_multi_core_interleaved(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool descending,
    const bool stable,
    Tensor& value_tensor,
    Tensor& index_tensor) {
    tt::tt_metal::Program program{};

    return {std::move(program), {}};
}

}  // namespace ttnn::operations::experimental::reduction::detail
