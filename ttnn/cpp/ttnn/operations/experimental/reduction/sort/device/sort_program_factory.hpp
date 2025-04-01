// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::reduction::detail {

operation::ProgramWithCallbacks sort_program_interleaved(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool descending,
    const bool stable,
    Tensor& value_tensor,
    Tensor& index_tensor) {
    tt::tt_metal::Program program{};

    // TODO: Implementation in next PR
    tt::log_warning("sort_program_interleaved not implemented yet!");

    return {std::move(program), {}};
}

}  // namespace ttnn::operations::experimental::reduction::detail
