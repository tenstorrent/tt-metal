// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

// start is inclusive, end is exclusive
struct PageRange {
    uint32_t start;
    uint32_t end;
};

struct CorePageRange {
    CoreCoord core;
    PageRange range;
};

tt::tt_metal::operation::ProgramWithCallbacks s2s_rm_concat_two_tensors_multi_core(
    const std::vector<Tensor>& input_tensors, uint32_t dim, Tensor& output, unsigned int groups = 1);

tt::tt_metal::operation::ProgramWithCallbacks s2i_rm_concat_multi_core(
    const std::vector<Tensor>& input_tensors, uint32_t dim, Tensor& output);

tt::tt_metal::operation::ProgramWithCallbacks sharded_concat_multi_core(
    const std::vector<Tensor>& input_tensors, uint32_t dim, Tensor& output, unsigned int groups = 1);

tt::tt_metal::operation::ProgramWithCallbacks concat_multi_core(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const Tensor& output);

}  // namespace ttnn::operations::data_movement::detail
