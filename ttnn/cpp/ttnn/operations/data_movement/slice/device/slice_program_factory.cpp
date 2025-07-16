// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_program_factory.hpp"
#include "slice_rm_multi_core.hpp"
#include "slice_rm_strided_single_core_n_dims.hpp"
#include "slice_rm_multi_core_sharded.hpp"
#include "slice_tile_multi_core.hpp"
#include "slice_op.hpp"
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks slice_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step) {
    bool has_step = false;
    for (int i = 0; i < step.size(); i++) {
        if (step[i] != 1) {
            has_step = true;
            break;
        }
    }
    switch (a.layout()) {
        case Layout::ROW_MAJOR:
            return a.is_sharded() ? slice_rm_multi_core_sharded(a, output, output_tensor_start, output_tensor_end)
                                  : (has_step ? slice_rm_strided_single_core_n_dims(
                                                    a, output, output_tensor_start, output_tensor_end, step)
                                              : slice_rm_multi_core(a, output, output_tensor_start, output_tensor_end));
        case Layout::TILE: return slice_tile_multi_core(a, output, output_tensor_start, output_tensor_end);
        default: TT_ASSERT(false, "Unsupported Layout");
    }
    return {};
}

}  // namespace ttnn::operations::data_movement::detail
