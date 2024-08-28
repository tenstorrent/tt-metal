// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace tt;

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

operation::ProgramWithCallbacks s2s_rm_concat_two_tensors_multi_core(
    const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output);

operation::ProgramWithCallbacks s2i_rm_concat_multi_core(
    const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output);

operation::ProgramWithCallbacks sharded_concat_multi_core(
    const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output);

operation::ProgramWithCallbacks concat_multi_core(
    const std::vector<Tensor> &input_tensors, const uint32_t dim, const Tensor &output);

}  // namespace ttnn::operations::data_movement::detail
