// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_log.h"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks transpose_cn_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_hc_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_hc_multi_core_sharded(const Tensor &a, Tensor &output); 
operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor &a, Tensor &output); 
operation::ProgramWithCallbacks transpose_wh_multi_core_sharded(const Tensor &a, Tensor &output);

} // namespace ttnn::operations::reduction::detail
