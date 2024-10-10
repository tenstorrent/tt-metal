// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include "tt_metal/common/work_split.hpp"

namespace ttnn::operations::experimental::paged_cache::detail {

operation::ProgramWithCallbacks paged_fill_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, const Tensor &page_table_tensor, const uint32_t batch_idx);
}  // ttnn::operations::experimental::paged_cache::detail
