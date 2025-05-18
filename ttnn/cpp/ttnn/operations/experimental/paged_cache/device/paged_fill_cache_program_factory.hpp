// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::paged_cache::detail {

tt::tt_metal::operation::ProgramWithCallbacks paged_fill_cache_multi_core(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const Tensor& page_table_tensor,
    std::optional<const Tensor> batch_idx_tensor,
    const uint32_t batch_idx_fallback);
}  // namespace ttnn::operations::experimental::paged_cache::detail
