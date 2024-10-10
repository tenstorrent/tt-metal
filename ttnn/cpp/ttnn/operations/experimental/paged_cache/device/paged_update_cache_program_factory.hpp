// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "ttnn/operations/moreh/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include "tt_metal/common/work_split.hpp"

namespace ttnn::operations::experimental::paged_cache::detail {

operation::ProgramWithCallbacks paged_update_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, std::optional<const Tensor> update_idxs_tensor, std::optional<const Tensor> page_table, const std::vector<uint32_t> update_idxs, const uint32_t batch_offset, ttnn::DeviceComputeKernelConfig compute_kernel_config, const bool share_cache);
}  // ttnn::operations::experimental::paged_cache::detail
