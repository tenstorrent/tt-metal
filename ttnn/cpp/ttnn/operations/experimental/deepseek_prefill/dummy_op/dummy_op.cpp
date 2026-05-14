// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dummy_op.hpp"
#include "device/dummy_op_device_operation.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dummy_op {

ttnn::Tensor dummy_op(
    const ttnn::Tensor& input_tensor, uint32_t num_iter, const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {
    auto* mesh_device = input_tensor.device();

    // If a subdevice_id is given, use its worker cores. Otherwise fall back to
    // the historical default: row 0 of the compute grid.
    CoreRangeSet worker_core_range_set;
    if (subdevice_id.has_value()) {
        worker_core_range_set =
            mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id.value());
    } else {
        const auto grid = mesh_device->compute_with_storage_grid_size();
        worker_core_range_set = CoreRangeSet{CoreRange{{0, 0}, {grid.x - 1, 0}}};
    }

    return ttnn::prim::prefill_dummy_op(input_tensor, num_iter, worker_core_range_set);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dummy_op
