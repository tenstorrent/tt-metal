// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/move/device/move_device_operation.hpp"
#include "ttnn/operations/data_movement/move/device/move_multi_core_with_overlap_program_factory.hpp"
#include "ttnn/operations/data_movement/move/device/move_multi_core_sharded_program_factory.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/allocator.hpp>
#include <algorithm>

#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

std::vector<CoreRange> get_multicast_regions(
    const IDevice* device, const CoreRangeSet& all_cores, const CoreCoord& logical_controller) {
    TT_ASSERT(0 < all_cores.ranges().size() and all_cores.ranges().size() <= 2);
    CoreCoord logical_zero = {0, 0};
    TT_ASSERT(logical_controller == logical_zero);

    std::vector<CoreRange> logical_core_ranges;
    auto split_core_range_containing_controller = [&](const CoreRange& controller_core_range) {
        TT_ASSERT(controller_core_range.start_coord == logical_controller);
        CoreRange right_block(
            CoreCoord(logical_controller.x + 1, logical_controller.y), controller_core_range.end_coord);
        CoreRange remaining_stick = CoreRange(
            CoreCoord(logical_controller.x, logical_controller.y + 1),
            CoreCoord(logical_controller.x, controller_core_range.end_coord.y));

        logical_core_ranges.push_back(right_block);
        logical_core_ranges.push_back(remaining_stick);
    };

    CoreRange range_0 = *all_cores.ranges().begin();
    if (all_cores.ranges().size() == 1) {
        split_core_range_containing_controller(range_0);
    } else {
        CoreRange range_1 = *all_cores.ranges().rbegin();
        if (range_0.start_coord == logical_controller) {
            split_core_range_containing_controller(range_0);
            logical_core_ranges.push_back(range_1);
        } else if (range_1.start_coord == logical_controller) {
            split_core_range_containing_controller(range_1);
            logical_core_ranges.push_back(range_0);
        } else {
            TT_THROW("Core {} is not included in set of core ranges!", logical_controller.str());
        }
    }

    TT_ASSERT(logical_core_ranges.size() == 2 or logical_core_ranges.size() == 3);
    return logical_core_ranges;
}

operation::ProgramWithCallbacks move_multi_core(const Tensor& input, Tensor& output) {
    bool src_and_dst_in_l1 = input.memory_config().is_l1() && output.memory_config().is_l1();
    return copy_multi_core(input, output, src_and_dst_in_l1);
}

}  // namespace ttnn::operations::data_movement
