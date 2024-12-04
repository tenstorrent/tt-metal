// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher.hpp"
#include <optional>

#include "device/dram_prefetcher_op.hpp"
#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"

namespace ttnn::operations::dram_prefetcher {

ttnn::Tensor ExecuteDramPrefetcher::invoke(std::vector<ttnn::Tensor>& tensors) {
    auto arch = tensors[0].storage_type() == StorageType::DEVICE
                    ? tensors[0].device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();

    uint32_t global_cb_size = 750000;
    uint32_t num_receivers = 2;
    auto device = tensors[0].device();

    // DRAM reader cores
    CoreCoord dram_reader_core_coord = CoreCoord{0, 0};
    CoreRangeSet dram_reader_core{std::set<CoreRange>{CoreRange{dram_reader_core_coord}}};

    // L1 receiver cores
    CoreRange l1_receiver_core_coord_range = CoreRange(CoreCoord{0, 0});
    if (arch == tt::ARCH::GRAYSKULL) {
        l1_receiver_core_coord_range = CoreRange{CoreCoord{0, 1}, CoreCoord{0, num_receivers}};
    } else {
        l1_receiver_core_coord_range = CoreRange{CoreCoord{1, 0}, CoreCoord{num_receivers, 0}};
    }
    CoreRangeSet l1_receiver_core{std::set<CoreRange>{l1_receiver_core_coord_range}};

    std::unordered_map<CoreCoord, CoreRangeSet> sender_receiver_core_mapping;
    sender_receiver_core_mapping[dram_reader_core_coord] = l1_receiver_core;

    auto global_cb = tt_metal::v1::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, global_cb_size, tt_metal::BufferType::L1);

    operation::run(
        DramPrefetcher{// .global_cb = std::make_shared<tt_metal::v1::experimental::GlobalCircularBuffer>(global_cb),
                       .num_receivers = num_receivers},
        {tensors},
        {});
    return tensors[0];
}

}  // namespace ttnn::operations::dram_prefetcher
