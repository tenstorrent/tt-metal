// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host_unified.hpp"

#include <algorithm>
#include <tuple>
#include <tt_stl/assert.hpp>

namespace ttnn::kernel_lib::host {
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::IDevice;

namespace {
McastConfig family_config(const McastUnifiedConfig& config) {
    return {
        .noc = config.noc,
        .handshake = config.handshake,
        .data_ready = config.data_ready,
        .base_sem_id = config.base_sem_id,
        .sem_ids = config.sem_ids,
        .irregular_receiver_set_mode = config.irregular_receiver_set_mode};
}

std::vector<CoreCoord> ordered_cores(const CoreRangeSet& cores, McastCoreOrder order) {
    TT_FATAL(order == McastCoreOrder::RowMajor || order == McastCoreOrder::ColumnMajor, "Mcast: invalid core order");
    auto result = tt::tt_metal::corerange_to_cores(cores);
    std::sort(result.begin(), result.end(), [order](const CoreCoord& a, const CoreCoord& b) {
        return order == McastCoreOrder::RowMajor ? std::tie(a.y, a.x) < std::tie(b.y, b.x)
                                                 : std::tie(a.x, a.y) < std::tie(b.x, b.y);
    });
    return result;
}
}  // namespace

Mcast::Mcast(
    IDevice* device,
    const McastUnifiedConfig& config,
    const CoreRangeSet& receivers,
    uint32_t receiver_group_size,
    McastCoreOrder receiver_order,
    const McastSenderConfig& sender_config) :
    family_(
        device,
        family_config(config),
        config.handshake_cores ? std::make_optional(*config.handshake_cores) : std::nullopt) {
    // CoreRangeSet already guarantees unique coordinates.
    const auto receiver_cores = ordered_cores(receivers, receiver_order);
    TT_FATAL(!receivers.empty(), "Mcast: receivers must not be empty");
    TT_FATAL(receiver_group_size > 0, "Mcast: receiver_group_size must be positive");
    TT_FATAL(
        receiver_cores.size() % receiver_group_size == 0,
        "Mcast: receiver groups must divide the receiver list exactly");
    const size_t num_groups = receiver_cores.size() / receiver_group_size;
    const auto* fixed = std::get_if<McastFixedSenderConfig>(&sender_config);
    const auto* grid = std::get_if<McastSenderGridConfig>(&sender_config);
    const auto* explicit_senders = std::get_if<McastExplicitSenderConfig>(&sender_config);
    std::vector<CoreCoord> grid_senders;
    if (fixed) {
        TT_FATAL(
            fixed->placement == McastSenderPlacement::Uniform || fixed->placement == McastSenderPlacement::Staggered,
            "Mcast: invalid fixed sender placement");
        TT_FATAL(
            fixed->placement == McastSenderPlacement::Staggered || fixed->sender_index < receiver_group_size,
            "Mcast: uniform sender_index is outside its receiver group");
    } else if (grid) {
        grid_senders = ordered_cores(grid->sender_cores, grid->sender_order.value_or(receiver_order));
        TT_FATAL(!grid_senders.empty(), "Mcast: sender grid must not be empty");
        TT_FATAL(grid_senders.size() % num_groups == 0, "Mcast: sender grid must divide evenly across receiver groups");
    } else if (explicit_senders) {
        TT_FATAL(
            explicit_senders->senders_per_group.size() == num_groups,
            "Mcast: explicit sender lists must match the receiver group count");
    }

    for (size_t group_index = 0; group_index < num_groups; ++group_index) {
        const auto begin = receiver_cores.begin() + group_index * receiver_group_size;
        const auto end = begin + receiver_group_size;
        std::vector<CoreRange> ranges;
        ranges.reserve(receiver_group_size);
        for (auto it = begin; it != end; ++it) {
            ranges.emplace_back(*it, *it);
        }
        std::vector<CoreCoord> senders;
        if (fixed) {
            const size_t index = fixed->placement == McastSenderPlacement::Staggered
                                     ? (uint64_t(fixed->sender_index) + group_index) % receiver_group_size
                                     : fixed->sender_index;
            senders.push_back(*(begin + index));
        } else if (grid) {
            const size_t count = grid_senders.size() / num_groups;
            const auto first = grid_senders.begin() + group_index * count;
            senders.assign(first, first + count);
        } else if (explicit_senders) {
            senders = explicit_senders->senders_per_group[group_index];
        } else {
            senders.assign(begin, end);
        }
        // The family owns schedule uniqueness/length and cross-group footprint
        // validation. Do not replace either receiver membership or sender order.
        family_.add_group(CoreRangeSet(std::move(ranges)), std::move(senders));
    }
    family_.prepare_arguments_();
}

}  // namespace ttnn::kernel_lib::host
