// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host_impl.hpp"

#include <algorithm>
#include <tuple>
#include <utility>
#include <tt_stl/assert.hpp>

namespace ttnn::kernel_lib::host {
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::IDevice;

namespace {
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
    const IDevice& device,
    const McastConfig& config,
    const CoreRangeSet& receivers,
    uint32_t receiver_group_size,
    const McastSenderConfig& sender_config,
    McastCoreOrder receiver_order) :
    impl_(std::make_unique<McastImpl>(device, config)) {
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
        // The implementation owns schedule uniqueness/length and cross-group footprint
        // validation. Do not replace either receiver membership or sender order.
        impl_->add_group(CoreRangeSet(std::move(ranges)), std::move(senders));
    }
    impl_->prepare_arguments_();
}

Mcast::~Mcast() = default;

Mcast::Mcast(const Mcast& other) : impl_(other.impl_ ? std::make_unique<McastImpl>(*other.impl_) : nullptr) {}

Mcast& Mcast::operator=(const Mcast& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<McastImpl>(*other.impl_) : nullptr;
    }
    return *this;
}

Mcast::Mcast(Mcast&&) noexcept = default;

Mcast& Mcast::operator=(Mcast&&) noexcept = default;

void Mcast::attach(
    tt::tt_metal::ProgramDescriptor& descriptor,
    std::string_view prefix,
    std::span<const std::reference_wrapper<tt::tt_metal::KernelDescriptor>> kernels,
    uint32_t first_semaphore_id) const {
    impl_->attach(descriptor, prefix, kernels, first_semaphore_id);
}

uint32_t Mcast::next_semaphore_id() const { return impl_->next_semaphore_id(); }

void Mcast::attach(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::ProgramRunArgs& args,
    std::string_view prefix,
    std::span<const tt::tt_metal::experimental::KernelSpecName> kernels) const {
    impl_->attach(spec, args, prefix, kernels);
}

void Mcast::append_semaphores(tt::tt_metal::Program& program) { impl_->append_semaphores(program); }

std::vector<uint32_t> Mcast::compile_time_args_() const {
    impl_->require_program_bound_();
    return impl_->compile_time_args_(impl_->program_semaphore_ids_, impl_->argument_metadata_());
}

std::vector<uint32_t> Mcast::runtime_args_(const CoreCoord& core) const {
    impl_->require_program_bound_();
    return impl_->runtime_args_(core, impl_->argument_metadata_());
}

McastArgumentOffsets Mcast::append_kernel_args_to(
    std::vector<uint32_t>& compile_time_args,
    tt::tt_metal::KernelDescriptor::RuntimeArgs& runtime_args,
    const CoreRangeSet& placement) const {
    return impl_->append_kernel_args_to(compile_time_args, runtime_args, placement);
}

const CoreRangeSet& Mcast::participating_cores() const { return impl_->participating_cores(); }

CoreRangeSet Mcast::sender_only_cores() const { return impl_->sender_only_cores(); }

}  // namespace ttnn::kernel_lib::host
