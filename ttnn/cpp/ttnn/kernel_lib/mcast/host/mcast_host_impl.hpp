// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"

#include <array>
#include <tt-metalium/program.hpp>

namespace ttnn::kernel_lib::host {

// Indicates that no consumer-ready semaphore is configured.
static constexpr uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;

// The host resolves this default from each sender's multicast fanout.
static constexpr uint32_t ACK_EQUALS_FANOUT = 0xFFFFFFFFu;

// Internal lowering engine. Operation code and language bindings use Mcast.
class McastImpl {
public:
    explicit McastImpl(const tt::tt_metal::IDevice& device, const McastConfig& cfg = {});
    void add_group(
        tt::tt_metal::CoreRangeSet receivers,
        std::vector<tt::tt_metal::CoreCoord> senders,
        std::optional<uint32_t> ack_count_override = std::nullopt);

    void attach(
        tt::tt_metal::ProgramDescriptor&,
        std::string_view prefix,
        std::span<const std::reference_wrapper<tt::tt_metal::KernelDescriptor>> kernels) const;
    void attach(
        tt::tt_metal::experimental::ProgramSpec&,
        tt::tt_metal::experimental::ProgramRunArgs&,
        std::string_view prefix,
        std::span<const tt::tt_metal::experimental::KernelSpecName> kernels,
        std::span<const tt::tt_metal::experimental::SemaphoreSpecName> adopted_semaphores = {}) const;

    void append_semaphores(tt::tt_metal::Program& program);

    template <typename Args>
    void append_compile_time_args_to(Args& destination) const {
        require_program_bound_();
        detail::append_args_to(destination, compile_time_args_(program_semaphore_ids_, argument_metadata_()));
    }

    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        require_program_bound_();
        detail::append_args_to(destination, runtime_args_(core, argument_metadata_()));
    }

    McastArgumentOffsets append_kernel_args_to(
        std::vector<uint32_t>& compile_time_args,
        tt::tt_metal::KernelDescriptor::RuntimeArgs& runtime_args,
        const tt::tt_metal::CoreRangeSet& placement) const;

    const tt::tt_metal::CoreRangeSet& participating_cores() const;
    tt::tt_metal::CoreRangeSet sender_only_cores() const;

private:
    friend class Mcast;
    void prepare_topology_() const;
    void prepare_arguments_() const;
    dataflow_kernel_lib::mcast_wire::ArgumentMetadata argument_metadata_(
        const tt::tt_metal::CoreRangeSet* placement = nullptr) const;
    std::vector<uint32_t> runtime_args_(
        const tt::tt_metal::CoreCoord& core, const dataflow_kernel_lib::mcast_wire::ArgumentMetadata& metadata) const;

    struct Group {
        Group(
            tt::tt_metal::CoreRangeSet receivers,
            std::vector<tt::tt_metal::CoreCoord> senders,
            std::optional<uint32_t> ack_count_override = std::nullopt);

        const tt::tt_metal::CoreRangeSet& receiver_cores() const { return receivers_; }
        const tt::tt_metal::CoreRangeSet& participating_cores() const { return participating_; }
        std::vector<uint32_t> runtime_args(
            const tt::tt_metal::CoreCoord& core,
            const dataflow_kernel_lib::mcast_wire::ArgumentMetadata& metadata) const;

        bool rotating() const { return senders_.size() > 1; }
        uint32_t num_senders() const;
        bool has_remote_receivers() const;
        uint32_t num_rectangles() const;

        struct PreparedMulticast {
            std::vector<std::vector<uint32_t>> receiver_rectangle_args_per_sender;
            dataflow_kernel_lib::SenderMcastMode sender_mcast_mode = dataflow_kernel_lib::SenderMcastMode::Unknown;
        };
        struct PreparedChain {
            std::vector<tt::tt_metal::CoreCoord> order;
            std::vector<dataflow_kernel_lib::ChainRuntimeArguments> nodes;
        };
        struct PreparedRectangle {
            tt::tt_metal::CoreRange logical;
            tt::tt_metal::CoreRange noc;
        };
        struct PreparedState {
            std::vector<PreparedRectangle> rectangles;
            std::vector<uint32_t> sender_coords;
            dataflow_kernel_lib::mcast_wire::SenderCoordinateMetadata coordinate_metadata;
            std::vector<uint32_t> sender_ranges;
            std::vector<uint32_t> acks;
            std::variant<PreparedMulticast, PreparedChain> transport = PreparedMulticast{};
        };
        void prepare_(
            const tt::tt_metal::IDevice& device,
            const McastConfig& cfg,
            dataflow_kernel_lib::TransferMode transfer_mode,
            const tt::tt_metal::CoreRangeSet* handshake_cores) const;
        PreparedMulticast prepare_multicast_(
            const McastConfig& cfg, PreparedState& state, const tt::tt_metal::CoreRangeSet* handshake_cores) const;
        PreparedChain prepare_chain_(const tt::tt_metal::IDevice& device, PreparedState& state) const;
        const PreparedState& prepared_state_() const;
        uint32_t sender_phase_(const tt::tt_metal::CoreCoord& core) const;

        tt::tt_metal::CoreRangeSet receivers_;
        std::vector<tt::tt_metal::CoreCoord> senders_;
        std::optional<uint32_t> ack_count_override_;
        tt::tt_metal::CoreRangeSet participating_;
        std::vector<uint32_t> fanouts_;
        mutable std::optional<PreparedState> prepared_;
    };

    void require_arguments_prepared_() const;
    void require_program_bound_() const;
    void require_unbound_() const;
    std::array<uint32_t, 3> resolve_semaphore_ids_(std::span<const tt::tt_metal::SemaphoreDescriptor> existing) const;
    void validate_semaphores_present_and_zeroed_(
        std::span<const tt::tt_metal::SemaphoreDescriptor> existing, const std::array<uint32_t, 3>& ids) const;
    uint32_t required_semaphores_() const;
    std::vector<uint32_t> compile_time_args_(
        const std::array<uint32_t, 3>& ids, const dataflow_kernel_lib::mcast_wire::ArgumentMetadata& metadata) const;

    std::reference_wrapper<const tt::tt_metal::IDevice> device_;
    mutable bool arguments_prepared_ = false;
    mutable bool topology_current_ = false;
    mutable tt::ARCH prepared_arch_{};
    mutable tt::tt_metal::CoreCoord prepared_device_grid_;
    std::optional<tt::tt_metal::ProgramId> bound_program_id_;
    std::array<uint32_t, 3> program_semaphore_ids_{UNUSED_SEM_ID, UNUSED_SEM_ID, UNUSED_SEM_ID};
    const Group* group_for_core_(const tt::tt_metal::CoreCoord& core) const;
    std::vector<Group> groups_;
    McastConfig cfg_;
    mutable tt::tt_metal::CoreRangeSet receivers_;
    mutable tt::tt_metal::CoreRangeSet participating_;
    mutable dataflow_kernel_lib::mcast_wire::McastMetadata layout_;
    mutable dataflow_kernel_lib::mcast_wire::ArgumentMetadata generic_metadata_;
};

}  // namespace ttnn::kernel_lib::host
