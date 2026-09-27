// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_common.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>

namespace tt::tt_metal::experimental {
struct ProgramSpec;
struct ProgramRunArgs;
}  // namespace tt::tt_metal::experimental

namespace ttnn::kernel_lib::host {

// =============================================================================
// Usage examples
//
// The examples assume that device, descriptor, noc, and the shown cores and receiver sets already exist, and
// that kernel is a placed KernelDescriptor with its operation-specific compile-time and runtime arguments.
//
// Mcast1D: create one independent multicast per row, with the first core in each row as its fixed sender.
//
// Mcast1D mcast(
//     device,
//     receivers,
//     Mcast1DShape::PerRow,
//     Mcast1DFixedSenderConfig{.starting_sender_index = 0},
//     McastConfig{.noc = noc});
// const std::array kernels{std::ref(kernel)};
// mcast.attach(descriptor, "input_mcast", kernels);
// descriptor.kernels.push_back(std::move(kernel));
//
// Mcast2D: create one multicast over the receiver rectangle from a fixed sender.
//
// Mcast2D mcast(device, receivers, Mcast2DFixedSenderConfig{.sender = sender}, McastConfig{.noc = noc});
// const std::array kernels{std::ref(kernel)};
// mcast.attach(descriptor, "input_mcast", kernels);
// descriptor.kernels.push_back(std::move(kernel));
//
// McastFamily: collect independent, disjoint groups that share one protocol and semaphore allocation.
//
// McastFamily mcast(device, McastConfig{.noc = noc});
// mcast.add_group(receivers_a, std::vector<tt::tt_metal::CoreCoord>{sender_a});
// mcast.add_group(receivers_b, std::vector<tt::tt_metal::CoreCoord>{sender_b});
// mcast.prepare_arguments();
// const std::array kernels{std::ref(kernel)};
// mcast.attach(descriptor, "input_mcast", kernels);
// descriptor.kernels.push_back(std::move(kernel));
//
// These examples use ProgramDescriptor. The same helpers also support ProgramSpec attachment and direct Program
// construction through append_semaphores(), append_compile_time_args_to(), and append_runtime_args_to().
// =============================================================================

// Indicates that no consumer-ready semaphore is configured.
static constexpr uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;

// The host resolves this default from each sender's multicast fanout.
static constexpr uint32_t ACK_EQUALS_FANOUT = 0xFFFFFFFFu;

struct McastConfig {
    // NoC used by the kernel pipe.
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    // Wait for receiver readiness before sending.
    bool handshake = true;
    // Select the data-ready signaling mode.
    dataflow_kernel_lib::DataReadySignal data_ready = dataflow_kernel_lib::DataReadySignal::Flag;
    // Exact first owned semaphore ID. Descriptor attachment and Program binding
    // allocate free IDs when omitted. Direct Program construction checks this against
    // the IDs returned by CreateSemaphore; it cannot reserve an arbitrary ID.
    std::optional<uint32_t> base_sem_id = std::nullopt;
    // Adopt caller-owned ids: data_ready, consumer_ready (required with handshake), and
    // signal_source (required for resolved ChainUnicast). Chain IDs must be distinct and
    // initialized to zero for a fresh invocation; signal_source cannot alias another live channel.
    std::optional<std::vector<uint32_t>> sem_ids = std::nullopt;
    // Override the derived receiver acknowledgment count. Used for cores without actual work
    // that are passively participating in the multicast.
    std::optional<uint32_t> ack_count_override = std::nullopt;
    // Family-wide delivery policy when any group is irregular. Entirely rectangular
    // families always use Multicast, regardless of this setting.
    dataflow_kernel_lib::TransferMode irregular_receiver_set_mode = dataflow_kernel_lib::TransferMode::Multicast;
};

void attach_absent(tt::tt_metal::KernelDescriptor& kernel, std::string_view prefix);
void attach_absent(
    tt::tt_metal::experimental::ProgramSpec& spec,
    std::string_view prefix,
    std::span<const tt::tt_metal::experimental::KernelSpecName> targets);

namespace detail {

// Compile-time representation of an absent multicast channel. It emits only
// the false presence tag and therefore has no runtime payload or semaphores.
std::vector<uint32_t> absent_mcast_compile_time_args();

template <typename Args>
void append_args_to(Args& destination, const std::vector<uint32_t>& args) {
    if constexpr (requires { destination.append(args); }) {
        destination.append(args);
    } else {
        destination.insert(destination.end(), args.begin(), args.end());
    }
}

}  // namespace detail

template <typename Args>
void append_absent_mcast_compile_time_args_to(Args& destination) {
    detail::append_args_to(destination, detail::absent_mcast_compile_time_args());
}

// Independent, disjoint groups sharing one protocol and semaphore allocation.
// Groups share a sender mode, number of sender rounds, and TransferMode.
class McastFamily {
public:
    // Collect groups, then prepare_arguments before querying. The borrowed device must stay open/alive
    // through successful argument preparation. Prepared queries and repeated prepare_arguments do not use it.
    // Copies own their data; building copies share the borrowed-device lifetime requirement.
    explicit McastFamily(tt::tt_metal::IDevice* device, const McastConfig& cfg = {});
    // One exact receiver set and a nonempty ordered sender list. One sender is fixed;
    // multiple senders rotate in the supplied order. Rejected additions preserve prior groups.
    void add_group(
        tt::tt_metal::CoreRangeSet receivers,
        std::vector<tt::tt_metal::CoreCoord> senders,
        std::optional<uint32_t> ack_count_override = std::nullopt);
    // Idempotent after success. Failure leaves queries blocked and collection data intact.
    void prepare_arguments();

    // All queries require successful argument preparation; additions are then forbidden.
    // =============================================================================
    // Operation construction paths:
    //
    // - ProgramDescriptor: attach resources and arguments to already placed kernel descriptors.
    // - ProgramSpec: attach named resources, argument schemas, and run arguments to kernel specs.
    // - Direct Program construction: append semaphores, compile-time args, and per-core runtime args separately.
    // =============================================================================

    // ProgramDescriptor path: attach resources and complete argument blocks to already placed kernels.
    void attach(
        tt::tt_metal::ProgramDescriptor& descriptor,
        std::string_view prefix,
        std::span<const std::reference_wrapper<tt::tt_metal::KernelDescriptor>> targets) const;
    void attach(
        tt::tt_metal::ProgramDescriptor& descriptor,
        std::string_view prefix,
        tt::tt_metal::KernelDescriptor& kernel) const {
        const std::array kernels{std::ref(kernel)};
        attach(descriptor, prefix, kernels);
    }

    // ProgramSpec path: attach named resources, argument schemas, and per-core run arguments.
    void attach(
        tt::tt_metal::experimental::ProgramSpec& spec,
        tt::tt_metal::experimental::ProgramRunArgs& run_args,
        std::string_view prefix,
        std::span<const tt::tt_metal::experimental::KernelSpecName> targets,
        std::span<const tt::tt_metal::experimental::SemaphoreSpecName> adopted = {}) const;

    // Direct Program construction, step 1: append multicast semaphores before constructing kernels.
    void append_semaphores(tt::tt_metal::Program& program);

    // Direct Program construction, step 2: append multicast compile-time arguments to existing kernel arguments.
    template <typename Args>
    void append_compile_time_args_to(Args& destination) const {
        require_program_bound_();
        detail::append_args_to(destination, compile_time_args_(program_semaphore_ids_));
    }

    // Direct Program construction, step 3: append this core's multicast runtime arguments to existing arguments.
    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        require_program_bound_();
        detail::append_args_to(destination, runtime_args_(core));
    }

    const tt::tt_metal::CoreRangeSet& participating_cores() const;
    tt::tt_metal::CoreRangeSet sender_only_cores() const;

private:
    std::vector<uint32_t> runtime_args_(const tt::tt_metal::CoreCoord& core) const;

    struct Group {
        Group(
            tt::tt_metal::CoreRangeSet receivers,
            std::vector<tt::tt_metal::CoreCoord> senders,
            std::optional<uint32_t> ack_count_override = std::nullopt);

        const tt::tt_metal::CoreRangeSet& receiver_cores() const { return receivers_; }

        const tt::tt_metal::CoreRangeSet& participating_cores() const { return participating_; }

        std::vector<uint32_t> runtime_args(
            const tt::tt_metal::CoreCoord& core, const dataflow_kernel_lib::mcast_wire::FamilyMetadata& layout) const;

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
            std::vector<uint32_t> acks;
            std::variant<PreparedMulticast, PreparedChain> transport = PreparedMulticast{};
        };
        void prepare_(
            tt::tt_metal::IDevice* device, const McastConfig& cfg, dataflow_kernel_lib::TransferMode transfer_mode);
        PreparedMulticast prepare_multicast_(const McastConfig& cfg, PreparedState& state) const;
        PreparedChain prepare_chain_(tt::tt_metal::IDevice* device, PreparedState& state) const;
        const PreparedState& prepared_state_() const;
        uint32_t sender_phase_(const tt::tt_metal::CoreCoord& core) const;
        tt::tt_metal::CoreRangeSet receivers_;
        std::vector<tt::tt_metal::CoreCoord> senders_;
        std::optional<uint32_t> ack_count_override_;
        tt::tt_metal::CoreRangeSet participating_;
        std::vector<uint32_t> fanouts_;
        std::optional<PreparedState> prepared_;
    };

    void require_arguments_prepared_() const;
    void require_program_bound_() const;
    void require_unbound_() const;
    std::array<uint32_t, 3> resolve_semaphore_ids_(std::span<const tt::tt_metal::SemaphoreDescriptor> existing) const;
    void validate_semaphores_present_and_zeroed_(
        std::span<const tt::tt_metal::SemaphoreDescriptor> existing, const std::array<uint32_t, 3>& ids) const;
    uint32_t required_semaphores_() const;
    std::vector<uint32_t> compile_time_args_(const std::array<uint32_t, 3>& ids) const;
    tt::tt_metal::IDevice* device_;
    bool arguments_prepared_ = false;
    // Preparation is the last use of the borrowed device; attachment uses these snapshots.
    tt::ARCH prepared_arch_{};
    tt::tt_metal::CoreCoord prepared_device_grid_;
    bool program_bound_ = false;
    std::array<uint32_t, 3> program_semaphore_ids_{UNUSED_SEM_ID, UNUSED_SEM_ID, UNUSED_SEM_ID};
    const Group* group_for_core_(const tt::tt_metal::CoreCoord& core) const;
    std::vector<Group> groups_;
    McastConfig cfg_;
    tt::tt_metal::CoreRangeSet receivers_;
    tt::tt_metal::CoreRangeSet participating_;
    dataflow_kernel_lib::mcast_wire::FamilyMetadata layout_;
};

// Mcast1D-specific types.

// Groups the receiver grid into independent row or column multicasts.
enum class Mcast1DShape {
    PerRow,
    PerColumn,
};

// Placement of the fixed sender on each row or column.
enum class Mcast1DSenderPlacement {
    Uniform,   // Use the same sender index on every line.
    Diagonal,  // Advance the fixed sender index with each line, wrapping at the receiver line length.
};

struct Mcast1DFixedSenderConfig {
    uint32_t starting_sender_index = 0;
    Mcast1DSenderPlacement sender_placement = Mcast1DSenderPlacement::Uniform;
};

struct Mcast1DRotatingSenderConfig {
    // Rotate over receivers when omitted. Explicit senders may be sparse or outside receivers,
    // but must align with receiver lines and provide the same nonzero count on every line.
    // Sender order is increasing x within rows or increasing y within columns.
    std::optional<tt::tt_metal::CoreRangeSet> sender_grid = std::nullopt;
};

using Mcast1DSenderConfig = std::variant<Mcast1DFixedSenderConfig, Mcast1DRotatingSenderConfig>;

// Configures independent row or column multicasts over a rectangular receiver grid.
// Fixed mode selects one sender per line. Rotating mode uses every core in sender_grid, or
// receivers when sender_grid is omitted.
class Mcast1D {
public:
    // Unified API: the sender configuration selects fixed or rotating behavior.
    Mcast1D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& receivers,
        Mcast1DShape shape,
        const Mcast1DSenderConfig& sender_config,
        const McastConfig& cfg = {});

    // =============================================================================
    // Operation construction paths:
    //
    // - ProgramDescriptor: attach resources and arguments to already placed kernel descriptors.
    // - ProgramSpec: attach named resources, argument schemas, and run arguments to kernel specs.
    // - Direct Program construction: append semaphores, compile-time args, and per-core runtime args separately.
    // =============================================================================

    // ProgramDescriptor path: attach resources and complete argument blocks to already placed kernels.
    void attach(
        tt::tt_metal::ProgramDescriptor&,
        std::string_view prefix,
        std::span<const std::reference_wrapper<tt::tt_metal::KernelDescriptor>> kernels) const;
    void attach(
        tt::tt_metal::ProgramDescriptor& descriptor,
        std::string_view prefix,
        tt::tt_metal::KernelDescriptor& kernel) const {
        const std::array kernels{std::ref(kernel)};
        attach(descriptor, prefix, kernels);
    }

    // ProgramSpec path: attach named resources, argument schemas, and per-core run arguments.
    void attach(
        tt::tt_metal::experimental::ProgramSpec&,
        tt::tt_metal::experimental::ProgramRunArgs&,
        std::string_view prefix,
        std::span<const tt::tt_metal::experimental::KernelSpecName> kernels,
        std::span<const tt::tt_metal::experimental::SemaphoreSpecName> adopted_semaphores = {}) const;

    // Direct Program construction, step 1: append multicast semaphores before constructing kernels.
    void append_semaphores(tt::tt_metal::Program& program);

    // Direct Program construction, step 2: append multicast compile-time arguments to existing kernel arguments.
    template <typename Args>
    void append_compile_time_args_to(Args& destination) const {
        family_->append_compile_time_args_to(destination);
    }

    // Direct Program construction, step 3: append this core's multicast runtime arguments. Cores outside the
    // participating topology receive a correctly sized argument block with neither role enabled.
    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        family_->append_runtime_args_to(destination, core);
    }

    const tt::tt_metal::CoreRangeSet& participating_cores() const;
    tt::tt_metal::CoreRangeSet sender_only_cores() const;

private:
    static std::vector<std::vector<tt::tt_metal::CoreCoord>> sender_lines_from_grid_(
        const tt::tt_metal::CoreRange& receiver_box, const tt::tt_metal::CoreRangeSet& sender_grid, Mcast1DShape shape);

    std::optional<McastFamily> family_;
};

// Mcast2D-specific types.

// Traversal order used to assign rotating sender rounds over a 2D sender grid.
enum class Mcast2DSenderOrder {
    RowMajor,
    ColumnMajor,
};

struct Mcast2DFixedSenderConfig {
    tt::tt_metal::CoreCoord sender;
};

struct Mcast2DRotatingSenderConfig {
    // Rotate over receivers when omitted.
    std::optional<tt::tt_metal::CoreRangeSet> sender_grid = std::nullopt;
    Mcast2DSenderOrder sender_order = Mcast2DSenderOrder::RowMajor;
};

using Mcast2DSenderConfig = std::variant<Mcast2DFixedSenderConfig, Mcast2DRotatingSenderConfig>;

// Configures one multicast over a rectangular receiver grid.
// The fixed sender may be inside or outside the receiver grid. Rotating mode uses sender_grid, or
// receivers when sender_grid is omitted.
class Mcast2D {
public:
    // Unified API: the sender configuration selects fixed or rotating behavior.
    Mcast2D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& receivers,
        const Mcast2DSenderConfig& sender_config,
        const McastConfig& cfg = {});

    // =============================================================================
    // Operation construction paths:
    //
    // - ProgramDescriptor: attach resources and arguments to already placed kernel descriptors.
    // - ProgramSpec: attach named resources, argument schemas, and run arguments to kernel specs.
    // - Direct Program construction: append semaphores, compile-time args, and per-core runtime args separately.
    // =============================================================================

    // ProgramDescriptor path: attach resources and complete argument blocks to already placed kernels.
    void attach(
        tt::tt_metal::ProgramDescriptor&,
        std::string_view prefix,
        std::span<const std::reference_wrapper<tt::tt_metal::KernelDescriptor>> kernels) const;
    void attach(
        tt::tt_metal::ProgramDescriptor& descriptor,
        std::string_view prefix,
        tt::tt_metal::KernelDescriptor& kernel) const {
        const std::array kernels{std::ref(kernel)};
        attach(descriptor, prefix, kernels);
    }

    // ProgramSpec path: attach named resources, argument schemas, and per-core run arguments.
    void attach(
        tt::tt_metal::experimental::ProgramSpec&,
        tt::tt_metal::experimental::ProgramRunArgs&,
        std::string_view prefix,
        std::span<const tt::tt_metal::experimental::KernelSpecName> kernels,
        std::span<const tt::tt_metal::experimental::SemaphoreSpecName> adopted_semaphores = {}) const;

    // Direct Program construction, step 1: append multicast semaphores before constructing kernels.
    void append_semaphores(tt::tt_metal::Program& program);

    // Direct Program construction, step 2: append multicast compile-time arguments to existing kernel arguments.
    template <typename Args>
    void append_compile_time_args_to(Args& destination) const {
        family_->append_compile_time_args_to(destination);
    }

    // Direct Program construction, step 3: append this core's multicast runtime arguments to existing arguments.
    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        family_->append_runtime_args_to(destination, core);
    }

private:
    static std::vector<tt::tt_metal::CoreCoord> senders_from_grid_(
        const tt::tt_metal::CoreRangeSet& sender_grid, Mcast2DSenderOrder sender_order);

    std::optional<McastFamily> family_;
};

}  // namespace ttnn::kernel_lib::host
