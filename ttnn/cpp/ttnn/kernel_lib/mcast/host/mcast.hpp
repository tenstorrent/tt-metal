// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "ttnn/kernel_lib/mcast/mcast_protocol.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>

namespace ttnn::kernel_lib::host {

class McastImpl;

// =============================================================================
// Usage examples
//
// The examples assume that device, descriptor, noc, and the shown cores and receiver sets already exist, and
// that kernel is a placed KernelDescriptor with its operation-specific compile-time and runtime arguments.
//
// Create one independent multicast per row, with the first core in each row as its fixed sender.
//
// Mcast mcast(
//     device,
//     McastConfig{.noc = noc},
//     receivers,
//     receiver_row_size,
//     McastFixedSenderConfig{},
//     McastCoreOrder::RowMajor);
// const std::array kernels{std::ref(kernel)};
// mcast.attach(descriptor, "input_mcast", kernels, 0);
// const uint32_t next_semaphore_id = mcast.next_semaphore_id();
// descriptor.kernels.push_back(std::move(kernel));
//
// Create one multicast over the receiver set from an explicit sender.
//
// Mcast mcast(
//     device,
//     McastConfig{.noc = noc},
//     receivers,
//     receivers.num_cores(),
//     McastExplicitSenderConfig{{{sender}}});
// const std::array kernels{std::ref(kernel)};
// mcast.attach(descriptor, "input_mcast", kernels, 0);
// descriptor.kernels.push_back(std::move(kernel));
//
// These examples use ProgramDescriptor. Mcast also supports ProgramSpec attachment and direct Program
// construction through append_semaphores(), append_compile_time_args_to(), and append_runtime_args_to().
// =============================================================================

// ProgramDescriptor mode: attach an absent channel to one KernelDescriptor.
void attach_absent_mcast(tt::tt_metal::KernelDescriptor& kernel, std::string_view prefix);

// ProgramSpec mode: attach an absent channel to the named target kernels.
void attach_absent_mcast(
    tt::tt_metal::experimental::ProgramSpec& spec,
    std::string_view prefix,
    std::span<const tt::tt_metal::experimental::KernelSpecName> targets);

// Legacy direct-Program mode: append the absent channel's compile-time tag.
template <typename Args>
void append_absent_mcast_compile_time_args_to(Args& destination) {
    destination.push_back(dataflow_kernel_lib::mcast_wire::ABSENT);
}

enum class McastCoreOrder { RowMajor, ColumnMajor };
enum class McastSenderPlacement { Uniform, Staggered };

struct McastConfig {
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    bool handshake = true;
    // nullopt means all receivers send the ready signal; an empty set means none do.
    // The current sender never sends the ready signal to itself. Chain forwarding requires
    // all receivers and therefore accepts only nullopt or the full receiver set.
    std::optional<tt::tt_metal::CoreRangeSet> handshake_cores = std::nullopt;
    dataflow_kernel_lib::DataReadySignal data_ready = dataflow_kernel_lib::DataReadySignal::Flag;
    dataflow_kernel_lib::TransferMode irregular_receiver_set_mode = dataflow_kernel_lib::TransferMode::Multicast;
};

struct McastFixedSenderConfig {
    // Uniform requires an in-group index. Staggered wraps index + group number.
    uint32_t sender_index = 0;
    McastSenderPlacement placement = McastSenderPlacement::Uniform;
};

struct McastRotatingSenderConfig {};

struct McastSenderGridConfig {
    tt::tt_metal::CoreRangeSet sender_cores;
    std::optional<McastCoreOrder> sender_order;  // Defaults to receiver order.
};

struct McastExplicitSenderConfig {
    std::vector<std::vector<tt::tt_metal::CoreCoord>> senders_per_group;
};

using McastSenderConfig =
    std::variant<McastFixedSenderConfig, McastRotatingSenderConfig, McastSenderGridConfig, McastExplicitSenderConfig>;

// Partitions the ordered receiver cores into equal consecutive groups and
// owns the resulting multicast lowering snapshot.
class Mcast {
public:
    Mcast(
        const tt::tt_metal::IDevice& device,
        const McastConfig& config,
        const tt::tt_metal::CoreRangeSet& receivers,
        uint32_t receiver_group_size,
        const McastSenderConfig& sender_config = McastFixedSenderConfig{},
        McastCoreOrder receiver_order = McastCoreOrder::RowMajor);
    ~Mcast();  // NOLINT(performance-trivially-destructible): McastImpl is incomplete here.

    Mcast(const Mcast&);
    Mcast& operator=(const Mcast&);
    Mcast(Mcast&&) noexcept;
    Mcast& operator=(Mcast&&) noexcept;

    // ProgramDescriptor adapter: allocates descriptor semaphores and appends positional
    // arguments plus named offsets to each target KernelDescriptor.
    void attach(
        tt::tt_metal::ProgramDescriptor& descriptor,
        std::string_view prefix,
        std::span<const std::reference_wrapper<tt::tt_metal::KernelDescriptor>> kernels,
        uint32_t first_semaphore_id) const;
    // Valid only after a successful ProgramDescriptor attachment.
    uint32_t next_semaphore_id() const;

    // Metal 2.0 adapter: adds named semaphore bindings and named compile-time/runtime
    // arguments to ProgramSpec and ProgramRunArgs.
    void attach(
        tt::tt_metal::experimental::ProgramSpec& spec,
        tt::tt_metal::experimental::ProgramRunArgs& args,
        std::string_view prefix,
        std::span<const tt::tt_metal::experimental::KernelSpecName> kernels) const;

    // Legacy direct-Program adapter: allocate semaphores first, then append positional
    // arguments while constructing the target kernel.
    void append_semaphores(tt::tt_metal::Program& program);

    // Append multicast arguments after the operation-specific prefix. Every core using
    // the same kernel must have the same runtime prefix length.
    template <typename Args>
    void append_compile_time_args_to(Args& destination) const {
        append_args_to_(destination, compile_time_args_());
    }

    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        append_args_to_(destination, runtime_args_(core));
    }

    // Topology queries used to place kernels and other program resources.
    const tt::tt_metal::CoreRangeSet& participating_cores() const;
    tt::tt_metal::CoreRangeSet sender_only_cores() const;

private:
    template <typename Args>
    static void append_args_to_(Args& destination, const std::vector<uint32_t>& args) {
        if constexpr (requires { destination.append(args); }) {
            destination.append(args);
        } else {
            destination.insert(destination.end(), args.begin(), args.end());
        }
    }

    std::unique_ptr<McastImpl> impl_;

    std::vector<uint32_t> compile_time_args_() const;
    std::vector<uint32_t> runtime_args_(const tt::tt_metal::CoreCoord& core) const;
};

}  // namespace ttnn::kernel_lib::host
