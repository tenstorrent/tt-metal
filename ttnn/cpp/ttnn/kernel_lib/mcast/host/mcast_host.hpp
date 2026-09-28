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

#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_common.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {
struct ProgramSpec;
struct ProgramRunArgs;
}  // namespace tt::tt_metal::experimental

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
// mcast.attach(descriptor, "input_mcast", kernels);
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
// mcast.attach(descriptor, "input_mcast", kernels);
// descriptor.kernels.push_back(std::move(kernel));
//
// These examples use ProgramDescriptor. Mcast also supports ProgramSpec attachment and direct Program
// construction through append_semaphores(), append_compile_time_args_to(), and append_runtime_args_to().
// =============================================================================

struct McastArgumentOffsets {
    uint32_t compile_time;
    uint32_t runtime;
};

void attach_absent(tt::tt_metal::KernelDescriptor& kernel, std::string_view prefix);
void attach_absent(
    tt::tt_metal::experimental::ProgramSpec&,
    std::string_view prefix,
    std::span<const tt::tt_metal::experimental::KernelSpecName> kernels);

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

enum class McastCoreOrder { RowMajor, ColumnMajor };
enum class McastSenderPlacement { Uniform, Staggered };

struct McastConfig {
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    bool handshake = true;
    // nullopt means all receivers; an empty set means no acknowledgments.
    // The current sender never acknowledges itself. Chain forwarding requires
    // all receivers and therefore accepts only nullopt or the full receiver set.
    std::optional<tt::tt_metal::CoreRangeSet> handshake_cores = std::nullopt;
    dataflow_kernel_lib::DataReadySignal data_ready = dataflow_kernel_lib::DataReadySignal::Flag;
    std::optional<uint32_t> base_sem_id;
    std::optional<std::vector<uint32_t>> sem_ids;
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
    ~Mcast();

    Mcast(const Mcast&);
    Mcast& operator=(const Mcast&);
    Mcast(Mcast&&) noexcept;
    Mcast& operator=(Mcast&&) noexcept;

    void attach(
        tt::tt_metal::ProgramDescriptor& descriptor,
        std::string_view prefix,
        std::span<const std::reference_wrapper<tt::tt_metal::KernelDescriptor>> kernels) const;
    void attach(
        tt::tt_metal::experimental::ProgramSpec& spec,
        tt::tt_metal::experimental::ProgramRunArgs& args,
        std::string_view prefix,
        std::span<const tt::tt_metal::experimental::KernelSpecName> kernels,
        std::span<const tt::tt_metal::experimental::SemaphoreSpecName> adopted_semaphores = {}) const;
    void append_semaphores(tt::tt_metal::Program& program);
    template <typename Args>
    void append_compile_time_args_to(Args& destination) const {
        detail::append_args_to(destination, compile_time_args_());
    }
    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        detail::append_args_to(destination, runtime_args_(core));
    }
    McastArgumentOffsets append_kernel_args_to(
        std::vector<uint32_t>& compile_time_args,
        tt::tt_metal::KernelDescriptor::RuntimeArgs& runtime_args,
        const tt::tt_metal::CoreRangeSet& placement) const;
    const tt::tt_metal::CoreRangeSet& participating_cores() const;
    tt::tt_metal::CoreRangeSet sender_only_cores() const;

private:
    std::unique_ptr<McastImpl> impl_;

    std::vector<uint32_t> compile_time_args_() const;
    std::vector<uint32_t> runtime_args_(const tt::tt_metal::CoreCoord& core) const;
};

}  // namespace ttnn::kernel_lib::host
