// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::ccl::barrier::detail {

static std::tuple<KernelHandle, KernelHandle, KernelHandle> schedule_kernel_compile(
    tt::tt_metal::Program& program,
    const CoreCoord sender_core,
    const CoreCoord receiver_core,
    const CoreCoord sem_init_core) {
    // This just creates a command of what needs to be compiled
    static std::string const& receiver_code =
        "ttnn/cpp/ttnn/operations/ccl/barrier/device/kernels/barrier_receiver.cpp";
    static std::string const& sender_code = "ttnn/cpp/ttnn/operations/ccl/barrier/device/kernels/barrier_sender.cpp";
    static std::string const& sem_init_code =
        "ttnn/cpp/ttnn/operations/ccl/barrier/device/kernels/barrier_sem_creator.cpp";

    KernelHandle receiver_kernel_id = tt::tt_metal::CreateKernel(
        program, receiver_code, receiver_core, tt::tt_metal::EthernetConfig{.compile_args = {}});
    KernelHandle sender_kernel_id =
        tt::tt_metal::CreateKernel(program, sender_code, sender_core, tt::tt_metal::EthernetConfig{.compile_args = {}});
    KernelHandle sem_init_id = tt::tt_metal::CreateKernel(
        program, sem_init_code, sem_init_core, tt::tt_metal::DataMovementConfig{.compile_args = {}});
    return {receiver_kernel_id, sender_kernel_id, sem_init_id};
}

static std::tuple<std::array<uint32_t, 7>, std::array<uint32_t, 10>, std::array<uint32_t, 5>> get_rt_args(
    tt::tt_metal::Program& program,
    IDevice* device,
    bool is_starting_core,
    CoreCoord const& eth_sender_core,
    CoreCoord const& eth_receiver_core,
    CoreCoord const& sem_init_core) {
    const uint32_t worker_sem0 = CreateSemaphore(program, sem_init_core, 0, CoreType::WORKER);
    const uint32_t worker_sem1 = CreateSemaphore(program, sem_init_core, 0, CoreType::WORKER);
    uint32_t start_semaphore_address = hal::get_erisc_l1_unreserved_base() + EriscDatamoverConfig::eth_word_size_bytes;
    uint32_t erisc_semaphore_address =
        hal::get_erisc_l1_unreserved_base() + (EriscDatamoverConfig::eth_word_size_bytes * 2);
    uint32_t erisc_buffer_address =
        hal::get_erisc_l1_unreserved_base() + (EriscDatamoverConfig::eth_word_size_bytes * 3);

    const std::array<uint32_t, 10> receiver_rt_args = {
        static_cast<uint32_t>(is_starting_core ? 1 : 0),
        hal::get_erisc_l1_unreserved_base(),
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).x),
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).y),
        erisc_semaphore_address,
        start_semaphore_address,
        erisc_buffer_address,
        static_cast<uint32_t>(device->virtual_core_from_logical_core(sem_init_core, CoreType::WORKER).x),
        static_cast<uint32_t>(device->virtual_core_from_logical_core(sem_init_core, CoreType::WORKER).y),
        worker_sem0};
    const std::array<uint32_t, 7> sender_rt_args = {
        static_cast<uint32_t>(is_starting_core ? 1 : 0),  // is_ring_start
        hal::get_erisc_l1_unreserved_base(),              // handshake_addr
        erisc_buffer_address,
        erisc_semaphore_address,
        static_cast<uint32_t>(device->virtual_core_from_logical_core(sem_init_core, CoreType::WORKER).x),
        static_cast<uint32_t>(device->virtual_core_from_logical_core(sem_init_core, CoreType::WORKER).y),
        worker_sem1};  // sample size
    const std::array<uint32_t, 5> sem_id_args = {
        worker_sem0,
        worker_sem1,
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_receiver_core).x),
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_receiver_core).y),
        start_semaphore_address};
    return {sender_rt_args, receiver_rt_args, sem_id_args};
}

operation::ProgramWithCallbacks barrier_with_workers(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const bool is_starting_core,
    const uint32_t ring_size,
    const uint32_t ring_index,
    chip_id_t target_device_id,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology) {
    // Configurable params
    const uint32_t num_links = 1;
    // Our intro into the kernel code
    // Turn our tensors into a vector of tensors with 1 entry each
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    // Configure operational parameters
    auto const& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);

    // Get the device from the tensor
    const auto& device =
        input_tensor.mesh_device() ? input_tensor.mesh_device()->get_device(target_device_id) : input_tensor.device();
    // Get a representation of the topology

    // Create the program
    tt::tt_metal::Program program{};
    //////////////////
    // query the core ids
    auto const& topology_config = ttnn::ccl::RingTopology(
        device, topology, sender_device_id, receiver_device_id, num_links, ring_size, ring_index);
    const CoreCoord eth_receiver_core = topology_config.eth_receiver_cores.at(0);
    const CoreCoord eth_sender_core = topology_config.eth_sender_cores.at(0);
    const CoreCoord sem_init_core = CoreCoord{0, 0};

    // Schedule the kernels to be compiled
    const auto [receiver_kernel_id, sender_kernel_id, sem_init_id] =
        schedule_kernel_compile(program, eth_sender_core, eth_receiver_core, sem_init_core);
    // Manage the runtime

    const auto [sender_eth_rt_args, receiver_eth_rt_args, sem_id_args] =
        get_rt_args(program, device, is_starting_core, eth_sender_core, eth_receiver_core, sem_init_core);

    tt::tt_metal::SetRuntimeArgs(program, receiver_kernel_id, eth_receiver_core, receiver_eth_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, sender_kernel_id, eth_sender_core, sender_eth_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, sem_init_id, sem_init_core, sem_id_args);
    return {.program = std::move(program)};
}

}  // namespace ttnn::ccl::barrier::detail
