// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "common/core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"

#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"

namespace ttnn {
namespace ccl {
namespace barrier_detail {


static std::tuple<KernelHandle,KernelHandle,KernelHandle> schedule_kernel_compile(
    tt::tt_metal::Program& program,
    CoreCoord sender_core,
    CoreCoord receiver_core,
    CoreCoord sem_init_core
) {
    //This just creates a command of what needs to be compiled
    static std::string const& receiver_code = "ttnn/cpp/ttnn/operations/ccl/barrier/device/kernels/barrier_receiver_HPerf_LPrec.cpp";
    static std::string const& sender_code = "ttnn/cpp/ttnn/operations/ccl/barrier/device/kernels/barrier_sender_HPerf_LPrec.cpp";
    static std::string const& sem_init_code = "ttnn/cpp/ttnn/operations/ccl/barrier/device/kernels/barrier_sem_creator.cpp";

    KernelHandle receiver_kernel_id = tt::tt_metal::CreateKernel(
        program,
        receiver_code,
        receiver_core,
        tt::tt_metal::EthernetConfig {
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {}});
    KernelHandle sender_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_code,
        sender_core,
        tt::tt_metal::EthernetConfig {
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = {}});
    KernelHandle sem_init_id = tt::tt_metal::CreateKernel(
        program,
        sem_init_code,
        sem_init_core,
        tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = {}});
    return {receiver_kernel_id, sender_kernel_id,sem_init_id};
}

static std::tuple<std::vector<uint32_t>,std::vector<uint32_t>,std::vector<uint32_t>> get_rt_args(
    tt::tt_metal::Program& program,
    Device *device,
    uint32_t ring_index,
    bool is_starting_core,
    uint32_t sample_page_size,
    CoreCoord const& eth_sender_core,
    CoreCoord const& eth_receiver_core,
    CoreCoord const& sem_init_core
    ) {

    uint32_t worker_sem0 = CreateSemaphore(program, sem_init_core, 0, CoreType::WORKER);
    uint32_t worker_sem1 = CreateSemaphore(program, sem_init_core, 0, CoreType::WORKER);
    uint32_t start_semaphore =              eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16;
    uint32_t erisc_semaphore_address =      eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16;
    uint32_t erisc_buffer_address=          eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16 + 16;

    std::vector<uint32_t> receiver_rt_args = {
        static_cast<uint32_t>(is_starting_core ? 1 : 0),
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        ring_index,
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).x),
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).y),
        start_semaphore,
        erisc_buffer_address,
        erisc_semaphore_address,
        static_cast<uint32_t>(device->physical_core_from_logical_core(sem_init_core, CoreType::WORKER).x),
        static_cast<uint32_t>(device->physical_core_from_logical_core(sem_init_core, CoreType::WORKER).y),
        worker_sem0
    };
    std::vector<uint32_t> sender_rt_args = {
        static_cast<uint32_t>(is_starting_core ? 1 : 0),//is_ring_start
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,//handshake_addr
        ring_index,
        erisc_buffer_address,
        erisc_semaphore_address,
        static_cast<uint32_t>(device->physical_core_from_logical_core(sem_init_core, CoreType::WORKER).x),
        static_cast<uint32_t>(device->physical_core_from_logical_core(sem_init_core, CoreType::WORKER).y),
        worker_sem1
        }; //sample size
    std::vector<uint32_t> sem_id_args = {
        worker_sem0,
        worker_sem1,
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_receiver_core).x),
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_receiver_core).y),
        start_semaphore
    };
    return {sender_rt_args, receiver_rt_args,sem_id_args};
    }

operation::ProgramWithCallbacks barrier_with_workers(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const bool is_starting_core,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology)
{
    //Configurable params
    const uint32_t num_links = 1;
    const uint32_t sample_page_size=16;
    //Our intro into the kernel code
    //Turn our tensors into a vector of tensors with 1 entry each
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    //Configure operational parameters
    //ttnn::ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode =ttnn::ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;
    auto const& op_config =ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    //Get the configuration file, works for both sharded or unsharded
    std::unique_ptr<ttnn::ccl::CclOpTensorConfig> input_tensor_config =
        ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(input_tensor);
    std::unique_ptr<ttnn::ccl::CclOpTensorConfig> output_tensor_config =
        ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(output_tensor);

    //Get the device from the tensor
    const auto& device = input_tensor.device();
    //Get a representation of the topology

    //Create the program
    tt::tt_metal::Program program{};
    //////////////////
    //query the core ids
    auto const& topology_config =
       ttnn::ccl::RingTopology(device, topology, sender_device_id, receiver_device_id, num_links, ring_size, ring_index);
    const CoreCoord eth_receiver_core = topology_config.eth_receiver_cores.at(0);
    const CoreCoord eth_sender_core = topology_config.eth_sender_cores.at(0);
    const CoreCoord sem_init_core = CoreCoord{0,0};

    //Schedule the kernels to be compiled
    auto [receiver_kernel_id,sender_kernel_id,sem_init_id] = schedule_kernel_compile(
        program,
        eth_sender_core,
        eth_receiver_core,
        sem_init_core
    );
    //Manage the runtime

    auto [sender_eth_rt_args, receiver_eth_rt_args, sem_id_args] = get_rt_args(
            program,
            device,
            ring_index,
            is_starting_core,
            sample_page_size,
            eth_sender_core,
            eth_receiver_core,
            sem_init_core);

    tt::tt_metal::SetRuntimeArgs(program, receiver_kernel_id, eth_receiver_core, receiver_eth_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, sender_kernel_id, eth_sender_core, sender_eth_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, sem_init_id, sem_init_core, sem_id_args);
    //Override the callback with no changes
    auto override_runtime_arguments_callback =
        [](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            ;
        };

    return { .program = std::move(program)};
}

} //barrier_detail namespace end
} //ccl namespace end
} //ttnn namespace end
