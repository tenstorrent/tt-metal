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


static std::tuple<KernelHandle,KernelHandle,KernelHandle> launch_programs_ct(
    tt::tt_metal::Program& program,
    CoreCoord coordination_core,
    CoreCoord receiver_core,
    CoreCoord sender_core
) {
    //This just creates a command of what needs to be compiled
    static std::string const& coordination_code = "ttnn/cpp/ttnn/operations/ccl/barrier/device/kernels/ubench_coordination.cpp";
    static std::string const& receiver_code = "ttnn/cpp/ttnn/operations/ccl/barrier/device/kernels/ubench_receiver.cpp";
    static std::string const& sender_code = "ttnn/cpp/ttnn/operations/ccl/barrier/device/kernels/ubench_sender.cpp";
    KernelHandle worker_coord_kernel_id = tt::tt_metal::CreateKernel(
        program,
        coordination_code,
        coordination_core,
        tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = {}});
    KernelHandle worker_receiver_kernel_id = tt::tt_metal::CreateKernel(
        program,
        receiver_code,
        receiver_core,
        tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = {}});
    KernelHandle worker_sender_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_code,
        sender_core,
        tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = {}});
    return {worker_coord_kernel_id, worker_receiver_kernel_id, worker_sender_kernel_id};
}

std::vector<uint32_t> get_eth_receiver_rt_args(
    Device *device,
    bool is_starting_core,
    uint32_t num_samples,
    uint32_t max_concurrent_samples,
    uint32_t sample_page_size,
    CoreCoord const& eth_sender_core,
    uint32_t start_semaphore,
    uint32_t init_handshake_core_x,
    uint32_t init_handshake_core_y,
    uint32_t init_handshake_semaphore_id
    ) {
    constexpr std::size_t semaphore_size = 16;
    std::vector<uint32_t> erisc_semaphore_addresses(max_concurrent_samples, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16);
    std::vector<uint32_t> erisc_buffer_addresses(max_concurrent_samples, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16 + tt::round_up(semaphore_size * max_concurrent_samples, 16));
    for (std::size_t i = 0; i < max_concurrent_samples; i++) {
        erisc_semaphore_addresses.at(i) += i * semaphore_size;
        erisc_buffer_addresses.at(i) += i * sample_page_size;
    }

    std::vector<uint32_t> rt_args = {
        static_cast<uint32_t>(is_starting_core ? 1 : 0),
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        num_samples,
        max_concurrent_samples,
        sample_page_size,
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).x),
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).y),
        start_semaphore,
        init_handshake_core_x,
        init_handshake_core_y,
        init_handshake_semaphore_id};
    for (std::size_t i = 0; i < max_concurrent_samples; i++) {
        rt_args.push_back(erisc_semaphore_addresses.at(i));
        rt_args.push_back(erisc_buffer_addresses.at(i));
    }

    return rt_args;
}

std::vector<uint32_t> get_eth_sender_rt_args(
    Device *device,
    bool is_starting_core,
    uint32_t num_samples,
    uint32_t max_concurrent_samples,
    uint32_t sample_page_size,
    uint32_t receiver_x,
    uint32_t receiver_y,
    uint32_t receiver_start_semaphore_id) {
    constexpr std::size_t semaphore_size = 16;
    std::vector<uint32_t> erisc_semaphore_addresses(max_concurrent_samples, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16);
    std::vector<uint32_t> erisc_buffer_addresses(max_concurrent_samples, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16 + tt::round_up(semaphore_size * max_concurrent_samples, 16));
    for (std::size_t i = 0; i < max_concurrent_samples; i++) {
        erisc_semaphore_addresses.at(i) += i * semaphore_size;
        erisc_buffer_addresses.at(i) += i * sample_page_size;
    }

    std::vector<uint32_t> rt_args = {
        static_cast<uint32_t>(is_starting_core ? 1 : 0),
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        num_samples,
        max_concurrent_samples,
        sample_page_size,
        receiver_x,
        receiver_y,
        receiver_start_semaphore_id};
    for (std::size_t i = 0; i < max_concurrent_samples; i++) {
        rt_args.push_back(erisc_semaphore_addresses.at(i));
        rt_args.push_back(erisc_buffer_addresses.at(i));
    }

    return rt_args;
}

operation::ProgramWithCallbacks barrier_with_workers(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const bool is_starting_core,
    const uint32_t num_samples,
    const uint32_t max_concurrent_samples,
    const uint32_t sample_page_size,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology) 
{
    //Configurable params
    const uint32_t num_links = 1;
    //Our intro into the kernel code
    log_trace(tt::LogOp, "barrier_with_workers entry");
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
    const CoreCoord coord_core = CoreCoord{0,0};
    const CoreCoord eth_receiver_core = topology_config.eth_receiver_cores.at(0);
    const CoreCoord eth_sender_core = topology_config.eth_sender_cores.at(0);
    const CoreCoord coord_core_ph = device->physical_core_from_logical_core(coord_core, CoreType::WORKER);
    const CoreCoord eth_receiver_core_ph = device->physical_core_from_logical_core(eth_receiver_core, CoreType::ETH);
    const CoreCoord eth_sender_core_ph = device->physical_core_from_logical_core(eth_sender_core, CoreType::ETH);


    //Set up the semaphores we use
    uint32_t worker_sem0 = tt::tt_metal::CreateSemaphore(program, coord_core, 0, CoreType::WORKER);
    uint32_t worker_sem1 = tt::tt_metal::CreateSemaphore(program, coord_core, 0, CoreType::WORKER);
    //TODO Use CoreType::ETH for the semaphore to replace it, also use get semaphore in kernel code
    uint32_t receiver_start_semaphore = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16;//CreateSemaphore(program, eth_receiver_core, 0, CoreType::ETH);
    //create eth_receiver_core and init_worker_core
    std::vector<uint32_t> worker_init_rt_args = {
            worker_sem0,
            worker_sem1,
            static_cast<uint32_t>(eth_receiver_core_ph.x),
            static_cast<uint32_t>(eth_receiver_core_ph.y),
            receiver_start_semaphore
        };
    // Prep the sender

    std::vector<uint32_t> const& sender_eth_rt_args = get_eth_sender_rt_args(
            device,
            is_starting_core,
            num_samples,
            max_concurrent_samples,
            sample_page_size,
            coord_core_ph.x,
            coord_core_ph.y,
            worker_sem1);
    // Prep the receiver

    std::vector<uint32_t> const& receiver_eth_rt_args = get_eth_receiver_rt_args(
            device,
            is_starting_core,
            num_samples,
            max_concurrent_samples,
            sample_page_size,
            eth_sender_core,
            receiver_start_semaphore,
            coord_core_ph.x,
            coord_core_ph.y,
            worker_sem0);

    //Launch the programs
    auto [worker_coord_kernel_id,worker_receiver_kernel_id,worker_sender_kernel_id] = launch_programs_ct(
        program,
        coord_core,
        eth_receiver_core,
        eth_sender_core
    );
    //Manage the runtime
    tt::tt_metal::SetRuntimeArgs(program, worker_receiver_kernel_id, eth_receiver_core, receiver_eth_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, worker_sender_kernel_id, eth_sender_core, sender_eth_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, worker_coord_kernel_id, coord_core, worker_init_rt_args);
    //Override the callback
    auto override_runtime_arguments_callback =
        [](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            ;
        };

    return { .program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

} //barrier_detail namespace end
} //ccl namespace end
} //ttnn namespace end