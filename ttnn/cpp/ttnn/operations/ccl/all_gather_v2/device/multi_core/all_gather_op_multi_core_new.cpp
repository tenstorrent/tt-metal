// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "tt_metal/common/core_coord.hpp"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"

#include <sstream>
#include <type_traits>
#include <ranges>

#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"


using namespace tt::constants;

namespace ttnn {

using namespace ccl;
// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
operation::ProgramWithCallbacks all_gather_multi_core_with_workers_new(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ccl::Topology topology) {

    // Sleep for 5 * ring_index seconds (for DEBUG only)
    // std::chrono::seconds sleep_duration(5 * ring_index);
    // std::this_thread::sleep_for(sleep_duration);

    // // Log device id and ring index
    // log_info(tt::LogOp, "Generating log for Ring Index: {}", ring_index);

    tt::tt_metal::Program program{};

    TT_FATAL(!(receiver_device_id == std::nullopt && sender_device_id == std::nullopt), "At least one of receiver_device_id or sender_device_id must be specified");

    std::unique_ptr<ccl::CclOpTensorConfig> input_tensor_config = ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(input_tensor);
    std::unique_ptr<ccl::CclOpTensorConfig> output_tensor_config = ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(output_tensor);

    const auto& device = input_tensor.device();

    bool is_sharded = input_tensor.is_sharded();

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    auto const& op_config =ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto const& input_tensor_partition = ttnn::ccl::TensorPartition(1, 0);  // one partition, 0 index
    auto const& output_tensor_partition = ttnn::ccl::TensorPartition(ring_size, ring_index);  // ring_size partitions, ring_index index

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    auto const& sender_worker_core_range = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_workers_per_link-1, num_links - 1)));
    auto const& sender_worker_cores = corerange_to_cores(sender_worker_core_range, std::nullopt, true);

    // CB Creation
    uint32_t num_pages_per_packet = 1; // we assume 1 page per packet for now
    uint32_t cb_num_pages = num_pages_per_packet; // There is a bug with double/tripple buffering. Still debugging.
    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t page_size_bytes = op_config.get_page_size();
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_size_bytes);
    CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Create Tensor slicer
    auto input_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicer ( // can be used for all gather as well if we set ring_size to 1, which means read the entire input tensor in reduce scatter sense.
        input_tensor,
        input_tensor,
        dim,
        0, // ring_index, set as 0 to "trick" the reduce scatter tensor slicer to read the entire input tensor
        1, // ring_size, set as 1 to "trick" the reduce scatter tensor slicer to read the entire input tensor
        num_links, // num_workers_per_slicer, set 1 per link for now
        UINT32_MAX, // max_worker_slice_in_bytes, set as infinite for now
        cb_num_pages / 2);
    auto output_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicer (
        output_tensor,
        output_tensor,
        dim,
        ring_index,
        ring_size,
        num_links, // num_workers_per_slicer, set 1 per link for now
        UINT32_MAX, // max_worker_slice_in_bytes, set as infinite for now
        cb_num_pages / 2);

    // KERNEL CREATION
    auto worker_arg_builder = ccl::worker_detail::CCLWorkerArgBuilder(
        device,
        op_config,
        input_tensor_partition,
        output_tensor_partition,
        dim);

    auto const& worker_defines = op_config.emit_worker_defines();
    static std::string const& sender_kernel_reader_path = "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader.cpp";
    static std::string const& sender_kernel_writer_path = "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_writer.cpp";

    KernelHandle worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_kernel_reader_path,
        sender_worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(worker_arg_builder.generate_sender_reader_kernel_ct_args(), worker_defines));

    KernelHandle worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_kernel_writer_path,
        sender_worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(worker_arg_builder.generate_sender_writer_kernel_ct_args(), worker_defines));

    // RT Args
    for (std::size_t link = 0; link < num_links; link++) {
        CoreCoord core = {num_workers_per_link-1, link};
        std::size_t worker_tensor_slice_index = link;
        auto const& input_worker_slice = input_tensor_slicer.get_worker_slice(worker_tensor_slice_index);
        auto const& output_worker_slice = output_tensor_slicer.get_worker_slice(worker_tensor_slice_index);
        auto worker_arg_builder = ccl::worker_detail::CCLWorkerArgBuilder(
            device,
            op_config,
            input_tensor_partition,
            output_tensor_partition,
            dim);

        // tt::log_info("Creating RT Args for worker core ({},{})", core.x, core.y);
        // input_worker_slice.print();
        // output_worker_slice.print();

        auto const sender_reader_rt_args = worker_arg_builder.generate_sender_reader_kernel_rt_args(input_worker_slice, worker_arg_builder.operating_dim, num_pages_per_packet, worker_tensor_slice_index);
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_sender_reader_kernel_id,
            core,
            sender_reader_rt_args);

        auto const sender_writer_rt_args = worker_arg_builder.generate_sender_writer_kernel_rt_args(output_worker_slice, worker_arg_builder.operating_dim, num_pages_per_packet, worker_tensor_slice_index);
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_sender_writer_kernel_id,
            core,
            sender_writer_rt_args);
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         sender_worker_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto& input = input_tensors[0];
        const auto& output = output_tensors[0];

        // update senders
        auto &worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
        auto &worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
        for (auto const& core : sender_worker_cores) {
            // reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args.at(0) = input.buffer()->address();
            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args.at(0) = output.buffer()->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace ttnn
