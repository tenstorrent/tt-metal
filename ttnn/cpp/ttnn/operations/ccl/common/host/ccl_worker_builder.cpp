// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iterator>

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"

namespace ttnn {
namespace ccl {
namespace worker_detail {

CCLWorkerArgBuilder::CCLWorkerArgBuilder (
    Device const* device,
    ttnn::ccl::CCLOpConfig const& op_config,
    ttnn::ccl::TensorPartition const& input_tensor_partition,
    ttnn::ccl::TensorPartition const& output_tensor_partition,
    std::size_t operating_dim):
    device(device),
    op_config(op_config),
    input_tensor_partition(input_tensor_partition),
    output_tensor_partition(output_tensor_partition),
    operating_dim(operating_dim) {
}

void emit_ccl_send_slice_sequence_commands(std::vector<TensorSlice> const& slices, std::vector<uint32_t>& args_out) {
    for (std::size_t i = 0; i < slices.size(); i++) {
        auto const& slice = slices[i];
        // Copy the header
        if (i == 0) {
            const std::size_t args_index_old = args_out.size();
            // push back Command Header
            args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandHeader::to_uint32(
                ttnn::ccl::cmd::CclCommandHeader{ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM, 1})));

            // push back arg 0 header
            args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES));
            auto const& ccl_command_tensor = ttnn::ccl::cmd::CclCommandTensor{
                Shape4D<uint32_t>(1, 1, slice.tensor_shape.y, slice.tensor_shape.x),
                Shape4D<uint32_t>(1, 1, slice.tensor_slice_shape.y, slice.tensor_slice_shape.x),
                Shape4D<uint32_t>(0, 0, slice.tensor_slice_offset.y, slice.tensor_slice_offset.x),
                Shape4D<uint32_t>(0, 0, slice.worker_slice_offset.y, slice.worker_slice_offset.x),
                slice.worker_slice_shape.x * slice.worker_slice_shape.y};
            const auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::size_in_words();
            log_trace(tt::LogOp, "Emitting {} args for full tensor slice command", num_words_for_args);
            args_out.resize(args_out.size() + num_words_for_args);
            // push_back arg 0 payload
            ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::
                pack_to(
                    &args_out[args_out.size() - num_words_for_args],
                    ccl_command_tensor
                    );
            const std::size_t args_index_new = args_out.size();

            TT_ASSERT(i < slices.size(), "Internal Error");
            std::stringstream ss; ss << "ccl_send command " << std::to_string(i) << " has " << args_index_new - args_index_old << " args:\n";
            for (std::size_t j = args_index_old; j < args_index_new; j++) {
                ss << "\targ " << j << ":" << args_out[j] << "\n";
            }
            log_trace(tt::LogOp, "{}", ss.str());
            // We can reused cached values for the first slice
        } else {
            auto const& last_slice = slices[i - 1];
            const std::size_t args_index_old = args_out.size();
            auto header_index = args_out.size();
            args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandHeader::to_uint32(
                ttnn::ccl::cmd::CclCommandHeader{ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM, 1})));
            std::size_t num_args = 0;

            // tensor shape
            if (last_slice.tensor_shape != slice.tensor_shape) {
                args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for tensor_shape field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::pack_to(
                    &args_out[args_out.size() - num_words_for_args],
                    Shape4D<uint32_t>(1, 1, slice.tensor_shape.y, slice.tensor_shape.x)
                );
                for (std::size_t j = args_out.size() - num_words_for_args; j < args_out.size(); j++) {
                    log_trace(tt::LogOp, "\t{}", args_out[j]);
                }

                num_args++;
            }

            // tensor slice shape
            if (last_slice.tensor_slice_shape != slice.tensor_slice_shape) {
                args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for tensor_slice_shape field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::pack_to(
                    &args_out[args_out.size() - num_words_for_args],
                    Shape4D<uint32_t>(1, 1, slice.tensor_slice_shape.y, slice.tensor_slice_shape.x)
                );
                for (std::size_t i = args_out.size() - num_words_for_args; i < args_out.size(); i++) {
                    log_trace(tt::LogOp, "\t{}", args_out[i]);
                }

                num_args++;
            }

            // tensor slice offset
            if (last_slice.tensor_slice_offset != slice.tensor_slice_offset) {
                args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for tensor_slice_offset field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::pack_to(
                    &args_out[args_out.size() - num_words_for_args],
                    Shape4D<uint32_t>(0, 0, slice.tensor_slice_offset.y, slice.tensor_slice_offset.x)
                );
                for (std::size_t j = args_out.size() - num_words_for_args; j < args_out.size(); j++) {
                    log_trace(tt::LogOp, "\t{}", args_out[j]);
                }

                num_args++;
            }

            // worker slice offset
            if (last_slice.worker_slice_offset != slice.worker_slice_offset) {
                args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for worker_slice_offset field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::pack_to(
                    &args_out[args_out.size() - num_words_for_args],
                    Shape4D<uint32_t>(0, 0, slice.worker_slice_offset.y, slice.worker_slice_offset.x)
                );

                for (std::size_t j = args_out.size() - num_words_for_args; j < args_out.size(); j++) {
                    log_trace(tt::LogOp, "\t{}", args_out[j]);
                }
                num_args++;
            }

            // worker_pages_per_slice
            if (last_slice.worker_slice_shape != slice.worker_slice_shape) {
                args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for worker_pages_per_slice field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::pack_to(
                    &args_out[args_out.size() - num_words_for_args],
                    slice.worker_slice_shape.y * slice.worker_slice_shape.x
                );
                for (std::size_t j = args_out.size() - num_words_for_args; j < args_out.size(); j++) {
                    log_trace(tt::LogOp, "\t{}", args_out[j]);
                }

                num_args++;
            }

            args_out[header_index] = static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandHeader::to_uint32(
                ttnn::ccl::cmd::CclCommandHeader{ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM, 1}));

            std::size_t args_index_new = args_out.size();
            std::stringstream ss; ss << "ccl_send command " << i << " has " << args_index_new - args_index_old << " args:\n";
            for (std::size_t j = args_index_old; j < args_index_new; j++) {
                ss << "\targ " << j << ":" << args_out[j] << "\n";
            }
            log_trace(tt::LogOp, "{}", ss.str());
        }
    }
}

std::vector<uint32_t> CCLWorkerArgBuilder::generate_sender_reader_kernel_rt_args(
    ttnn::ccl::InterleavedTensorWorkerSlice worker_slice,
    std::size_t operating_dim,
    uint32_t num_pages_per_packet,
    uint32_t worker_slice_index) const
{
    const std::size_t num_commands_expected = this->input_tensor_partition.partition_size - 1;

    auto const& tensor_shape = worker_slice.tensor_shape;
    auto const& tensor_slice_shape = worker_slice.tensor_slice_shape;

    auto num_slices = input_tensor_partition.partition_size;
    auto start_slice_index = input_tensor_partition.partition_index;
    std::int64_t end_slice_index_exclusive = input_tensor_partition.partition_index + 1;

    if (input_tensor_partition.partition_index==0){
        log_trace(tt::LogOp, "ccl_send_writer start_slice_index = {}", start_slice_index);
        log_trace(tt::LogOp, "ccl_send_writer end_slice_index_exclusive = {}", end_slice_index_exclusive);
    }

    // Add the command args
    auto const& slices = generate_slice_sequence_on_dim_v2(
        tensor_shape,
        worker_slice.worker_slice_shape,
        worker_slice.worker_slice_offset,
        operating_dim,
        num_slices,
        start_slice_index,
        end_slice_index_exclusive,
        worker_slice_index
    );
    TT_ASSERT(num_commands_expected == slices.size());

    // If we are on device zero, we send n-1 chunks in ascending order
    auto &input_tensor = this->op_config.get_input_tensor(0);
    TT_ASSERT(input_tensor.get_legacy_shape().size() == 4, "Only 4D tensors are supported for ccl");
    ttnn::ccl::Shape4D<uint32_t> input_tensor_shape = {input_tensor.get_legacy_shape()[0], input_tensor.get_legacy_shape()[1],input_tensor.get_legacy_shape()[2],input_tensor.get_legacy_shape()[3]};

    std::vector<uint32_t> args = {
        static_cast<uint32_t>(input_tensor.buffer()->address()),
        static_cast<uint32_t>(slices.size())
    };
    std::size_t logged_arg_idx = 0;
    if (input_tensor_partition.partition_index==0) log_trace(tt::LogOp, "ccl_send_reader arg[{}]: buffer_address = {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;
    if (input_tensor_partition.partition_index==0) log_trace(tt::LogOp, "ccl_send_reader arg[{}]: num_commands = {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    std::ranges::copy(std::vector<uint32_t>{num_pages_per_packet}, std::back_inserter(args));
    if (input_tensor_partition.partition_index==0) log_trace(tt::LogOp, "ccl_send_reader arg[{}]: pages_per_packet {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    std::ranges::copy(std::vector<uint32_t>{this->op_config.get_page_size()}, std::back_inserter(args));
    if (input_tensor_partition.partition_index==0) log_trace(tt::LogOp, "ccl_send_reader arg[{}]: page_size {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    auto const& addr_gen_rt_args = ttnn::ccl::emit_address_generator_runtime_args(this->device, input_tensor);
    std::ranges::copy(addr_gen_rt_args, std::back_inserter(args));
    for (auto const& arg : addr_gen_rt_args) {
        if (input_tensor_partition.partition_index==0) log_trace(tt::LogOp, "ccl_send_reader arg[{}]: addr_gen_rt_args[] {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;
    }

    if (input_tensor_partition.partition_index==0) log_trace(tt::LogOp, "ccl_send_reader Generating {} ccl send commands", slices.size());
    emit_ccl_send_slice_sequence_commands(slices, args);

    if (input_tensor_partition.partition_index==0) log_trace(tt::LogOp, "ccl_send_reader Sender Worker has {} RT Args: {}", args.size(), args);

    return args;
}

std::vector<uint32_t> CCLWorkerArgBuilder::generate_sender_writer_kernel_rt_args(
    ttnn::ccl::InterleavedTensorWorkerSlice worker_slice,
    std::size_t operating_dim,
    uint32_t num_pages_per_packet,
    uint32_t worker_slice_index) const
{
    const std::size_t num_commands_expected = this->output_tensor_partition.partition_size - 1;

    auto const& tensor_shape = worker_slice.tensor_shape;
    auto const& tensor_slice_shape = worker_slice.tensor_slice_shape;

    auto num_slices = output_tensor_partition.partition_size;
    auto start_slice_index = output_tensor_partition.partition_index;
    std::int64_t end_slice_index_exclusive = output_tensor_partition.partition_index + 1;

    log_trace(tt::LogOp, "ccl_send_writer start_slice_index = {}", start_slice_index);
    log_trace(tt::LogOp, "ccl_send_writer end_slice_index_exclusive = {}", end_slice_index_exclusive);

    // Add the command args
    auto const& slices = generate_slice_sequence_on_dim_v2(
        tensor_shape,
        worker_slice.worker_slice_shape,
        worker_slice.worker_slice_offset,
        operating_dim,
        num_slices,
        start_slice_index,
        end_slice_index_exclusive,
        worker_slice_index
    );
    TT_ASSERT(num_commands_expected == slices.size());

    // If we are on device zero, we send n-1 chunks in ascending order
    auto &output_tensor = this->op_config.get_output_tensor(0);
    TT_ASSERT(output_tensor.get_legacy_shape().size() == 4, "Only 4D tensors are supported for ccl");
    ttnn::ccl::Shape4D<uint32_t> output_tensor_shape = {output_tensor.get_legacy_shape()[0], output_tensor.get_legacy_shape()[1],output_tensor.get_legacy_shape()[2],output_tensor.get_legacy_shape()[3]};

    std::vector<uint32_t> args = {
        static_cast<uint32_t>(output_tensor.buffer()->address()),
        static_cast<uint32_t>(slices.size())
    };
    std::size_t logged_arg_idx = 0;
    log_trace(tt::LogOp, "ccl_send_writer arg[{}]: buffer_address = {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;
    log_trace(tt::LogOp, "ccl_send_writer arg[{}]: num_commands = {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    std::ranges::copy(std::vector<uint32_t>{num_pages_per_packet}, std::back_inserter(args));
    log_trace(tt::LogOp, "ccl_send_writer arg[{}]: pages_per_packet {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    std::ranges::copy(std::vector<uint32_t>{this->op_config.get_page_size()}, std::back_inserter(args));
    log_trace(tt::LogOp, "ccl_send_writer arg[{}]: page_size {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    auto const& addr_gen_rt_args = ttnn::ccl::emit_address_generator_runtime_args(this->device, output_tensor);
    std::ranges::copy(addr_gen_rt_args, std::back_inserter(args));
    for (auto const& arg : addr_gen_rt_args) {
        log_trace(tt::LogOp, "ccl_send_writer arg[{}]: addr_gen_rt_args[] {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;
    }

    log_trace(tt::LogOp, "ccl_send_writer Generating {} ccl send commands", slices.size());
    emit_ccl_send_slice_sequence_commands(slices, args);

    log_trace(tt::LogOp, "ccl_send_writer Sender Worker has {} RT Args: {}", args.size(), args);

    return args;
}

std::vector<uint32_t> CCLWorkerArgBuilder::generate_sender_reader_kernel_ct_args() const
{
    std::vector<uint32_t> args = {
        static_cast<uint32_t>(this->op_config.get_input_tensor(0).memory_config().memory_layout), // tensor memory layout
        static_cast<uint32_t>(this->op_config.get_input_tensor(0).buffer()->buffer_type()), // buffer type
        static_cast<uint32_t>(this->op_config.get_input_tensor(0).layout()), // page layout
        static_cast<uint32_t>(tt::CB::c_in0) // cb_id
    };

    auto const& input_tensor = this->op_config.get_input_tensor(0);
    auto const& addr_gen_rt_args = ttnn::ccl::emit_address_generator_compile_time_args(input_tensor);
    std::ranges::copy(addr_gen_rt_args, std::back_inserter(args));

    return args;
}

std::vector<uint32_t> CCLWorkerArgBuilder::generate_sender_writer_kernel_ct_args() const
{
    std::vector<uint32_t> args = {
        static_cast<uint32_t>(this->op_config.get_output_tensor(0).memory_config().memory_layout), // tensor memory layout
        static_cast<uint32_t>(this->op_config.get_output_tensor(0).buffer()->buffer_type()), // buffer type
        static_cast<uint32_t>(this->op_config.get_output_tensor(0).layout()), // page layout
        static_cast<uint32_t>(tt::CB::c_in0) // cb_id
    };

    auto const& output_tensor = this->op_config.get_output_tensor(0);
    auto const& addr_gen_rt_args = ttnn::ccl::emit_address_generator_compile_time_args(output_tensor);
    std::ranges::copy(addr_gen_rt_args, std::back_inserter(args));

    return args;
}

} // namespace worker_detail
} // namespace ccl
} // namespace ttnn
