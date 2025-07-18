// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iterator>

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include <tt-metalium/fabric.hpp>
#include "tt-metalium/kernel_types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/erisc_datamover_builder.hpp>

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"
#include <tt_stl/overloaded.hpp>

#include <optional>
#include <variant>
#include <vector>

namespace ttnn::ccl::worker_detail {

CCLWorkerArgBuilder::CCLWorkerArgBuilder(
    IDevice const* device,
    ttnn::ccl::CCLOpConfig const& op_config,
    ttnn::ccl::TensorPartition const& input_tensor_partition,
    ttnn::ccl::TensorPartition const& output_tensor_partition,
    std::size_t operating_dim) :
    device(device),
    op_config(op_config),
    input_tensor_partition(input_tensor_partition),
    output_tensor_partition(output_tensor_partition),
    operating_dim(operating_dim) {}

Shape4D<uint32_t> to_4d_shape(Shape4D<uint32_t> const& shape) { return shape; }
Shape4D<uint32_t> to_4d_offset(Shape4D<uint32_t> const& offset) { return offset; }
size_t get_volume(Shape4D<uint32_t> const& shape) { return shape.volume(); }

Shape4D<uint32_t> to_4d_shape(tt_xy_pair const& shape) { return Shape4D<uint32_t>(1, 1, shape.y, shape.x); }
Shape4D<uint32_t> to_4d_offset(tt_xy_pair const& offset) { return Shape4D<uint32_t>(0, 0, offset.y, offset.x); }
size_t get_volume(tt_xy_pair const& shape) { return shape.x * shape.y; }

template <cmd::CclCommandArgCode code>
struct tensor_slice_command_arg_field {
    using type = std::nullptr_t;
};
template <>
struct tensor_slice_command_arg_field<cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES> {
    static auto get_value(v2::TensorSlice const& s) { return s.tensor_shape; };
};
template <>
struct tensor_slice_command_arg_field<cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES> {
    static auto get_value(v2::TensorSlice const& s) { return s.tensor_slice_shape; };
};
template <>
struct tensor_slice_command_arg_field<cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES> {
    static auto get_value(v2::TensorSlice const& s) { return s.tensor_slice_offset; };
};
template <>
struct tensor_slice_command_arg_field<cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES> {
    static auto get_value(v2::TensorSlice const& s) { return s.worker_slice_offset; };
};
template <>
struct tensor_slice_command_arg_field<cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE> {
    static auto get_value(v2::TensorSlice const& s) { return get_volume(s.worker_slice_shape); };
};
template <>
struct tensor_slice_command_arg_field<cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES> {
    static auto get_value(v2::TensorSlice const& s) {
        return ttnn::ccl::cmd::CclCommandTensor{
            to_4d_shape(s.tensor_shape),
            to_4d_shape(s.tensor_slice_shape),
            to_4d_offset(s.tensor_slice_offset),
            to_4d_offset(s.worker_slice_offset),
            get_volume(s.worker_slice_shape)};
    };
};

template <ttnn::ccl::cmd::CclCommandArgCode arg_code>
void add_ccl_command_arg_to_runtime_args(v2::TensorSlice const& tensor_slice, std::vector<uint32_t>& rt_args_out) {
    rt_args_out.push_back(static_cast<uint32_t>(arg_code));
    auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<arg_code>::size_in_words();
    log_trace(tt::LogOp, "Emitting {} args for tensor_shape field", num_words_for_args);
    rt_args_out.resize(rt_args_out.size() + num_words_for_args);

    ttnn::ccl::cmd::CclCommandArg<arg_code>::pack_to(
        &rt_args_out[rt_args_out.size() - num_words_for_args],
        tensor_slice_command_arg_field<arg_code>::get_value(tensor_slice));

    for (std::size_t j = rt_args_out.size() - num_words_for_args; j < rt_args_out.size(); j++) {
        log_trace(tt::LogOp, "\t{}", rt_args_out[j]);
    }
}
template <>
void add_ccl_command_arg_to_runtime_args<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>(
    v2::TensorSlice const& tensor_slice, std::vector<uint32_t>& rt_args_out) {
    rt_args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE));
    auto num_words_for_args = 1;
    log_trace(tt::LogOp, "Emitting {} args for tensor_shape field", num_words_for_args);
    rt_args_out.resize(rt_args_out.size() + num_words_for_args);

    ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::pack_to(
        &rt_args_out[rt_args_out.size() - num_words_for_args],
        tensor_slice_command_arg_field<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::get_value(
            tensor_slice));

    for (std::size_t j = rt_args_out.size() - num_words_for_args; j < rt_args_out.size(); j++) {
        log_trace(tt::LogOp, "\t{}", rt_args_out[j]);
    }
}

template <typename TensorSliceType>
void generate_ccl_slice_sequence_commands_impl(
    std::vector<TensorSliceType> const& slices,
    ttnn::ccl::cmd::CclCommandCode command_type,
    std::vector<uint32_t>& args_out,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args) {
    for (std::size_t i = 0; i < slices.size(); i++) {
        auto const& slice = slices[i];
        // Copy the header
        if (i == 0) {
            const std::size_t args_index_old = args_out.size();
            // push back Command Header
            args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandHeader::to_uint32(
                ttnn::ccl::cmd::CclCommandHeader{command_type, dest_args, 1})));

            // push back arg 0 header
            args_out.push_back(
                static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES));
            auto const& ccl_command_tensor = ttnn::ccl::cmd::CclCommandTensor{
                to_4d_shape(slice.tensor_shape),
                to_4d_shape(slice.tensor_slice_shape),
                to_4d_offset(slice.tensor_slice_offset),
                to_4d_offset(slice.worker_slice_offset),
                get_volume(slice.worker_slice_shape)};
            const auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<
                ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::size_in_words();
            log_trace(tt::LogOp, "Emitting {} args for full tensor slice command", num_words_for_args);
            args_out.resize(args_out.size() + num_words_for_args);
            // push_back arg 0 payload
            ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::
                pack_to(&args_out[args_out.size() - num_words_for_args], ccl_command_tensor);
            const std::size_t args_index_new = args_out.size();

            TT_ASSERT(i < slices.size(), "Internal Error");
            std::stringstream ss;
            ss << "ccl_send command " << std::to_string(i) << " has " << args_index_new - args_index_old << " args:\n";
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
                ttnn::ccl::cmd::CclCommandHeader{command_type, dest_args, 1})));

            // tensor shape
            if (last_slice.tensor_shape != slice.tensor_shape) {
                args_out.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<
                    ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for tensor_shape field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::pack_to(
                    &args_out[args_out.size() - num_words_for_args], to_4d_shape(slice.tensor_shape));
                for (std::size_t j = args_out.size() - num_words_for_args; j < args_out.size(); j++) {
                    log_trace(tt::LogOp, "\t{}", args_out[j]);
                }
            }

            // tensor slice shape
            if (last_slice.tensor_slice_shape != slice.tensor_slice_shape) {
                args_out.push_back(
                    static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<
                    ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for tensor_slice_shape field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::
                    pack_to(&args_out[args_out.size() - num_words_for_args], to_4d_shape(slice.tensor_slice_shape));
                for (std::size_t i = args_out.size() - num_words_for_args; i < args_out.size(); i++) {
                    log_trace(tt::LogOp, "\t{}", args_out[i]);
                }
            }

            // tensor slice offset
            if (last_slice.tensor_slice_offset != slice.tensor_slice_offset) {
                args_out.push_back(
                    static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<
                    ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for tensor_slice_offset field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::
                    pack_to(&args_out[args_out.size() - num_words_for_args], to_4d_offset(slice.tensor_slice_offset));
                for (std::size_t j = args_out.size() - num_words_for_args; j < args_out.size(); j++) {
                    log_trace(tt::LogOp, "\t{}", args_out[j]);
                }
            }

            // worker slice offset
            if (last_slice.worker_slice_offset != slice.worker_slice_offset) {
                args_out.push_back(static_cast<uint32_t>(
                    ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<
                    ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for worker_slice_offset field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<
                    ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::
                    pack_to(&args_out[args_out.size() - num_words_for_args], to_4d_offset(slice.worker_slice_offset));

                for (std::size_t j = args_out.size() - num_words_for_args; j < args_out.size(); j++) {
                    log_trace(tt::LogOp, "\t{}", args_out[j]);
                }
            }

            // worker_pages_per_slice
            if (last_slice.worker_slice_shape != slice.worker_slice_shape) {
                args_out.push_back(
                    static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<
                    ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for worker_pages_per_slice field", num_words_for_args);
                args_out.resize(args_out.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::pack_to(
                    &args_out[args_out.size() - num_words_for_args], get_volume(slice.worker_slice_shape));
                for (std::size_t j = args_out.size() - num_words_for_args; j < args_out.size(); j++) {
                    log_trace(tt::LogOp, "\t{}", args_out[j]);
                }
            }

            args_out[header_index] = static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandHeader::to_uint32(
                ttnn::ccl::cmd::CclCommandHeader{command_type, dest_args, 1}));

            std::size_t args_index_new = args_out.size();
            std::stringstream ss;
            ss << "ccl_send command " << i << " has " << args_index_new - args_index_old << " args:\n";
            for (std::size_t j = args_index_old; j < args_index_new; j++) {
                ss << "\targ " << j << ":" << args_out[j] << "\n";
            }
            log_trace(tt::LogOp, "{}", ss.str());
        }
    }
}

/*
 * Number of CCL command arguments generated - note that this does not necessarily match
 * the number of runtime args generated.
 */
size_t generate_ccl_tensor_slice_command_args(
    std::optional<v2::TensorSlice> const& last_tensor_slice,
    v2::TensorSlice const& current_tensor_slice,
    std::vector<uint32_t>& args_out) {
    // Copy the header
    std::size_t num_command_args_added = 0;
    auto const args_index_old = args_out.size();
    if (!last_tensor_slice.has_value()) {
        const std::size_t args_index_old = args_out.size();
        // push back Command Header
        // push back arg 0 header
        log_trace(tt::LogOp, "Generating full tensor spec command args");
        add_ccl_command_arg_to_runtime_args<ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>(
            current_tensor_slice, args_out);
        const size_t args_index_new = args_out.size();
        // We can reused cached values for the first slice
        num_command_args_added++;
    } else {
        auto const& last_slice = last_tensor_slice.value();
        const std::size_t args_index_old = args_out.size();
        auto header_index = args_out.size();

        // tensor shape
        if (last_slice.tensor_shape != current_tensor_slice.tensor_shape) {
            add_ccl_command_arg_to_runtime_args<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>(
                current_tensor_slice, args_out);
            num_command_args_added++;
        }

        // tensor slice shape
        if (last_slice.tensor_slice_shape != current_tensor_slice.tensor_slice_shape) {
            add_ccl_command_arg_to_runtime_args<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>(
                current_tensor_slice, args_out);
            num_command_args_added++;
        }

        // tensor slice offset
        if (last_slice.tensor_slice_offset != current_tensor_slice.tensor_slice_offset) {
            add_ccl_command_arg_to_runtime_args<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>(
                current_tensor_slice, args_out);
            num_command_args_added++;
        }

        // worker slice offset
        if (last_slice.worker_slice_offset != current_tensor_slice.worker_slice_offset) {
            add_ccl_command_arg_to_runtime_args<
                ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>(
                current_tensor_slice, args_out);
            num_command_args_added++;
        }

        // worker_pages_per_slice
        if (last_slice.worker_slice_shape != current_tensor_slice.worker_slice_shape) {
            add_ccl_command_arg_to_runtime_args<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>(
                current_tensor_slice, args_out);
            num_command_args_added++;
        }
    }

    log_trace(
        tt::LogOp, "\t{} rt_args added, {} cmd args added", args_out.size() - args_index_old, num_command_args_added);

    return num_command_args_added;
}

// TODO: commonize with all uncached arg types (e.g. this can be commonized with atomic inc arg generation)
size_t generate_ccl_wait_value_command_args(
    ttnn::ccl::cmd::CclCommandWaitValue const& wait_value_args, std::vector<uint32_t>& args_out) {
    auto const arg_code = ttnn::ccl::cmd::CclCommandArgCode::SET_TARGET_VALUE;
    ttnn::ccl::cmd::CclCommandArgHeader hdr;
    hdr.code = arg_code;
    hdr.inline_value0 = static_cast<uint8_t>(true);
    hdr.inline_value1 = wait_value_args.target_value;
    args_out.push_back(hdr.to_uint32());
    log_trace(
        tt::LogOp,
        "Emitting header only for for wait_value field. header.code={}, .inline_val0={}, .inline_val1={}",
        static_cast<int>(hdr.code),
        hdr.inline_value0,
        hdr.inline_value1);

    return 1;
}

size_t generate_ccl_raw_inline_write_command_args(
    ttnn::ccl::cmd::CclCommandInlineReadWrite const& inline_rw_args, std::vector<uint32_t>& args_out) {
    auto const arg_code = ttnn::ccl::cmd::CclCommandArgCode::SET_TARGET_VALUE;
    ttnn::ccl::cmd::CclCommandArgHeader hdr;
    hdr.code = arg_code;
    hdr.inline_value0 = static_cast<uint8_t>(true);
    hdr.inline_value1 = inline_rw_args.value;
    args_out.push_back(hdr.to_uint32());
    log_trace(
        tt::LogOp,
        "Emitting header only for for inline write field. header.code={}, .inline_val0={}, .inline_val1={}",
        static_cast<int>(hdr.code),
        hdr.inline_value0,
        hdr.inline_value1);
    return 1;
}

static size_t generate_ccl_atomic_inc_command_args(
    ttnn::ccl::cmd::CclCommandAtomicInc const& atomic_inc_args, std::vector<uint32_t>& args_out) {
    auto const arg_code = ttnn::ccl::cmd::CclCommandArgCode::SET_ATOMIC_INC_VALUE;
    ttnn::ccl::cmd::CclCommandArgHeader hdr;
    hdr.code = arg_code;
    hdr.inline_value0 = static_cast<uint8_t>(true);
    hdr.inline_value1 = atomic_inc_args.value;
    TT_FATAL(
        atomic_inc_args.value < std::numeric_limits<uint8_t>::max(),
        "Atomic increment value is too large: {}",
        atomic_inc_args.value);
    args_out.push_back(hdr.to_uint32());

    log_trace(
        tt::LogOp,
        "Emitting header only for for atomic_inc field. header.code={}, .inline_val0={}, .inline_val1={}",
        static_cast<int>(hdr.code),
        hdr.inline_value0,
        hdr.inline_value1);

    return 1;
}

/*
 * Returns the number of ccl command args added
 */
static size_t generate_ccl_address_info_command_args(
    std::optional<std::pair<ttnn::ccl::cmd::CclCommandAddrType, ttnn::ccl::cmd::CclCommandAddrArgs>> const&
        last_addr_type,
    std::pair<ttnn::ccl::cmd::CclCommandAddrType, ttnn::ccl::cmd::CclCommandAddrArgs> const& current_addr_type_args,
    ttnn::ccl::cmd::SRC_DEST_TYPE src_dest_type,
    std::vector<uint32_t>& args_out) {
    auto requires_args_to_be_generated = [](auto const& last_addr_type, auto const& current_addr_type_args) {
        bool different_type_or_args = !last_addr_type.has_value();
        different_type_or_args =
            different_type_or_args || (last_addr_type.value().first != current_addr_type_args.first);
        different_type_or_args =
            different_type_or_args || (last_addr_type.value().second.index() != current_addr_type_args.second.index());
        if (different_type_or_args) {
            return true;
        }
        if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrSemaphoreId>(current_addr_type_args.second)) {
            auto const& last_semaphore_id =
                std::get<ttnn::ccl::cmd::CclCommandAddrSemaphoreId>(last_addr_type.value().second);
            auto const& current_semaphore_id =
                std::get<ttnn::ccl::cmd::CclCommandAddrSemaphoreId>(current_addr_type_args.second);
            return last_semaphore_id.semaphore_id != current_semaphore_id.semaphore_id;
        }
        if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrCircularBufferId>(current_addr_type_args.second)) {
            auto const& last_circular_buffer_id =
                std::get<ttnn::ccl::cmd::CclCommandAddrCircularBufferId>(last_addr_type.value().second);
            auto const& current_circular_buffer_id =
                std::get<ttnn::ccl::cmd::CclCommandAddrCircularBufferId>(current_addr_type_args.second);
            return last_circular_buffer_id.circular_buffer_id != current_circular_buffer_id.circular_buffer_id;
        }
        if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress>(current_addr_type_args.second)) {
            auto const& last_absolute_address =
                std::get<ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress>(last_addr_type.value().second);
            auto const& current_absolute_address =
                std::get<ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress>(current_addr_type_args.second);
            return last_absolute_address.absolute_address != current_absolute_address.absolute_address;
        }
        if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrRelativeAddress>(current_addr_type_args.second)) {
            auto const& last_relative_address =
                std::get<ttnn::ccl::cmd::CclCommandAddrRelativeAddress>(last_addr_type.value().second);
            auto const& current_relative_address =
                std::get<ttnn::ccl::cmd::CclCommandAddrRelativeAddress>(current_addr_type_args.second);
            return last_relative_address.relative_address != current_relative_address.relative_address;
        }
        if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrNone>(current_addr_type_args.second)) {
            return false;
        }
        return true;
    };

    size_t num_ccl_command_args_added = 0;
    if (requires_args_to_be_generated(last_addr_type, current_addr_type_args)) {
        const size_t header_index = args_out.size();
        args_out.push_back(0);
        num_ccl_command_args_added++;
        ttnn::ccl::cmd::CclCommandArgHeader header;
        header.code = ttnn::ccl::cmd::CclCommandArgCode::SET_ADDRESS_INFO;
        if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress>(current_addr_type_args.second)) {
            log_trace(tt::LogOp, "Emitting {} args for absolute_address field", 2);
            header.inline_value0 = src_dest_type;
            header.inline_value1 = static_cast<uint8_t>(ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS);

            auto const& absolute_address =
                std::get<ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress>(current_addr_type_args.second);
            args_out.push_back(absolute_address.absolute_address);
        } else if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrRelativeAddress>(
                       current_addr_type_args.second)) {
            log_trace(tt::LogOp, "Emitting {} args for relative_address field at index {}", 2, header_index);
            header.inline_value0 = src_dest_type;
            header.inline_value1 = static_cast<uint8_t>(ttnn::ccl::cmd::CclCommandAddrType::RELATIVE_ADDRESS);

            auto const& relative_address =
                std::get<ttnn::ccl::cmd::CclCommandAddrRelativeAddress>(current_addr_type_args.second);
            args_out.push_back(relative_address.relative_address);
        } else if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrSemaphoreId>(current_addr_type_args.second)) {
            log_trace(tt::LogOp, "Emitting {} args for semaphore_id field at index {}", 1, header_index);
            header.inline_value0 = src_dest_type;
            header.inline_value1 = static_cast<uint8_t>(ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID);

            auto const& semaphore_id =
                std::get<ttnn::ccl::cmd::CclCommandAddrSemaphoreId>(current_addr_type_args.second);
            header.inline_value2 = semaphore_id.semaphore_id;
        } else if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrCircularBufferId>(
                       current_addr_type_args.second)) {
            log_trace(tt::LogOp, "Emitting {} args for circular_buffer_id field at index {}", 1, header_index);
            header.inline_value0 = src_dest_type;
            header.inline_value1 = static_cast<uint8_t>(ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID);

            auto const& circular_buffer_id =
                std::get<ttnn::ccl::cmd::CclCommandAddrCircularBufferId>(current_addr_type_args.second);
            header.inline_value2 = circular_buffer_id.circular_buffer_id;
        } else if (std::holds_alternative<ttnn::ccl::cmd::CclCommandAddrNone>(current_addr_type_args.second)) {
            log_trace(tt::LogOp, "Emitting {} args for NONE addr field at index {}", 1, header_index);
            header.inline_value0 = src_dest_type;
            header.inline_value1 = static_cast<uint8_t>(ttnn::ccl::cmd::CclCommandAddrType::NONE);
            // do nothing
        } else {
            TT_THROW("Unsupported address type: {}", static_cast<int>(current_addr_type_args.first));
        }
        log_trace(
            tt::LogOp,
            "\theader.code={}, .inline_val0={}, .inline_val1={}, .inline_val2={}",
            static_cast<int>(header.code),
            header.inline_value0,
            header.inline_value1,
            header.inline_value2);
        args_out[header_index] = header.to_uint32();
    }

    return num_ccl_command_args_added;
}

size_t generate_ccl_core_descriptor_info_command_args(
    std::optional<
        std::pair<ttnn::ccl::cmd::CclCommandCoreDescriptorType, ttnn::ccl::cmd::CclCommandCoreDescriptorArgs>> const&
        last_core_descriptor,
    std::pair<ttnn::ccl::cmd::CclCommandCoreDescriptorType, ttnn::ccl::cmd::CclCommandCoreDescriptorArgs> const&
        current_core_descriptor,
    std::vector<uint32_t>& args_out) {
    size_t num_ccl_command_args_added = 0;
    bool requires_update_to_args =
        !last_core_descriptor.has_value() || (last_core_descriptor.value().first != current_core_descriptor.first);
    requires_update_to_args = requires_update_to_args ||
                              (last_core_descriptor.value().second.index() != current_core_descriptor.second.index());
    if (!requires_update_to_args) {
        if (std::holds_alternative<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeAddrgen>(
                current_core_descriptor.second)) {
            requires_update_to_args = false;
        } else if (std::holds_alternative<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeLocal>(
                       current_core_descriptor.second)) {
            requires_update_to_args = true;
        } else if (std::holds_alternative<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY>(
                       current_core_descriptor.second)) {
            auto const& last_noc_xy =
                std::get<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY>(last_core_descriptor.value().second);
            auto const& current_noc_xy =
                std::get<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY>(current_core_descriptor.second);
            requires_update_to_args = (last_noc_xy.x != current_noc_xy.x) || (last_noc_xy.y != current_noc_xy.y);
        } else if (std::holds_alternative<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeMcast>(
                       current_core_descriptor.second)) {
            auto const& last_rectangle =
                std::get<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeMcast>(last_core_descriptor.value().second);
            auto const& current_rectangle =
                std::get<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeMcast>(current_core_descriptor.second);
            requires_update_to_args = (last_rectangle.noc0_start_x != current_rectangle.noc0_start_x) ||
                                      (last_rectangle.noc0_start_y != current_rectangle.noc0_start_y) ||
                                      (last_rectangle.noc0_end_x != current_rectangle.noc0_end_x) ||
                                      (last_rectangle.noc0_end_y != current_rectangle.noc0_end_y);
        }
    }
    if (requires_update_to_args) {
        const size_t header_index = args_out.size();
        log_trace(tt::LogOp, "Emitting {} args for core_descriptor field at index {}", 1, header_index);
        args_out.push_back(0);
        ttnn::ccl::cmd::CclCommandArgHeader hdr;
        hdr.code = ttnn::ccl::cmd::CclCommandArgCode::SET_CORE_DESCRIPTOR_INFO;
        hdr.inline_value0 = static_cast<uint8_t>(current_core_descriptor.first);
        if (std::holds_alternative<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY>(current_core_descriptor.second)) {
            auto const& noc_xy =
                std::get<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY>(current_core_descriptor.second);
            hdr.inline_value1 = noc_xy.x;
            hdr.inline_value2 = noc_xy.y;
        } else if (std::holds_alternative<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeMcast>(
                       current_core_descriptor.second)) {
            auto const& rectangle =
                std::get<ttnn::ccl::cmd::CclCommandCoreDescriptorTypeMcast>(current_core_descriptor.second);
            args_out.push_back(rectangle.to_uint32());
        }
        log_trace(
            tt::LogOp,
            "\theader.code={}, .inline_val0={}, .inline_val1={}, .inline_val2={}",
            static_cast<int>(hdr.code),
            hdr.inline_value0,
            hdr.inline_value1,
            hdr.inline_value2);
        args_out[header_index] = hdr.to_uint32();
        num_ccl_command_args_added++;
    }
    return num_ccl_command_args_added;
}

static size_t generate_ccl_noc_transfer_burst_command_args(
    const ttnn::ccl::cmd::HostCclCommandNocTransferBurst& noc_burst_descriptor,
    size_t tensor_index,
    ttnn::ccl::tensor_address_runtime_args_overrider &rt_args_overrider_out,
    std::vector<uint32_t>& args_out) {
    ttnn::ccl::cmd::CclCommandArgHeader hdr;
    hdr.code = ttnn::ccl::cmd::CclCommandArgCode::SET_NOC_TRANSFER_BURST_START_INFO;
    TT_FATAL(noc_burst_descriptor.num_transfers_total > 0, "Internal Error. num_transfers_total uninitialized when generating runtime args for noc read/write commands");
    hdr.inline_value0 = noc_burst_descriptor.num_transfers_total;
    // Bank base address must be set in the next arg since we may need the full 32-bit value
    args_out.push_back(hdr.to_uint32());
    rt_args_overrider_out.add_runtime_arg_index(tensor_index, args_out.size());
    args_out.push_back(noc_burst_descriptor.bank_base_address);

    for (auto const& transfer_group : noc_burst_descriptor.transfer_burst_groupings) {
        args_out.push_back(transfer_group.num_transfers_per_packet);
        for (auto const& transfer : transfer_group.transfer_infos) {
            args_out.push_back(transfer.noc_addr & 0xFFFFFFFF);
            args_out.push_back(transfer.noc_addr >> 32);
            args_out.push_back(transfer.noc_transfer_size_bytes);
        }
    }

    return 1;
}

void validate_ccl_command_dest_args(ttnn::ccl::cmd::CclCommandDestArgs const& dest_args) {
    bool valid = std::holds_alternative<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args) ||
                 std::holds_alternative<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args) ||
                 std::holds_alternative<ttnn::ccl::cmd::LocalOnlyCommandDestArgs>(dest_args);
    if (!valid) {
        TT_THROW(
            "Unsupported CCL command dest args. Expected one of UnicastCommandDestArgs, MulticastCommandDestArgs, or "
            "LocalOnlyCommandDestArgs");
    }
}
void validate_ccl_command_dest_type(ttnn::ccl::cmd::CclCommandDestType dest_type) {
    bool valid = dest_type == ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST ||
                 dest_type == ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST ||
                 dest_type == ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY;
    if (!valid) {
        TT_THROW("Unsupported CCL command dest type: {}", static_cast<int>(dest_type));
    }
}

void validate_command(ttnn::ccl::cmd::CclHostLowLevelWorkerCommand const& command) {
    validate_ccl_command_dest_type(command.fabric_transfer_type);
    validate_ccl_command_dest_args(command.fabric_transfer_args);
}

void generate_ccl_command_stream_to_kernel_args(
    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> const& ccl_command_stream,
    std::optional<size_t> tensor_index,
    std::optional<std::vector<size_t>> const& tensor_indices,
    ttnn::ccl::tensor_address_runtime_args_overrider *rt_args_overrider_out,
    std::vector<uint32_t>& rt_args_out) {
    std::optional<v2::TensorSlice> last_tensor_slice = std::nullopt;

    bool fill_args_overrider = rt_args_overrider_out != nullptr;
    TT_FATAL(!fill_args_overrider || tensor_index != std::nullopt, "Internal Error: When generating CCL command stream to kernel args, a runtime args overrider was provided but no tensor command index map was provided.");
    std::optional<std::pair<ttnn::ccl::cmd::CclCommandAddrType, ttnn::ccl::cmd::CclCommandAddrArgs>>
        last_src_addr_type = std::nullopt;
    std::optional<std::pair<ttnn::ccl::cmd::CclCommandAddrType, ttnn::ccl::cmd::CclCommandAddrArgs>>
        last_dest_addr_type = std::nullopt;
    std::optional<std::pair<ttnn::ccl::cmd::CclCommandCoreDescriptorType, ttnn::ccl::cmd::CclCommandCoreDescriptorArgs>>
        last_core_descriptor = std::nullopt;

    log_trace(tt::LogOp, "Generating CCL command stream to kernel args, starting at index {}", rt_args_out.size());

    for (size_t i = 0; i < ccl_command_stream.size(); i++) {
        log_trace(tt::LogOp, "New command starting at arg idx: {}", rt_args_out.size());
        auto const& command = ccl_command_stream[i];
        validate_command(command);

        // Set aside the placeholder rt arg for the command header
        const size_t command_header_rt_arg_index = rt_args_out.size();
        static_assert(sizeof(ttnn::ccl::cmd::CclCommandHeader) == sizeof(uint32_t));
        const size_t old_rt_args_start_index = rt_args_out.size();
        rt_args_out.push_back(0);
        // populate the body (ccl command args)of the command
        size_t num_ccl_command_args_added = 0;

        // populate the src_addr_type
        num_ccl_command_args_added += generate_ccl_address_info_command_args(
            last_src_addr_type,
            {command.source_addr_type, command.source_addr_args},
            ttnn::ccl::cmd::SRC_DEST_TYPE::SRC,
            rt_args_out);
        last_src_addr_type = {command.source_addr_type, command.source_addr_args};

        // populate the dest_addr_type
        num_ccl_command_args_added += generate_ccl_address_info_command_args(
            last_dest_addr_type,
            {command.dest_addr_type, command.dest_addr_args},
            ttnn::ccl::cmd::SRC_DEST_TYPE::DEST,
            rt_args_out);
        last_dest_addr_type = {command.dest_addr_type, command.dest_addr_args};

        // populate the core_desc_type
        num_ccl_command_args_added += generate_ccl_core_descriptor_info_command_args(
            last_core_descriptor, {command.core_desc_type, command.core_desc_args}, rt_args_out);
        last_core_descriptor = {command.core_desc_type, command.core_desc_args};

        switch (command.command_code) {
            case ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR:
            case ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB: {
                auto const& current_tensor_slice =
                    std::get<ttnn::ccl::cmd::CclCommandStreamTensorSlice>(command.command_args);
                num_ccl_command_args_added +=
                    generate_ccl_tensor_slice_command_args(last_tensor_slice, current_tensor_slice, rt_args_out);
                last_tensor_slice = current_tensor_slice;
            } break;

            case ttnn::ccl::cmd::CclCommandCode::RAW_INLINE_WRITE_BYTES:
                num_ccl_command_args_added += generate_ccl_raw_inline_write_command_args(
                    std::get<ttnn::ccl::cmd::CclCommandInlineReadWrite>(command.command_args), rt_args_out);
                break;

            case ttnn::ccl::cmd::CclCommandCode::ATOMIC_INC:
                num_ccl_command_args_added += generate_ccl_atomic_inc_command_args(
                    std::get<ttnn::ccl::cmd::CclCommandAtomicInc>(command.command_args), rt_args_out);
                break;
            case ttnn::ccl::cmd::CclCommandCode::WAIT_VALUE:
                num_ccl_command_args_added += generate_ccl_wait_value_command_args(
                    std::get<ttnn::ccl::cmd::CclCommandWaitValue>(command.command_args), rt_args_out);
                break;

            case ttnn::ccl::cmd::CclCommandCode::NOC_READ_BURST:
                TT_FATAL(fill_args_overrider, "Internal Error: When generating noc read burst command args, an rt args override must be provided so that runtime args can be overridden on re-invocations of the owning operation");
                num_ccl_command_args_added += generate_ccl_noc_transfer_burst_command_args(
                    std::get<ttnn::ccl::cmd::HostCclCommandNocTransferBurst>(command.command_args), tensor_indices->at(tensor_index.value()), *rt_args_overrider_out, rt_args_out);
                break;

            case ttnn::ccl::cmd::CclCommandCode::NOC_WRITE_BURST:
                TT_FATAL(fill_args_overrider, "Internal Error: When generating noc write burst command args, an rt args override must be provided so that runtime args can be overridden on re-invocations of the owning operation");
                num_ccl_command_args_added += generate_ccl_noc_transfer_burst_command_args(
                    std::get<ttnn::ccl::cmd::HostCclCommandNocTransferBurst>(command.command_args), tensor_indices->at(tensor_index.value()), *rt_args_overrider_out, rt_args_out);
                break;

            case ttnn::ccl::cmd::CclCommandCode::FLOW_CONTROLLED_NOC_READ_BURST:
                TT_THROW("Command encoding support for CclCommandCode::FLOW_CONTROLLED_NOC_READ_BURST is unimplemented");
                break;

            case ttnn::ccl::cmd::CclCommandCode::NOC_WRITE_AND_ATOMIC_INC:
                TT_THROW("Command encoding support for CclCommandCode::NOC_WRITE_AND_ATOMIC_INC is unimplemented");
                break;

            case ttnn::ccl::cmd::CclCommandCode::STREAM_EDM_TO_TENSOR:
                TT_THROW(
                    "CCL command STREAM_EDM_TO_TENSOR is not useable, supported, or intended to be supported in CCL "
                    "v2. This command is deprecated.");
                break;
                TT_THROW(
                    "CCL command STREAM_TENSOR_TO_EDM is not useable, supported, or intended to be supported in CCL "
                    "v2. This command is deprecated.");
                break;

            default:
                TT_THROW("Unsupported CCL command code: {}. Support missing", static_cast<int>(command.command_code));
                break;
        }

        // populate the fabric_transfer_type
        // Handled by header
        log_trace(
            tt::LogOp,
            "Emitting command_header at index {}. code={}. fabric_transfer_type={}",
            command_header_rt_arg_index,
            command.command_code,
            command.fabric_transfer_type);
        TT_FATAL(command.command_code != ttnn::ccl::cmd::CclCommandCode::INVALID, "Invalid command code");
        rt_args_out[command_header_rt_arg_index] =
            static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandHeader::to_uint32(ttnn::ccl::cmd::CclCommandHeader{
                command.command_code,
                command.fabric_transfer_args,
                num_ccl_command_args_added,
            }));
        TT_FATAL(
            ttnn::ccl::cmd::CclCommandHeader::from_uint32(rt_args_out[command_header_rt_arg_index]).code !=
                ttnn::ccl::cmd::CclCommandCode::INVALID,
            "Invalid command code");

        const size_t new_rt_args_start_index = rt_args_out.size();
        std::stringstream ss;
        ss << "ccl_send command " << i << " has " << new_rt_args_start_index - old_rt_args_start_index
           << " args starting at arg index: " << old_rt_args_start_index << "\n";
        for (std::size_t j = old_rt_args_start_index; j < new_rt_args_start_index; j++) {
            ss << "\targ " << j << ":" << rt_args_out[j] << "\n";
        }
        log_trace(tt::LogOp, "{}", ss.str());
    }
}

void generate_ccl_slice_sequence_commands(
    std::vector<TensorSlice> const& slices,
    ttnn::ccl::cmd::CclCommandCode command_type,
    std::vector<uint32_t>& args_out) {
    generate_ccl_slice_sequence_commands_impl(
        slices, command_type, args_out, ttnn::ccl::cmd::LocalOnlyCommandDestArgs{});
}
void generate_ccl_slice_sequence_commands(
    std::vector<v2::TensorSlice> const& slices,
    ttnn::ccl::cmd::CclCommandCode command_type,
    std::vector<uint32_t>& args_out,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args) {
    generate_ccl_slice_sequence_commands_impl(slices, command_type, args_out, dest_args);
}

void emit_ccl_send_slice_sequence_commands(std::vector<TensorSlice> const& slices, std::vector<uint32_t>& args_out) {
    generate_ccl_slice_sequence_commands(slices, ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM, args_out);
}
void generate_ccl_read_to_cb_slice_sequence_commands(
    std::vector<v2::TensorSlice> const& slices,
    std::vector<uint32_t>& args_out,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args) {
    generate_ccl_slice_sequence_commands(
        slices, ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB, args_out, dest_args);
}
void generate_ccl_cb_to_tensor_slice_sequence_commands(
    std::vector<v2::TensorSlice> const& slices,
    std::vector<uint32_t>& args_out,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args) {
    generate_ccl_slice_sequence_commands(
        slices, ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR, args_out, dest_args);
}

tt::tt_metal::KernelHandle generate_multi_command_stream_kernel_ct_args(
    Program& program,
    std::vector<uint32_t> const& cb_indices,  // TODO: move to RT arg
    std::vector<Tensor const*> const& tensors,
    CoreRangeSet const& worker_core_range,
    tt::tt_metal::DataMovementConfig datamovement_kernel_config,
    const size_t num_command_streams,
    std::optional<chip_id_t> my_chip_id) {
    TT_FATAL(
        num_command_streams > 0 && num_command_streams <= 2,
        "Invalid number of command streams: {}. Must be 1 or 2",
        num_command_streams);

    log_trace(tt::LogOp, "Generating multi command stream kernel CT args");

    std::ranges::for_each(tensors, [](auto const& t) {
        TT_FATAL(t != nullptr, "Null tensor passed to generate_multi_command_stream_kernel_ct_args");
    });
    if (tensors.size() > 0 && tensors[0]->is_sharded()) {
        datamovement_kernel_config.defines["TENSOR0_SHARDED_MEM_LAYOUT"] = "1";
    }
    if (tensors.size() > 1 && tensors[1]->is_sharded()) {
        datamovement_kernel_config.defines["TENSOR1_SHARDED_MEM_LAYOUT"] = "1";
    }
    if (num_command_streams == 1) {
        // single input so we need to disable the second one
        datamovement_kernel_config.defines["SINGLE_INPUT_MODE"] = "1";
    }
    if (tensors.size() == 2) {
        datamovement_kernel_config.defines["TWO_TENSOR"] = "1";
    } else if (tensors.size() == 1) {
        datamovement_kernel_config.defines["SINGLE_TENSOR"] = "1";
    } else {
        datamovement_kernel_config.defines["NO_TENSOR_MODE"] = "1";
    }
    if (datamovement_kernel_config.defines.size() > 0) {
        log_trace(tt::LogOp, "Command Kernel Defines:");
        for (auto const& [k, v] : datamovement_kernel_config.defines) {
            log_trace(tt::LogOp, "\t{}: {}", k, v);
        }
    }


    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index =
        datamovement_kernel_config.processor == tt::tt_metal::DataMovementProcessor::RISCV_0 ? tt::CB::c_in6 : tt::CB::c_in7;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    log_trace(
        tt::LogOp,
        "Setting up reserved packet header CB for {} processor at CB index {} of size {} and page size {}. Core range: "
        "{}",
        datamovement_kernel_config.processor,
        reserved_packet_header_CB_index,
        num_packet_headers_storable * packet_header_size_bytes,
        packet_header_size_bytes,
        worker_core_range);
    auto reserved_packet_header_CB_handle = CreateCircularBuffer(program, worker_core_range, cb_config);

    {  // CT ARGS
        std::vector<uint32_t> ct_args = {my_chip_id.value_or(0xFFFF), reserved_packet_header_CB_index};
        for (size_t i = 0; i < tensors.size(); i++) {
            std::ranges::copy(
                std::array<uint32_t, 4>{
                    static_cast<uint32_t>(
                        tensors[i]->buffer()->buffer_layout()),  // TODO: refactor out to generate_tensor_ct_args
                    static_cast<uint32_t>(tensors[i]->buffer()->buffer_type()),
                    static_cast<uint32_t>(tensors[i]->layout()),
                    static_cast<uint32_t>(0)},
                std::back_inserter(ct_args));
        }
        for (size_t i = 0; i < tensors.size(); i++) {
            std::ranges::copy(
                ttnn::ccl::emit_address_generator_compile_time_args(*tensors[i]), std::back_inserter(ct_args));
        }

        datamovement_kernel_config.compile_args = ct_args;
        log_trace(tt::LogOp, "\tSenderReader Kernel Defines");
        for (auto const& [k, v] : datamovement_kernel_config.defines) {
            log_trace(tt::LogOp, "\t\t{}: {}", k, v);
        }
        log_trace(tt::LogOp, "\tSenderReader CT Args");
        for (size_t i = 0; i < ct_args.size(); i++) {
            auto const& arg = ct_args[i];
            log_trace(tt::LogOp, "\t\t{}: {}", i, arg);
        }
    }
    // Kernel overflowed with O2
    datamovement_kernel_config.opt_level = tt::tt_metal::KernelBuildOptLevel::Os;
    auto sender_worker_reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader_two_input.cpp",
        worker_core_range,
        datamovement_kernel_config);

    return sender_worker_reader_kernel;
}

static void log_command_stream(ttnn::ccl::cmd::CclHostLowLevelCommandSequence const& commands, size_t tab_level = 0) {
    using namespace ttnn::ccl;
    using namespace ttnn::ccl::cmd;
    size_t index = 0;
    for (auto const& c : commands) {
        index++;
        std::stringstream tabs_ss;
        for (size_t i = 0; i < tab_level; i++) {
            tabs_ss << "\t";
        }

        auto get_addr_args_str = [](std::stringstream& ss, CclCommandAddrArgs const& args) {
            std::visit(
                tt::stl::overloaded{
                    [&ss](CclCommandAddrRelativeAddress const& a) {
                        ss << fmt::format("(relative_address:{})", a.relative_address);
                    },
                    [&ss](CclCommandAddrAbsoluteAddress const& a) {
                        ss << fmt::format("(absolute_address:{})", a.absolute_address);
                    },
                    [&ss](CclCommandAddrSemaphoreId const& a) {
                        ss << fmt::format("(semaphore_id:{})", a.semaphore_id);
                    },
                    [&ss](CclCommandAddrCircularBufferId const& a) {
                        ss << fmt::format("(circular_buffer_id:{})", a.circular_buffer_id);
                    },
                    [&ss](CclCommandAddrNone const& a) { ss << "none"; }},
                args);
        };
        auto get_cmd_args_str = [](std::stringstream& ss, CclCommandArgs const& args) {
            std::visit(
                tt::stl::overloaded{
                    [&ss](CclCommandStreamTensorSlice const& a) {
                        ss << fmt::format(
                            "(shape: (w:{},z:{},y:{},x:{}), slice_shape: (w:{},z:{},y:{},x:{}), slice_offset: "
                            "(w:{},z:{},y:{},x:{}), worker_slice_shape: (w:{},z:{},y:{},x:{}), worker_slice_offset: "
                            "(w:{},z:{},y:{},x:{}))",
                            a.tensor_shape.w,
                            a.tensor_shape.z,
                            a.tensor_shape.y,
                            a.tensor_shape.x,
                            a.tensor_slice_shape.w,
                            a.tensor_slice_shape.z,
                            a.tensor_slice_shape.y,
                            a.tensor_slice_shape.x,
                            a.tensor_slice_offset.w,
                            a.tensor_slice_offset.z,
                            a.tensor_slice_offset.y,
                            a.tensor_slice_offset.x,
                            a.worker_slice_shape.w,
                            a.worker_slice_shape.z,
                            a.worker_slice_shape.y,
                            a.worker_slice_shape.x,
                            a.worker_slice_offset.w,
                            a.worker_slice_offset.z,
                            a.worker_slice_offset.y,
                            a.worker_slice_offset.x);
                    },
                    [&ss](CclCommandAtomicInc const& a) {
                        ss << fmt::format("(val:{}, wrap: {})", a.value, a.wrap_value);
                    },
                    [&ss](CclCommandWaitValue const& a) { ss << fmt::format("(wait_value: {})", a.target_value); },
                    [&ss](CclCommandInlineReadWrite const& a) { ss << fmt::format("(value: {})", a.value); },
                    [&ss](CclCommandReadWrite const& a) { ss << fmt::format("(size_bytes: {})", a.size_bytes); },
                    [&ss](HostCclCommandNocTransferBurst const& a) {
                        ss << fmt::format("(base_addr: {}, n_transfers: {})", a.bank_base_address, a.num_transfers_total);
                    },
                    [&ss](auto const&&) { ss << "ERROR"; }},
                args);
        };

        auto get_core_desc_args_str = [](std::stringstream& ss, CclCommandCoreDescriptorArgs const& args) {
            std::visit(
                tt::stl::overloaded{
                    [&ss](CclCommandCoreDescriptorTypeAddrgen const& a) { ss << fmt::format("(addrgen)"); },
                    [&ss](CclCommandCoreDescriptorTypeLocal const& a) { ss << fmt::format("(local_core)"); },
                    [&ss](CclCommandCoreDescriptorTypeNocXY const& a) { ss << fmt::format("(x:{}, y:{})", a.x, a.y); },
                    [&ss](CclCommandCoreDescriptorTypeMcast const& a) {
                        ss << fmt::format(
                            "(noc0_start_x:{}, noc0_start_y:{}, noc0_end_x:{}, noc0_end_y:{})",
                            a.noc0_start_x,
                            a.noc0_start_y,
                            a.noc0_end_x,
                            a.noc0_end_y);
                    },
                    [&ss](CclCommandCoreDescriptorTypeNone const& a) { ss << fmt::format("(None)"); },
                },
                args);
        };

        auto get_fabric_transfer_args_str = [](std::stringstream& ss, CclCommandDestArgs const& args) {
            std::visit(
                tt::stl::overloaded{
                    [&ss](UnicastCommandDestArgs const& a) {
                        ss << fmt::format(
                            "(distance_in_hops:{}, is_forward_direction:{})",
                            a.distance_in_hops,
                            a.is_forward_direction);
                    },
                    [&ss](MulticastCommandDestArgs const& a) {
                        ss << fmt::format(
                            "(num_targets_forward_direction:{}, num_targets_backward_direction:{})",
                            a.num_targets_forward_direction,
                            a.num_targets_backward_direction);
                    },
                    [&ss](LocalOnlyCommandDestArgs const& a) { ss << fmt::format("(None)"); },
                },
                args);
        };

        std::stringstream cmd_attrs_ss;
        std::stringstream src_attrs_ss;
        std::stringstream dest_attrs_ss;
        std::stringstream core_attrs_ss;
        std::stringstream fabric_attrs_ss;
        get_addr_args_str(src_attrs_ss, c.source_addr_args);
        get_addr_args_str(dest_attrs_ss, c.dest_addr_args);
        get_core_desc_args_str(core_attrs_ss, c.core_desc_args);
        get_fabric_transfer_args_str(fabric_attrs_ss, c.fabric_transfer_args);
        get_cmd_args_str(cmd_attrs_ss, c.command_args);

        log_trace(
            tt::LogOp,
            "{}{}. SRC({})[{}] -> CMD({})[{}] -> DST({})[{}]; CORE({})[{}]; FABRIC({})[{}]",
            tabs_ss.str(),
            index,
            c.source_addr_type,
            src_attrs_ss.str(),
            c.command_code,
            cmd_attrs_ss.str(),
            c.dest_addr_type,
            dest_attrs_ss.str(),
            c.core_desc_type,
            core_attrs_ss.str(),
            c.fabric_transfer_type,
            fabric_attrs_ss.str());
    }
}

std::vector<uint32_t> generate_edm_connection_rt_args(
    const tt::tt_fabric::SenderWorkerAdapterSpec& connection_info,
    chip_id_t chip_id,
    Program &program,
    CoreRangeSet worker_cores) {
    std::vector<uint32_t> new_rt_args;
    auto worker_flow_control_semaphore_id = CreateSemaphore(program, worker_cores, 0);
    auto worker_teardown_semaphore_id = CreateSemaphore(program, worker_cores, 0);
    auto worker_buffer_index_semaphore_id = CreateSemaphore(program, worker_cores, 0);
    tt::tt_fabric::append_worker_to_fabric_edm_sender_rt_args(
        connection_info,
        chip_id,
        worker_cores,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        new_rt_args);

    return new_rt_args;
}

void generate_multi_input_command_stream_kernel_rt_args(
    Program& program,
    tt::tt_metal::KernelHandle kernel_id,
    std::vector<Tensor const*> const& tensors,
    std::vector<size_t> const& page_sizes,
    IDevice* device,
    uint32_t num_pages_per_edm_buffer,  // TODO: get from fabric
    CoreRangeSet const& worker_core_range,
    ttnn::ccl::cmd::CclHostLowLevelCommandSequence const& ccl_command_stream0,
    std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence> const& ccl_command_stream1,
    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> const& forward_fabric_connections,
    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> const& backward_fabric_connections,
    std::optional<std::unordered_map<const Tensor*, IDevice*>> const& tensor_device_override,
    std::optional<std::vector<size_t>> const& tensor_indices,
    ttnn::ccl::tensor_address_runtime_args_overrider *rt_args_overrider) {

    bool fill_args_overrider = rt_args_overrider != nullptr;

    if (fill_args_overrider) {
        TT_FATAL(tensor_indices.has_value(), "Internal Error. Tensor indices must be provided when using rt_args_overrider");
        const size_t tensor_count = std::count_if(tensors.begin(), tensors.end(), [](Tensor const* t) { return t != nullptr; });
        TT_FATAL(tensor_indices.value().size() == tensor_count, "Internal Error. Tensor indices must match the number of tensors");
        for (auto tensor_index : tensor_indices.value()) {
            while (rt_args_overrider->size() <= tensor_index) {
                rt_args_overrider->add_tensor();
            }
        }
    }

    // TODO: see if we can pull the kernel defines to understand if we built the kernel in single command stream mode
    log_trace(
        tt::LogOp,
        "Generating multi command stream kernel RT args for kernel {} on core(s): {}",
        kernel_id,
        worker_core_range);
    log_trace(tt::LogOp, "Command stream 0:");
    log_command_stream(ccl_command_stream0, 1);
    if (ccl_command_stream1) {
        log_trace(tt::LogOp, "Command stream 1:");
        log_command_stream(ccl_command_stream1.value(), 1);
    }

    std::vector<const std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand>*> command_streams = {
        &ccl_command_stream0};
    if (ccl_command_stream1.has_value()) {
        command_streams.push_back(&ccl_command_stream1.value());
    }

    // RT ARGS
    const size_t num_command_streams = command_streams.size();
    TT_FATAL(
        tensors.size() <= num_command_streams,
        "Current CCL Command Processor kernel only supports a 1-to-1 mapping between command streams and tensors. "
        "Switching between tensors within a command stream is future work");
    TT_FATAL(page_sizes.size() == tensors.size(), "Number of page sizes must match with the number of tensors");
    auto command_stream_start_arg_indices = std::vector<size_t>(num_command_streams, 0);
    std::vector<uint32_t> rt_args;
    rt_args.reserve(200);
    for (size_t i = 0; i < tensors.size(); i++) {
        if (tensors[i]) {
            if (fill_args_overrider) {
                rt_args_overrider->add_runtime_arg_index(tensor_indices.value()[i], rt_args.size());
            }
            rt_args.push_back(tensors[i]->buffer()->address());
        } else {
            // take up the rt arg with filler value  in case user built a kernel across a core range
            // set with multiple command streams/tensors, but this particular core doesn't actualy need/use
            // both tensors/command streams
            rt_args.push_back(0xdeaddead);
        }
    }
    for (size_t i = 0; i < num_command_streams; i++) {
        rt_args.push_back(command_streams[i]->size());  // in0_read_command_slices
        command_stream_start_arg_indices[i] = rt_args.size();
        rt_args.push_back(0);  // in0_command_start_offset
    }
    rt_args.push_back(num_pages_per_edm_buffer);
    TT_FATAL(tensors.size() == page_sizes.size(), "Number of pages must match with the number of tensors");
    for (size_t i = 0; i < tensors.size(); i++) {
        if (tensors[i]) {
            rt_args.push_back(page_sizes[i]);  // in0
        } else {
            rt_args.push_back(0xdeaddead);
        }
    }

    for (Tensor const* t : tensors) {
        if (t) {
            bool rt_args_enabled = true;
            rt_args.push_back(rt_args_enabled);
            if (tensor_device_override.has_value() and
                tensor_device_override.value().find(t) != tensor_device_override.value().end()) {
                std::ranges::copy(
                    ttnn::ccl::emit_address_generator_runtime_args(tensor_device_override->at(t), *t),
                    std::back_inserter(rt_args));
            } else {
                std::ranges::copy(
                    ttnn::ccl::emit_address_generator_runtime_args(t->buffer()->device(), *t),
                    std::back_inserter(rt_args));
            }
        } else {
            bool rt_args_enabled = false;
            rt_args.push_back(rt_args_enabled);
        }
        // else: Interleaved addrgen passes no additional args - we specify interleaved addrgen as the default
    }

    rt_args.push_back(forward_fabric_connections.has_value());
    if (forward_fabric_connections.has_value()) {
        const auto new_rt_args =
            generate_edm_connection_rt_args(*forward_fabric_connections, device->id(), program, worker_core_range);
        std::copy(new_rt_args.begin(), new_rt_args.end(), std::back_inserter(rt_args));
    }
    rt_args.push_back(backward_fabric_connections.has_value());
    if (backward_fabric_connections.has_value()) {
        const auto new_rt_args =
            generate_edm_connection_rt_args(*backward_fabric_connections, device->id(), program, worker_core_range);
        std::copy(new_rt_args.begin(), new_rt_args.end(), std::back_inserter(rt_args));
    }

    for (size_t i = 0; i < num_command_streams; i++) {
        // Update the command stream start arg index argument to point to here (i.e. where
        // this command stream's commands will start)
        rt_args[command_stream_start_arg_indices[i]] = rt_args.size();
        generate_ccl_command_stream_to_kernel_args((*command_streams[i]), i, tensor_indices, rt_args_overrider, rt_args);
    }

    log_trace(tt::LogOp, "\tMulti-input command processor RT Args");
    for (size_t i = 0; i < rt_args.size(); i++) {
        auto const& arg = rt_args[i];
        log_trace(tt::LogOp, "\t\t{}: {}", i, arg);
    }
    tt::tt_metal::SetRuntimeArgs(program, kernel_id, worker_core_range, rt_args);

}

void generate_multi_input_command_stream_kernel_rt_args(
    Program& program,
    tt::tt_metal::KernelHandle kernel_id,
    std::vector<Tensor const*> const& tensors,
    std::vector<size_t> const& page_sizes,
    IDevice* device,
    uint32_t link,
    uint32_t num_pages_per_edm_buffer,  // TODO: get from fabric
    CoreRangeSet const& worker_core_range,
    ttnn::ccl::cmd::CclHostLowLevelCommandSequence const& ccl_command_stream0,
    std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence> const& ccl_command_stream1,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    std::optional<std::unordered_map<const Tensor*, IDevice*>> const& tensor_device_override,
    std::optional<std::vector<size_t>> const& tensor_indices,
    ttnn::ccl::tensor_address_runtime_args_overrider *rt_args_overrider) {

    bool fill_args_overrider = rt_args_overrider != nullptr;

    if (fill_args_overrider) {
        TT_FATAL(tensor_indices.has_value(), "Internal Error. Tensor indices must be provided when using rt_args_overrider");
        const size_t tensor_count = std::count_if(tensors.begin(), tensors.end(), [](Tensor const* t) { return t != nullptr; });
        TT_FATAL(tensor_indices.value().size() == tensor_count, "Internal Error. Tensor indices must match the number of tensors");
        for (auto tensor_index : tensor_indices.value()) {
            while (rt_args_overrider->size() <= tensor_index) {
                rt_args_overrider->add_tensor();
            }
        }
    }

    // TODO: see if we can pull the kernel defines to understand if we built the kernel in single command stream mode
    log_trace(
        tt::LogOp,
        "Generating multi command stream kernel RT args for kernel {} on core(s): {}",
        kernel_id,
        worker_core_range);
    log_trace(tt::LogOp, "Command stream 0:");
    log_command_stream(ccl_command_stream0, 1);
    if (ccl_command_stream1) {
        log_trace(tt::LogOp, "Command stream 1:");
        log_command_stream(ccl_command_stream1.value(), 1);
    }

    std::vector<const std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand>*> command_streams = {
        &ccl_command_stream0};
    if (ccl_command_stream1.has_value()) {
        command_streams.push_back(&ccl_command_stream1.value());
    }

    // RT ARGS
    const size_t num_command_streams = command_streams.size();
    TT_FATAL(
        tensors.size() <= num_command_streams,
        "Current CCL Command Processor kernel only supports a 1-to-1 mapping between command streams and tensors. "
        "Switching between tensors within a command stream is future work");
    TT_FATAL(page_sizes.size() == tensors.size(), "Number of page sizes must match with the number of tensors");
    auto command_stream_start_arg_indices = std::vector<size_t>(num_command_streams, 0);
    std::vector<uint32_t> rt_args;
    rt_args.reserve(200);
    for (size_t i = 0; i < tensors.size(); i++) {
        if (tensors[i]) {
            if (fill_args_overrider) {
                rt_args_overrider->add_runtime_arg_index(tensor_indices.value()[i], rt_args.size());
            }
            rt_args.push_back(tensors[i]->buffer()->address());
        } else {
            // take up the rt arg with filler value  in case user built a kernel across a core range
            // set with multiple command streams/tensors, but this particular core doesn't actualy need/use
            // both tensors/command streams
            rt_args.push_back(0xdeaddead);
        }
    }
    for (size_t i = 0; i < num_command_streams; i++) {
        rt_args.push_back(command_streams[i]->size());  // in0_read_command_slices
        command_stream_start_arg_indices[i] = rt_args.size();
        rt_args.push_back(0);  // in0_command_start_offset
    }
    rt_args.push_back(num_pages_per_edm_buffer);
    TT_FATAL(tensors.size() == page_sizes.size(), "Number of pages must match with the number of tensors");
    for (size_t i = 0; i < tensors.size(); i++) {
        if (tensors[i]) {
            rt_args.push_back(page_sizes[i]);  // in0
        } else {
            rt_args.push_back(0xdeaddead);
        }
    }

    for (Tensor const* t : tensors) {
        if (t) {
            bool rt_args_enabled = true;
            rt_args.push_back(rt_args_enabled);
            if (tensor_device_override.has_value() and
                tensor_device_override.value().find(t) != tensor_device_override.value().end()) {
                std::ranges::copy(
                    ttnn::ccl::emit_address_generator_runtime_args(tensor_device_override->at(t), *t),
                    std::back_inserter(rt_args));
            } else {
                std::ranges::copy(
                    ttnn::ccl::emit_address_generator_runtime_args(t->buffer()->device(), *t),
                    std::back_inserter(rt_args));
            }
        } else {
            bool rt_args_enabled = false;
            rt_args.push_back(rt_args_enabled);
        }
        // else: Interleaved addrgen passes no additional args - we specify interleaved addrgen as the default
    }
    rt_args.push_back(forward_device.has_value() and forward_device.value());
    auto worker_core = corerange_to_cores(worker_core_range).at(0);
    if (forward_device.has_value() and forward_device.value()) {
        const auto device_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id());
        const auto forward_device_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
        tt::tt_fabric::append_fabric_connection_rt_args(device_fabric_node_id, forward_device_fabric_node_id, link, program, {worker_core}, rt_args);
    }

    rt_args.push_back(backward_device.has_value() and backward_device.value());
    if (backward_device.has_value() and backward_device.value()) {
        const auto device_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id());
        const auto backward_device_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
        tt::tt_fabric::append_fabric_connection_rt_args(device_fabric_node_id, backward_device_fabric_node_id, link, program, {worker_core}, rt_args);
    }

    for (size_t i = 0; i < num_command_streams; i++) {
        // Update the command stream start arg index argument to point to here (i.e. where
        // this command stream's commands will start)
        rt_args[command_stream_start_arg_indices[i]] = rt_args.size();
        generate_ccl_command_stream_to_kernel_args((*command_streams[i]), i, tensor_indices, rt_args_overrider, rt_args);
    }

    log_trace(tt::LogOp, "\tMulti-input command processor RT Args");
    for (size_t i = 0; i < rt_args.size(); i++) {
        auto const& arg = rt_args[i];
        log_trace(tt::LogOp, "\t\t{}: {}", i, arg);
    }
    tt::tt_metal::SetRuntimeArgs(program, kernel_id, worker_core_range, rt_args);

}

ttnn::ccl::cmd::CclHostLowLevelCommandSequence build_ccl_cmd_proc_teardown_commands(
    Program& program,
    IDevice* device,
    IDevice* forward_device,
    size_t line_size,
    size_t line_index,
    std::vector<tt::tt_fabric::edm_termination_info_t> const& edm_termination_infos,
    ccl::SyncModeSpec const& sync_details,
    ccl::EdmLineFabricOpInterface& fabric_interface) {
    TT_FATAL(sync_details.num_signals == 1, "Only one signal is supported for CCL command processor teardown");
    TT_FATAL(sync_details.sem_ids.size() == 1, "Only one signal is supported for CCL command processor teardown");
    TT_FATAL(sync_details.wait_counts.size() == 1, "Only one signal is supported for CCL command processor teardown");

    auto local_wait_sem_id = sync_details.sem_ids.at(0);
    auto remote_sem_id = sync_details.sem_ids.at(0);

    ttnn::ccl::cmd::CclHostLowLevelCommandSequence teardown_cmd_stream = {
        // + 1 because we need to wait for our left/backward neighbour to tell us it's safe to teardown (because they
        // are
        // done tearing down - we teardown from first to last)
        cmd::uops::local_semaphore_wait(local_wait_sem_id, sync_details.wait_counts.at(0) + (line_index != 0)),
    };

    // If there is a forward connection, notify that neighbour that they can teardown
    if (forward_device != nullptr) {
        auto remote_worker_noc0_core = forward_device->worker_core_from_logical_core(sync_details.core);
        teardown_cmd_stream.push_back(cmd::uops::fabric_unicast_semaphore_inc(
            remote_sem_id,
            ttnn::ccl::cmd::CclCommandAtomicInc{1},
            remote_worker_noc0_core.x,
            remote_worker_noc0_core.y,
            ttnn::ccl::cmd::UnicastCommandDestArgs{1, true}));
    }

    // Finally teardown our local chip's fabric endpoint(s)
    if (edm_termination_infos.size() > 0) {
        log_trace(tt::LogOp, "{} termination infos", edm_termination_infos.size());
    }
    for (auto& info : edm_termination_infos) {
        if (info.distance == 0) {
            log_trace(
                tt::LogOp,
                "Adding local chip fabric teardown command for termination address {},",
                info.termination_addr);
            teardown_cmd_stream.push_back(cmd::uops::local_chip_noc_absolute_address_semaphore_inc(
                info.edm_noc_x, info.edm_noc_y, info.termination_addr, 1));
        } else {
            log_trace(
                tt::LogOp,
                "Adding remote chip fabric teardown command for termination address {} of distance {}",
                info.termination_addr,
                info.distance);
            teardown_cmd_stream.push_back(ttnn::ccl::cmd::uops::fabric_unicast_absolute_address_semaphore_inc(
                ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress{info.termination_addr},
                ttnn::ccl::cmd::CclCommandAtomicInc{1},
                info.edm_noc_x,
                info.edm_noc_y,
                ttnn::ccl::cmd::UnicastCommandDestArgs{info.distance, true}));
        }
    }

    return teardown_cmd_stream;
}

void build_sync_kernels(
    IDevice* device,
    Program& program,
    ccl::SyncModeSpec const& sync_details,
    bool terminate_fabric,
    ccl::EdmLineFabricOpInterface& fabric_interface) {
    auto const sync_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_wait_completion.cpp",
        sync_details.core,
        tt::tt_metal::ReaderDataMovementConfig({sync_details.num_signals, terminate_fabric}));

    std::vector<uint32_t> rt_args;
    rt_args.reserve(sync_details.num_signals * 2);
    for (size_t i = 0; i < sync_details.num_signals; ++i) {
        rt_args.push_back(sync_details.sem_ids[i]);
        rt_args.push_back(sync_details.wait_counts[i]);
    }

    if (terminate_fabric) {
        auto termination_infos = fabric_interface.generate_local_chip_fabric_termination_infos(device);
        rt_args.push_back(termination_infos.size());
        for (auto& info : termination_infos) {
            if (info.distance != 0) {
                continue;
            }
            rt_args.push_back(info.termination_addr);
            rt_args.push_back(info.edm_noc_x);
            rt_args.push_back(info.edm_noc_y);
        }
    }

    tt::tt_metal::SetRuntimeArgs(program, sync_kernel_id, sync_details.core, rt_args);
}

std::vector<uint32_t> CCLWorkerArgBuilder::generate_sender_reader_kernel_rt_args(
    ttnn::ccl::InterleavedTensorWorkerSlice worker_slice,
    std::size_t operating_dim,
    uint32_t num_pages_per_packet,
    uint32_t worker_slice_index) const {
    const std::size_t num_commands_expected = this->input_tensor_partition.partition_size;

    auto const& tensor_shape = worker_slice.tensor_shape;
    auto const& tensor_slice_shape = worker_slice.tensor_slice_shape;

    auto num_slices = input_tensor_partition.partition_size;
    auto start_slice_index = input_tensor_partition.partition_index;
    std::int64_t end_slice_index_exclusive = input_tensor_partition.partition_index + 1;

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
        worker_slice_index);
    TT_ASSERT(num_commands_expected == slices.size());

    // If we are on device zero, we send n-1 chunks in ascending order
    auto& input_tensor = this->op_config.get_input_tensor(0);
    TT_ASSERT(input_tensor.padded_shape().size() == 4, "Only 4D tensors are supported for ccl");
    ttnn::ccl::Shape4D<uint32_t> input_tensor_shape = {
        input_tensor.padded_shape()[0],
        input_tensor.padded_shape()[1],
        input_tensor.padded_shape()[2],
        input_tensor.padded_shape()[3]};

    std::vector<uint32_t> args = {
        static_cast<uint32_t>(input_tensor.buffer()->address()),
        static_cast<uint32_t>(slices.size()),
        num_pages_per_packet,
        this->op_config.get_page_size()};
    std::size_t logged_arg_idx = 0;
    log_trace(tt::LogOp, "ccl_send_reader arg[{}]: buffer_address = {}", logged_arg_idx, args[logged_arg_idx]);
    logged_arg_idx++;
    log_trace(tt::LogOp, "ccl_send_reader arg[{}]: num_commands = {}", logged_arg_idx, args[logged_arg_idx]);
    logged_arg_idx++;
    log_trace(tt::LogOp, "ccl_send_reader arg[{}]: pages_per_packet {}", logged_arg_idx, args[logged_arg_idx]);
    logged_arg_idx++;
    log_trace(tt::LogOp, "ccl_send_reader arg[{}]: page_size {}", logged_arg_idx, args[logged_arg_idx]);
    logged_arg_idx++;

    auto const& addr_gen_rt_args = ttnn::ccl::legacy_emit_address_generator_runtime_args(this->device, input_tensor);
    std::ranges::copy(addr_gen_rt_args, std::back_inserter(args));
    for (auto const& arg : addr_gen_rt_args) {
        log_trace(tt::LogOp, "ccl_send_reader arg[{}]: addr_gen_rt_args[] {}", logged_arg_idx, args[logged_arg_idx]);
        logged_arg_idx++;
    }

    log_trace(tt::LogOp, "ccl_send_reader Generating {} ccl send commands", slices.size());
    emit_ccl_send_slice_sequence_commands(slices, args);

    log_trace(tt::LogOp, "ccl_send_reader Sender Worker has {} RT Args: {}", args.size(), args);

    return args;
}

std::vector<uint32_t> CCLWorkerArgBuilder::generate_sender_reader_kernel_ct_args() const {
    auto const& input_tensor = this->op_config.get_input_tensor(0);
    std::vector<uint32_t> args = {
        static_cast<uint32_t>(input_tensor.memory_config().memory_layout()),  // tensor memory layout
        static_cast<uint32_t>(input_tensor.buffer()->buffer_type()),        // buffer type
        static_cast<uint32_t>(input_tensor.layout()),                       // page layout
        static_cast<uint32_t>(tt::CB::c_in0)                                // cb_id
    };

    auto const& addr_gen_rt_args = ttnn::ccl::emit_address_generator_compile_time_args(input_tensor);
    std::ranges::copy(addr_gen_rt_args, std::back_inserter(args));

    return args;
}

std::vector<uint32_t> CCLWorkerArgBuilder::generate_sender_writer_kernel_ct_args() const {
    auto const& output_tensor = this->op_config.get_output_tensor(0);
    std::vector<uint32_t> args = {
        static_cast<uint32_t>(output_tensor.memory_config().memory_layout()),  // tensor memory layout
        static_cast<uint32_t>(output_tensor.buffer()->buffer_type()),        // buffer type
        static_cast<uint32_t>(output_tensor.layout()),                       // page layout
        static_cast<uint32_t>(tt::CB::c_in0)                                 // cb_id
    };

    auto const& addr_gen_rt_args = ttnn::ccl::emit_address_generator_compile_time_args(output_tensor);
    std::ranges::copy(addr_gen_rt_args, std::back_inserter(args));

    return args;
}

bool can_command_stream_be_lowered_to_noc_commands(const Tensor& tensor) {
    static constexpr size_t baseline_arg_count = 12;
    // approximately... this is only very rough estimate until unlimited command stream length is enabled
    static constexpr size_t args_per_noc_command = 4;
    static constexpr size_t max_noc_commands = 256;
    size_t page_num_elements =
        tensor.layout() == Layout::TILE ? tensor.tensor_spec().tile().get_tile_hw(): tensor.padded_shape()[-1];
    size_t num_tensor_pages = tensor.padded_shape().volume() / page_num_elements;

    // Interleaved tensors are currently not iterable on host so we can't resolve the page locations
    return tensor.is_sharded() &&
           (num_tensor_pages * args_per_noc_command + baseline_arg_count < max_noc_commands);
}


}  // namespace ttnn::ccl::worker_detail
