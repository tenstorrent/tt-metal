// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"

#ifdef DEBUG_PRINT_ENABLED
#include "debug/dprint.h"
#endif

#include <cstdint>
#include <type_traits>

namespace ttnn {
namespace ccl {


template<typename T>
auto build_from_args(std::size_t &rt_arg_idx) -> T {
    static_assert(!std::is_same<T, T>::value, "This base template cannot be instantiated. Please provide a specialization.");
}
template<typename T>
constexpr std::size_t ct_args_consumed(std::size_t ct_arg_offset, std::size_t &rt_arg_idx) {
    static_assert(!std::is_same<T, T>::value, "This base template cannot be instantiated. Please provide a specialization.");
    return 0;
}
template<typename T>
constexpr std::size_t ct_args_consumed() {
    static_assert(!std::is_same<T, T>::value, "This base template cannot be instantiated. Please provide a specialization.");
    return 0;
}

template <>
auto build_from_args<Shape4D<uint32_t>>(std::size_t &rt_arg_idx) -> Shape4D<uint32_t> {
    auto w = get_arg_val<uint32_t>(rt_arg_idx++);
    auto z = get_arg_val<uint32_t>(rt_arg_idx++);
    auto y = get_arg_val<uint32_t>(rt_arg_idx++);
    auto x = get_arg_val<uint32_t>(rt_arg_idx++);

    return Shape4D<uint32_t>{w, z, y, x};
}

namespace cmd {

CclCommandHeader update_command_tensor(std::size_t &arg_idx, CclCommandTensor &cmd_tensor) {
    auto cmd = CclCommandHeader::from_uint32(get_arg_val<uint32_t>(arg_idx++));
    #ifdef DEBUG_PRINT_ENABLED
    DPRINT << "CMD (code=" << (uint32_t)cmd.code << ", dst_t=" << (uint32_t)cmd.dest_type << ", arg_count=" << (uint32_t)cmd.arg_count << ")\n";
    #endif

    for (size_t i = 0; i < cmd.arg_count; i++) {

        // Note that we choose to reinterpret our pointers as volatile so that in the future we can add streaming
        // of additional commands from some backing memory (e.g. dram or L1), potentially by another core, without
        // having to track down this code and add volatile casts later (which would be a potentially tricky bug to
        // root cause).
        const CclCommandArgHeader command_arg_header = CclCommandArgHeader::from_uint32(get_arg_val<uint32_t>(arg_idx++));
        const CclCommandArgCode command_arg_code = command_arg_header.code;
        switch (command_arg_code) {
            case CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::unpack(reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor.tensor_shape);
                #ifdef DEBUG_PRINT_ENABLED
                DPRINT << "Updating tensor shape: (w=" << (uint32_t)cmd_tensor.tensor_shape.w << ", z=" << (uint32_t)cmd_tensor.tensor_shape.z << ", y=" << (uint32_t)cmd_tensor.tensor_shape.y << ", x=" << (uint32_t)cmd_tensor.tensor_shape.x << ")\n";
                #endif
                arg_idx += CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::size_in_words();
                break;
            case CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::unpack(reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor.tensor_slice_shape);
                #ifdef DEBUG_PRINT_ENABLED
                DPRINT << "Updating tensor slice shape: (w=" << (uint32_t)cmd_tensor.tensor_slice_shape.w << ", z=" << (uint32_t)cmd_tensor.tensor_slice_shape.z << ", y=" << (uint32_t)cmd_tensor.tensor_slice_shape.y << ", x=" << (uint32_t)cmd_tensor.tensor_slice_shape.x << ")\n";
                #endif
                arg_idx += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::size_in_words();
                break;
            case CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::unpack(
                    reinterpret_cast<volatile uint32_t *>(get_arg_addr(arg_idx)), cmd_tensor.tensor_slice_offset);
                #ifdef DEBUG_PRINT_ENABLED
                DPRINT << "Updating tensor slice offset: (w=" << (uint32_t)cmd_tensor.tensor_slice_offset.w << ", z=" << (uint32_t)cmd_tensor.tensor_slice_offset.z << ", y=" << (uint32_t)cmd_tensor.tensor_slice_offset.y << ", x=" << (uint32_t)cmd_tensor.tensor_slice_offset.x << ")\n";
                #endif
                arg_idx += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();
                break;
            case CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::unpack(reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor.worker_start_offset_in_slice);
                #ifdef DEBUG_PRINT_ENABLED
                DPRINT << "Updating worker start offset in slice: (w=" << (uint32_t)cmd_tensor.worker_start_offset_in_slice.w << ", z=" << (uint32_t)cmd_tensor.worker_start_offset_in_slice.z << ", y=" << (uint32_t)cmd_tensor.worker_start_offset_in_slice.y << ", x=" << (uint32_t)cmd_tensor.worker_start_offset_in_slice.x << ")\n";
                #endif
                arg_idx += CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::size_in_words();
                break;
            case CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE:
                CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::unpack(reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor.worker_pages_per_slice);
                #ifdef DEBUG_PRINT_ENABLED
                DPRINT << "Updating worker pages per slice: " << (uint32_t)cmd_tensor.worker_pages_per_slice << "\n";
                #endif
                arg_idx += CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::size_in_words();
                break;
            case CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::unpack(reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor);
                #ifdef DEBUG_PRINT_ENABLED
                DPRINT << "Updating full tensor slice spec: (tensor_shape: w=" << (uint32_t)cmd_tensor.tensor_shape.w << ", z=" << (uint32_t)cmd_tensor.tensor_shape.z << ", y=" << (uint32_t)cmd_tensor.tensor_shape.y << ", x=" << (uint32_t)cmd_tensor.tensor_shape.x << ")\n";
                #endif
                arg_idx += CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::size_in_words();
                break;
            default:
                ASSERT(false);
        };
    }

    return cmd;
}




} // namespace cmd

} // namespace ccl
} // namespace ttnn
