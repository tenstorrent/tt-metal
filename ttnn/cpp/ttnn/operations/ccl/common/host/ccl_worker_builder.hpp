// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"

#include <cstdint>

namespace tt {
namespace tt_metal {
inline namespace v0 {

// Forward declarations
class Device;

}  // namespace v0
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace ccl {
class WorkerEdmInterfaceArgs;

namespace worker_detail {

void emit_ccl_send_slice_sequence_commands(std::vector<TensorSlice> const& slices, std::vector<uint32_t>& args_out);

struct CCLWorkerArgBuilder {
    CCLWorkerArgBuilder (
        tt::tt_metal::Device const* device,
        ttnn::ccl::CCLOpConfig const& op_config,
        ttnn::ccl::TensorPartition const& input_tensor_partition,
        ttnn::ccl::TensorPartition const& output_tensor_partition,
        std::size_t operating_dim);

    std::vector<uint32_t> generate_sender_reader_kernel_rt_args(
        ttnn::ccl::InterleavedTensorWorkerSlice worker_slice,
        std::size_t operating_dim,
        uint32_t num_pages_per_packet,
        uint32_t worker_slice_index) const;

    std::vector<uint32_t> generate_sender_writer_kernel_rt_args(
        ttnn::ccl::InterleavedTensorWorkerSlice worker_slice,
        std::size_t operating_dim,
        uint32_t num_pages_per_packet,
        uint32_t worker_slice_index) const;

    std::vector<uint32_t> generate_sender_reader_kernel_ct_args() const;

    std::vector<uint32_t> generate_sender_writer_kernel_ct_args() const;

    tt::tt_metal::Device const*device;
    ttnn::ccl::TensorPartition const input_tensor_partition;
    ttnn::ccl::TensorPartition const output_tensor_partition;
    ttnn::ccl::CCLOpConfig const op_config;
    std::size_t operating_dim;
    bool src_is_dram;
    bool dst_is_dram;
};

} // namespace worker_detail
} // namespace ccl
} // namespace ttnn
