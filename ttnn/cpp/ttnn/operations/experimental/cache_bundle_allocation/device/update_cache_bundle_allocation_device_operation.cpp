// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "update_cache_bundle_allocation_device_operation.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/cache_bundle_allocation/cache_bundle_allocation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
// A 4 KiB window amortizes stack transfers while keeping scratch independent of pool capacity.
namespace {
constexpr uint32_t kFreeListWindowBytes = 4096;
constexpr uint32_t kMaxScratchBytes = 512 * 1024;

std::array<const Tensor*, 4> metadata(const CacheBundleAllocationInputs& t) {
    return {&t.page_table, &t.allocated_pages, &t.free_list, &t.free_count};
}
std::array<uint32_t, 4> scratch_sizes(const CacheBundleAllocationInputs& t) {
    const auto tensors = metadata(t);
    std::array<uint32_t, 4> sizes{};
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto row_bytes = tensors[i]->buffer()->aligned_page_size();
        sizes[i] = i == 2 ? std::min<DeviceAddr>(row_bytes, kFreeListWindowBytes) : row_bytes;
    }
    return sizes;
}
}  // namespace

void UpdateCacheBundleAllocationDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a, const tensor_args_t& t) {
    // Buffer identity/placement can change without changing the program hash.
    // Metadata contents are caller-owned invariants; validation does not read them back from the device.
    const auto tensors = metadata(t);
    for (size_t i = 0; i < tensors.size(); ++i) {
        TT_FATAL(
            tensors[i]->storage_type() == StorageType::DEVICE && tensors[i]->buffer(), "Metadata must be on device");
        TT_FATAL(tensors[i]->device() == t.page_table.device(), "Metadata must use the same device");
        for (size_t j = 0; j < i; ++j) {
            TT_FATAL(tensors[i]->buffer() != tensors[j]->buffer(), "Metadata buffers must not alias");
        }
    }
    TT_FATAL(a.page_size > 0, "page_size must be positive");
    TT_FATAL(a.slot_id < t.page_table.logical_shape()[0], "slot_id is out of range");
    TT_FATAL(a.actual_start <= a.actual_end, "actual_start must not exceed actual_end");
    TT_FATAL(
        uint64_t(a.actual_end) <= uint64_t(t.page_table.logical_shape()[1]) * a.page_size,
        "actual_end exceeds page_table capacity");
}

void UpdateCacheBundleAllocationDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    const auto tensors = metadata(t);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = *tensors[i];
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE && tensor.buffer(), "Metadata must be on device");
        TT_FATAL(tensor.device() == t.page_table.device(), "Metadata must use the same device");
        TT_FATAL(tensor.layout() == Layout::ROW_MAJOR, "Metadata must be ROW_MAJOR");
        TT_FATAL(tensor.memory_config() == DRAM_MEMORY_CONFIG, "Metadata must be interleaved DRAM");
        TT_FATAL(tensor.logical_shape().rank() == 2, "Metadata must be rank 2");
        TT_FATAL(tensor.dtype() == DataType::UINT32, "Incorrect metadata dtype: all metadata must be UINT32");
        TT_FATAL(
            tensor.buffer()->aligned_page_size() <= std::numeric_limits<uint32_t>::max(),
            "Metadata row exceeds the 32-bit NoC byte-offset range");
    }
    const auto& pt = t.page_table.logical_shape();
    const auto& fl = t.free_list.logical_shape();
    TT_FATAL(pt[0] > 0 && pt[1] > 0, "page_table dimensions must be positive");
    TT_FATAL(fl[0] > 0 && fl[1] > 0, "free_list dimensions must be positive");
    TT_FATAL(
        t.allocated_pages.logical_shape() == ttnn::Shape({1, pt[0]}), "allocated_pages must have shape [1, slots]");
    TT_FATAL(t.free_count.logical_shape() == ttnn::Shape({1, fl[0]}), "free_count must have shape [1, SP]");
    // One table row, two counter rows, and a bounded free-list window.
    uint64_t scratch_bytes = 0;
    for (const auto bytes : scratch_sizes(t)) {
        scratch_bytes += bytes;
    }
    TT_FATAL(scratch_bytes <= kMaxScratchBytes, "Metadata rows require more than 512 KiB of L1 scratch");
    validate_on_program_cache_hit(a, t);
}

TensorSpec UpdateCacheBundleAllocationDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.page_table.tensor_spec();
}

Tensor UpdateCacheBundleAllocationDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.page_table;
}

ttsl::hash::hash_t UpdateCacheBundleAllocationDeviceOperation::compute_program_hash(
    const operation_attributes_t& a, const tensor_args_t& t) {
    // Slot and token positions are runtime arguments, including on program-cache hits.
    return operation::hash_operation<UpdateCacheBundleAllocationDeviceOperation>(a.page_size, t);
}

ProgramDescriptor CacheBundleAllocationProgramFactory::create_descriptor(
    const CacheBundleAllocationParams& a, const CacheBundleAllocationInputs& t, Tensor&) {
    ProgramDescriptor desc;
    const CoreRangeSet cores({CoreRange({0, 0}, {0, 0})});
    std::vector<uint32_t> ct = {t.free_list.logical_shape()[0], a.page_size};
    auto tensors = metadata(t);
    const auto sizes = scratch_sizes(t);
    ct.insert(ct.end(), sizes.begin(), sizes.end());
    ct.push_back(t.free_list.buffer()->aligned_page_size());
    for (uint8_t i = 0; i < sizes.size(); ++i) {
        const uint32_t bytes = sizes[i];
        desc.cbs.push_back(CBDescriptor{
            .total_size = bytes,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = i, .data_format = tt::DataFormat::UInt32, .page_size = bytes}}}});
    }
    for (const auto* tensor : tensors) {
        TensorAccessorArgs(tensor->buffer()).append_to(ct);
    }
    KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/cache_bundle_allocation/kernels/"
        "update_cache_bundle_allocation.cpp";
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = cores;
    kernel.compile_time_args = std::move(ct);
    kernel.config = ReaderConfigDescriptor{};
    kernel.emplace_runtime_args(
        CoreCoord{0, 0},
        {t.page_table.buffer(),
         t.allocated_pages.buffer(),
         t.free_list.buffer(),
         t.free_count.buffer(),
         a.slot_id,
         a.actual_start,
         a.actual_end});
    desc.kernels.push_back(std::move(kernel));
    return desc;
}

void CacheBundleAllocationProgramFactory::override_runtime_arguments(
    Program& program,
    const CacheBundleAllocationParams& a,
    const CacheBundleAllocationInputs& t,
    Tensor&,
    const std::optional<ttnn::MeshCoordinate>&) {
    auto& args = GetRuntimeArgs(program, 0, CoreCoord{0, 0});
    const auto tensors = metadata(t);
    for (size_t i = 0; i < tensors.size(); ++i) {
        args[i] = tensors[i]->buffer()->address();
    }
    args[4] = a.slot_id;
    args[5] = a.actual_start;
    args[6] = a.actual_end;
}
}  // namespace ttnn::experimental::prim

namespace ttnn::experimental {
Tensor update_cache_bundle_allocation(
    const Tensor& page_table,
    const Tensor& allocated_pages,
    const Tensor& free_list,
    const Tensor& free_count,
    uint32_t slot_id,
    uint32_t actual_start,
    uint32_t actual_end,
    uint32_t page_size) {
    using Op = prim::UpdateCacheBundleAllocationDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{slot_id, actual_start, actual_end, page_size},
        Op::tensor_args_t{page_table, allocated_pages, free_list, free_count});
}
}  // namespace ttnn::experimental
