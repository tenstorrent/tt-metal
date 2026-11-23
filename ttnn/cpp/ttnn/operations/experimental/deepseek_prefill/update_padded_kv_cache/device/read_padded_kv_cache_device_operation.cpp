// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "read_padded_kv_cache_device_operation.hpp"
#include "kv_cache_linearization.hpp"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/chip_position/chip_position_table.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

constexpr auto kReadKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/"
    "read_padded_kv_cache.cpp";
constexpr uint32_t kRowsCbIndex = 0;
constexpr uint32_t kReadMetaCbIndex = 1;
constexpr uint32_t kReadMetadataBytes = 16;
// Rows read before they are written back; one batch is in flight per core.
constexpr uint32_t kReadBatchRows = 8;
// Common runtime args the cache-hit path patches: the metadata tensors' raw addresses.
constexpr uint32_t kSlotAddrArg = 6;
constexpr uint32_t kKvActualAddrArg = 7;

using Op = ReadPaddedKvCacheDeviceOperation;

uint32_t chunk_rows_per_chip(const Op::operation_attributes_t& args, const Op::tensor_args_t& tensor_args) {
    return args.chunk_size_global / kv_cache_linear_factor(tensor_args.cache, args.cluster_axis, args.tp_axis);
}

std::optional<std::vector<uint32_t>> whole_mesh_chip_table(
    const Op::operation_attributes_t& args, const Op::tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    return chip_lookup_table(
        *cache.device(), cache.device_storage().get_coords(), [&](const ttnn::MeshCoordinate& coord) {
            return kv_cache_linear_coord(cache, coord, args.cluster_axis, args.tp_axis);
        });
}

// Checks every call runs, hit or miss: the program bakes in none of the metadata values, and the output may be a
// different preallocated tensor each call.
void validate_runtime(const Op::operation_attributes_t& args, const Op::tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(cache.device()->get_view().is_mesh_2d(), "read_padded_kv_cache requires a 2D mesh");
    TT_FATAL(
        !args.cluster_axis.has_value() || *args.cluster_axis < 2,
        "cluster_axis must be 0, 1, or None; got {}",
        args.cluster_axis);
    if (args.tp_axis.has_value()) {
        TT_FATAL(args.cluster_axis.has_value(), "tp_axis requires an explicit cluster_axis");
        TT_FATAL(
            *args.tp_axis < 2 && *args.tp_axis != *args.cluster_axis,
            "tp_axis ({}) must be 0 or 1 and differ from cluster_axis ({})",
            *args.tp_axis,
            *args.cluster_axis);
    }
    for (const Tensor* meta : {&tensor_args.slot_idx, &tensor_args.kv_actual_global}) {
        TT_FATAL(meta->storage_type() == StorageType::DEVICE, "slot_idx and kv_actual_global must be on device");
        TT_FATAL(meta->dtype() == DataType::UINT32, "slot_idx and kv_actual_global must be uint32");
    }
    // One accessor serves both metadata reads.
    TT_FATAL(
        tensor_args.slot_idx.memory_config() == tensor_args.kv_actual_global.memory_config(),
        "slot_idx and kv_actual_global must share a memory config");
    if (tensor_args.output.has_value()) {
        const auto expected = Op::compute_output_specs(args, tensor_args);
        TT_FATAL(
            tensor_args.output->tensor_spec() == expected,
            "output spec {} does not match the read's spec {}",
            tensor_args.output->tensor_spec(),
            expected);
    }
}

}  // namespace

Op::program_factory_t Op::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return MeshWorkloadFactory{};
}

void Op::validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(cache.layout() == Layout::ROW_MAJOR, "read_padded_kv_cache supports ROW_MAJOR caches only");
    const auto& shape = cache.padded_shape();
    TT_FATAL(shape.rank() == 4 && shape[1] == 1, "cache must be [entries, 1, rows, width], got {}", shape);
    TT_FATAL(
        args.num_layers > 0 && shape[0] % args.num_layers == 0,
        "cache entries ({}) must be a multiple of num_layers ({})",
        shape[0],
        args.num_layers);
    TT_FATAL(
        args.num_read_layers > 0 && args.first_layer + args.num_read_layers <= args.num_layers,
        "layers [{}, {}) must be a non-empty range inside num_layers {}",
        args.first_layer,
        args.first_layer + args.num_read_layers,
        args.num_layers);
    validate_runtime(args, tensor_args);
    const uint32_t linear_factor = kv_cache_linear_factor(cache, args.cluster_axis, args.tp_axis);
    TT_FATAL(
        args.chunk_size_global % (linear_factor * TILE_HEIGHT) == 0,
        "chunk_size_global ({}) must give every one of the {} chips a whole number of 32-row tiles",
        args.chunk_size_global,
        linear_factor);
    TT_FATAL(
        shape[-2] % chunk_rows_per_chip(args, tensor_args) == 0,
        "cache rows ({}) must be a multiple of the per-chip chunk ({})",
        shape[-2],
        chunk_rows_per_chip(args, tensor_args));
}

void Op::validate_on_program_cache_hit(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime(args, tensor_args);
}

Op::spec_return_value_t Op::compute_output_specs(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    const uint32_t rows = args.num_read_layers * chunk_rows_per_chip(args, tensor_args);
    // FP8 rows come out as their bytes (UINT8, same width): FP8 has no D2D stream or host-mapper path, and
    // update_padded_kv_cache takes UINT8 rows for an FP8 cache.
    const DataType dtype = cache.dtype() == DataType::FP8_E4M3 ? DataType::UINT8 : cache.dtype();
    return TensorSpec(
        ttnn::Shape({1, 1, rows, cache.padded_shape()[-1]}),
        TensorLayout(
            dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
}

Op::topology_return_value_t Op::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Each chip's rows are its own slice of the chunk, distributed exactly as the cache's sequence dim.
    return {tensor_args.cache.tensor_topology()};
}

Op::tensor_return_value_t Op::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return *tensor_args.output;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.cache.device());
}

ttsl::hash::hash_t Op::compute_program_hash(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    return tt::tt_metal::operation::hash_operation<Op>(
        args.first_layer,
        args.num_read_layers,
        args.num_layers,
        args.chunk_size_global,
        args.cluster_axis,
        args.tp_axis,
        cache.dtype(),
        cache.memory_config(),
        cache.padded_shape(),
        tensor_args.slot_idx.memory_config(),
        chip_mapping_hash(*cache.device(), cache.device_storage().get_coords()));
}

tt::tt_metal::ProgramDescriptor Op::ProgramFactory::create_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "read_padded_kv_cache requires a mesh dispatch coordinate");
    const auto& cache = tensor_args.cache;
    auto* device = cache.device();
    const std::vector<uint32_t> chip_table = whole_mesh_chip_table(args, tensor_args).value_or(std::vector<uint32_t>{});
    const uint32_t linear_coord =
        chip_table.empty() ? kv_cache_linear_coord(cache, *mesh_dispatch_coordinate, args.cluster_axis, args.tp_axis)
                           : 0;
    const uint32_t linear_factor = kv_cache_linear_factor(cache, args.cluster_axis, args.tp_axis);
    const uint32_t chunk_local = chunk_rows_per_chip(args, tensor_args);
    const uint32_t total_rows = args.num_read_layers * chunk_local;
    const uint32_t page_bytes = cache.mesh_buffer().aligned_page_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2] =
        tt::tt_metal::split_work_to_cores(device->compute_with_storage_grid_size(), total_rows, /*row_wise=*/true);

    tt::tt_metal::ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = kReadBatchRows * page_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kRowsCbIndex,
            .data_format = datatype_to_dataformat_converter(cache.dtype()),
            .page_size = page_bytes,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = kReadMetadataBytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kReadMetaCbIndex,
            .data_format = tt::DataFormat::UInt32,
            .page_size = kReadMetadataBytes,
        }}},
    });

    KernelDescriptor kernel;
    kernel.kernel_source = kReadKernelPath;
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = all_cores;
    kernel.config = ReaderConfigDescriptor{};
    kernel.compile_time_args = {kRowsCbIndex, kReadMetaCbIndex, kReadBatchRows};
    TensorAccessorArgs(cache.buffer()).append_to(kernel.compile_time_args);
    TensorAccessorArgs(output.buffer()).append_to(kernel.compile_time_args);
    TensorAccessorArgs(tensor_args.slot_idx.buffer()).append_to(kernel.compile_time_args);
    kernel.emplace_common_runtime_args({
        linear_coord,
        linear_factor,
        chunk_local,
        args.first_layer,
        args.num_layers,
        cache.padded_shape()[-2],
        tensor_args.slot_idx.mesh_buffer().address(),          // smuggled-rta-ok: 1-element metadata tensor, read
                                                               // on device; patched on every cache hit
        tensor_args.kv_actual_global.mesh_buffer().address(),  // smuggled-rta-ok: as above
    });
    kernel.emplace_common_runtime_args({cache.buffer()});
    kernel.emplace_common_runtime_args({output.buffer()});
    kernel.named_compile_time_args.emplace_back(
        "chip_table_offset", static_cast<uint32_t>(kernel.compile_time_args.size()));
    kernel.named_compile_time_args.emplace_back("chip_table_size", static_cast<uint32_t>(chip_table.size()));
    kernel.compile_time_args.insert(kernel.compile_time_args.end(), chip_table.begin(), chip_table.end());

    const uint32_t g1_cores = core_group_1.num_cores();
    uint32_t first_row = 0;
    for (const auto& core : corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)) {
        const uint32_t rows = (first_row < g1_cores * rows_per_core_g1) ? rows_per_core_g1 : rows_per_core_g2;
        kernel.emplace_runtime_args(core, {rows, first_row});
        first_row += rows;
    }
    desc.kernels.push_back(std::move(kernel));
    return desc;
}

Op::MeshWorkloadFactory::cached_mesh_workload_t Op::MeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    if (!whole_mesh_chip_table(args, tensor_args).has_value()) {
        return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output);
    }
    // One program per range, built as the adapter builds its per-coordinate ones (same cache-hit path).
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, descriptor_adapter_t::shared_variables_t> shared_variables;
    const tt::tt_metal::WorkloadDescriptor no_workload_descriptor;
    auto& mesh_device = *ttnn::device_operation::mesh_device_operation_utils::get_mesh_device(args, tensor_args);
    for (const auto& range : tensor_coords.ranges()) {
        auto desc = ProgramFactory::create_descriptor(args, tensor_args, output, range.start_coord());
        tt::tt_metal::Program program{mesh_device, desc};
        auto collected = descriptor_adapter_t::collect_tensor_buffers(tensor_args, output, no_workload_descriptor);
        auto bindings = tt::tt_metal::resolve_bindings(program, desc, collected.buffers, collected.num_input_buffers);
        workload.add_program(range, std::move(program));
        shared_variables[range] = descriptor_adapter_t::shared_variables_t{.resolved_bindings = std::move(bindings)};
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void Op::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
    constexpr uint32_t kKernelHandle = 0;
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& common = GetCommonRuntimeArgs(program, kKernelHandle);
        TT_FATAL(kKvActualAddrArg < common.size(), "read_padded_kv_cache kernel is missing its metadata args");
        common[kSlotAddrArg] = tensor_args.slot_idx.mesh_buffer().address();
        common[kKvActualAddrArg] = tensor_args.kv_actual_global.mesh_buffer().address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor read_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& slot_idx,
    const ttnn::Tensor& kv_actual_global,
    uint32_t first_layer,
    uint32_t num_read_layers,
    uint32_t num_layers,
    uint32_t chunk_size_global,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> tp_axis,
    const std::optional<ttnn::Tensor>& output) {
    using Op =
        ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::ReadPaddedKvCacheDeviceOperation;
    auto attrs = Op::operation_attributes_t{
        .first_layer = first_layer,
        .num_read_layers = num_read_layers,
        .num_layers = num_layers,
        .chunk_size_global = chunk_size_global,
        .cluster_axis = cluster_axis,
        .tp_axis = tp_axis,
    };
    auto tensor_args = Op::tensor_args_t{
        .cache = cache,
        .slot_idx = slot_idx,
        .kv_actual_global = kv_actual_global,
        .output = output,
    };
    return ttnn::device_operation::launch<Op>(attrs, tensor_args);
}

}  // namespace ttnn::prim
