// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_padded_kv_cache_device_operation.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// Reader kernel is reused from the kv_cache fill path — purely (src_addr, num_tiles, src_start) rt-args.
// Writer is a forked variant that derives `start_id` on-device from common rt-args so that
// `batch_idx`, `kv_actual_global` and `cluster_axis` can stay out of the program hash.
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/"
    "writer_update_padded_kv_cache.cpp";

constexpr uint32_t kSrcCbIndex = 0;
constexpr uint32_t kNumInputTilesDoubleBuffered = 2;
// L1-scratch CB the writer reads the metadata page into. Payload is the
// h2d_socket_sync metadata: 4 x uint32 = [kv_actual_global, slot_idx, dst_slot, reserved].
constexpr uint32_t kMetaCbIndex = 1;
constexpr uint32_t kMetadataBytes = 16;

// Count distinct values along the cluster axis among the participating devices to determine
// sp_factor without round-tripping to the mesh view.
uint32_t sp_factor_for_tensor(const Tensor& tensor, uint32_t cluster_axis) {
    const auto device_coords = tensor.device_storage().get_coords();
    TT_FATAL(!device_coords.empty(), "device_coords is empty when computing sp_factor");
    uint32_t min_v = std::numeric_limits<uint32_t>::max();
    uint32_t max_v = 0;
    for (const auto& c : device_coords) {
        TT_FATAL(c.dims() > cluster_axis, "cluster_axis {} out of range for coord rank {}", cluster_axis, c.dims());
        const uint32_t v = c[cluster_axis];
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    return max_v - min_v + 1;
}

// Validation of the runtime-arg-dependent constraints. `slot_idx` and
// `kv_actual_global` now live in the `metadata` device tensor and are read
// on-device by the writer kernel, so their range/alignment/overflow checks
// cannot run host-side without a device read-back — that validation is the
// caller's responsibility (host-side, where the metadata payload is packed).
// Only `layer_idx` (still a host attribute) is checked here; it is invoked from
// both the cache-miss and cache-hit paths since it is not hashed.
void validate_runtime_args(
    const UpdatePaddedKvCacheDeviceOperation::operation_attributes_t& args,
    const UpdatePaddedKvCacheDeviceOperation::tensor_args_t& /*tensor_args*/) {
    TT_FATAL(
        args.layer_idx < args.num_layers,
        "layer_idx {} out of range for num_layers {}",
        args.layer_idx,
        args.num_layers);
}

}  // namespace

UpdatePaddedKvCacheDeviceOperation::program_factory_t UpdatePaddedKvCacheDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;

    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
    TT_FATAL(cache.layout() == Layout::TILE, "cache must be TILE layout");
    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(cache.dtype() == input.dtype(), "cache and input dtype must match");

    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();
    TT_FATAL(cache_shape.rank() == 4, "cache must be 4D (got rank {})", cache_shape.rank());
    TT_FATAL(input_shape.rank() == 4, "input must be 4D (got rank {})", input_shape.rank());
    TT_FATAL(cache_shape[-1] == input_shape[-1], "cache and input head dim must match");
    TT_FATAL(cache_shape[1] == input_shape[1], "cache and input num-heads dim must match");

    const uint32_t cache_seq = cache_shape[-2];
    const uint32_t input_seq = input_shape[-2];
    TT_FATAL(input_seq % TILE_HEIGHT == 0, "input seq dim ({}) must be tile-aligned", input_seq);
    TT_FATAL(cache_seq % input_seq == 0, "cache seq ({}) must be a multiple of input seq ({})", cache_seq, input_seq);

    TT_FATAL(args.num_layers > 0, "num_layers must be positive");
    TT_FATAL(
        cache_shape[0] % args.num_layers == 0,
        "cache batch dim ({}) must be a multiple of num_layers ({})",
        cache_shape[0],
        args.num_layers);

    // Runtime-arg-dependent constraints (slot_idx, layer_idx, kv_actual_global). Also re-checked
    // on the program-cache-hit path since those args are not hashed.
    validate_runtime_args(args, tensor_args);
}

void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // slot_idx, layer_idx and kv_actual_global are runtime args (not in the program hash) and can
    // differ from the call that compiled the cached program, so re-validate them every hit. The
    // structural constraints are hashed and therefore guaranteed unchanged here.
    validate_runtime_args(args, tensor_args);
}

UpdatePaddedKvCacheDeviceOperation::spec_return_value_t UpdatePaddedKvCacheDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place: output spec = cache spec.
    return tensor_args.cache.tensor_spec();
}

UpdatePaddedKvCacheDeviceOperation::tensor_return_value_t UpdatePaddedKvCacheDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place: return a handle to the cache.
    return tensor_args.cache;
}

ttsl::hash::hash_t UpdatePaddedKvCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Per-call data values (slot_idx, layer_idx, kv_actual_global) are runtime args read by
    // the writer kernel and intentionally NOT in the hash, so successive chunks reuse the
    // cached program; rt-args refresh on cache hits via apply_descriptor_runtime_args.
    // num_layers and cluster_axis stay IN: both are structural — they govern the cache slot
    // linearization (num_layers) and which mesh dim is sp (cluster_axis) — not per-call data.
    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;
    return tt::tt_metal::operation::hash_operation<UpdatePaddedKvCacheDeviceOperation>(
        args.num_layers,
        args.cluster_axis,
        input.dtype(),
        input.memory_config(),
        input.padded_shape().volume(),
        cache.memory_config(),
        cache.padded_shape().volume());
}

tt::tt_metal::ProgramDescriptor UpdatePaddedKvCacheDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "UpdatePaddedKvCache::create_descriptor requires a mesh dispatch coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();

    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;
    auto* device = input.device();

    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(data_format);

    const uint32_t Wt = cache_shape[-1] / TILE_WIDTH;
    const uint32_t input_Ht = input_shape[-2] / TILE_HEIGHT;
    const uint32_t cache_HtWt = cache_shape[-2] * Wt / TILE_HEIGHT;
    const uint32_t cache_CHtWt = cache_shape[1] * cache_HtWt;

    // Per-chip kernel inputs: kernel does the update_idxt + start_id math itself from these.
    const uint32_t sp_factor = sp_factor_for_tensor(cache, args.cluster_axis);
    const uint32_t my_sp_coord = ::ttnn::ccl::get_linearized_index_from_physical_coord(cache, coord, args.cluster_axis);
    // kv_actual_global is no longer a host value — the writer kernel reads it (and
    // slot_idx) from the metadata DRAM tensor and divides by TILE_HEIGHT on-device.

    // Work split: one tile per "block". num_blocks_of_work = input_C * input_Ht (= num_heads * seq_tiles).
    const uint32_t num_blocks_of_work = input_shape[1] * input_Ht;

    const auto compute_grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_g1, num_blocks_per_core_g2] =
        tt::tt_metal::split_work_to_cores(compute_grid, num_blocks_of_work, /*row_major=*/true);

    tt::tt_metal::ProgramDescriptor desc;

    // CB for the input tiles.
    desc.cbs.push_back(CBDescriptor{
        .total_size = kNumInputTilesDoubleBuffered * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kSrcCbIndex,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    // L1-scratch CB the writer reads the metadata page into (one 16B page).
    desc.cbs.push_back(CBDescriptor{
        .total_size = kMetadataBytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kMetaCbIndex,
            .data_format = tt::DataFormat::UInt32,
            .page_size = kMetadataBytes,
        }}},
    });

    // Reader kernel descriptor.
    KernelDescriptor::CompileTimeArgs reader_compile_args;
    TensorAccessorArgs(input.buffer()).append_to(reader_compile_args);

    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source = kReaderKernelPath;
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = all_cores;
    reader_kernel.compile_time_args = std::move(reader_compile_args);
    reader_kernel.config = ReaderConfigDescriptor{};

    // Writer kernel descriptor. Compile args: [kSrcCbIndex, kMetaCbIndex, <cache accessor...>].
    KernelDescriptor::CompileTimeArgs writer_compile_args = {kSrcCbIndex, kMetaCbIndex};
    TensorAccessorArgs(cache.buffer()).append_to(writer_compile_args);

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = kWriterKernelPath;
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = all_cores;
    writer_kernel.compile_time_args = std::move(writer_compile_args);
    writer_kernel.config = WriterConfigDescriptor{};

    // Common rt-args: per-chip kernel inputs for on-device update_idxt + start_id derivation.
    // `kv_actual_global` and `slot_idx` are NOT here — the kernel reads them from the
    // metadata tensor at `metadata_addr` (last arg). layer kept separate (kernel composes
    // batch_idx = slot_idx * num_layers + layer_idx with the slot_idx it read).
    const uint32_t metadata_addr = tensor_args.metadata.buffer()->address();
    writer_kernel.emplace_common_runtime_args({
        my_sp_coord,
        sp_factor,
        input_Ht,
        args.layer_idx,
        args.num_layers,
        Wt,
        cache_HtWt,
        cache_CHtWt,
        metadata_addr,
    });

    // Per-core runtime args.
    auto* src_buffer = input.buffer();
    auto* dst_buffer = cache.buffer();
    const uint32_t g1_numcores = core_group_1.num_cores();

    const auto cores = corerange_to_cores(all_cores, num_cores, /*row_major=*/true);
    reader_kernel.runtime_args.reserve(num_cores);
    writer_kernel.runtime_args.reserve(num_cores);

    uint32_t num_blocks_written = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t num_blocks_per_core = (i < g1_numcores) ? num_blocks_per_core_g1 : num_blocks_per_core_g2;

        // Reader: (src_addr, num_tiles, src_start_tile_id)
        reader_kernel.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                src_buffer->address(),
                num_blocks_per_core * Wt,
                num_blocks_written * Wt,
            });

        // Writer: (dst_addr, num_pages, core_blocks_written) — kernel adds update_idxt+head offset itself.
        writer_kernel.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                dst_buffer->address(),
                num_blocks_per_core * Wt,
                num_blocks_written,
            });

        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const ttnn::Tensor& metadata,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t cluster_axis) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::UpdatePaddedKvCacheDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .layer_idx = layer_idx,
        .num_layers = num_layers,
        .cluster_axis = cluster_axis,
    };
    auto tensor_args = OperationType::tensor_args_t{.cache = cache, .input = input, .metadata = metadata};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
