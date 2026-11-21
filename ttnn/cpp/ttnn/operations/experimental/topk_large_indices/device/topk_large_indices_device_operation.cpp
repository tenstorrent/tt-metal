// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_device_operation.hpp"

#include <tt-metalium/math.hpp>

#include <algorithm>
#include <limits>

namespace ttnn::operations::experimental::topk_large_indices {

namespace {

constexpr uint32_t max_supported_k = 2048;
constexpr uint32_t max_row_elements = 1u << 30;

void validate_static_args(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    // These fields are part of the program hash or process/device setup. They are validated on cache
    // miss before compiling the program and do not need to be rechecked on every cache hit.
    TT_FATAL(
        attrs.k > 0 && attrs.k <= max_supported_k && attrs.k % 16 == 0,
        "topk_large_indices supports k in [16, {}] in multiples of 16, got {}",
        max_supported_k,
        attrs.k);
    const tt::ARCH arch = input.device()->arch();
    TT_FATAL(
        arch == tt::ARCH::BLACKHOLE, "topk_large_indices is only supported on Blackhole architecture, got {}", arch);
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "topk_large_indices input must be ROW_MAJOR");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "topk_large_indices input must be BFLOAT16");
    TT_FATAL(!input.is_sharded(), "topk_large_indices input must use interleaved memory");
}

void validate_two_level_tensor(
    const Tensor& input, const std::optional<Tensor>& tensor, DataType dtype, uint32_t width, const char* name) {
    if (!tensor.has_value()) {
        return;
    }
    const auto& t = *tensor;
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.is_allocated() && t.device() == input.device(),
        "topk_large_indices {} must be allocated on the input's device",
        name);
    TT_FATAL(
        t.dtype() == dtype && t.layout() == Layout::ROW_MAJOR,
        "topk_large_indices {} has the wrong dtype or layout",
        name);
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            t.memory_config().buffer_type() == BufferType::DRAM,
        "topk_large_indices {} must be DRAM interleaved",
        name);
    const auto& shape = t.logical_shape();
    const uint32_t rows = width == 1 ? 1u : flattened_rows_excluding_last_dim(input.logical_shape());
    TT_FATAL(
        shape[-1] == width && flattened_rows_excluding_last_dim(shape) == rows,
        "topk_large_indices {} has shape {}, expected {} rows of last dimension {}",
        name,
        shape,
        rows,
        width);
    // The kernels address rows by a packed pitch of `width` elements.
    TT_FATAL(
        t.buffer()->page_size() == width * t.element_size(),
        "topk_large_indices {} must have one packed row per page ({} B), got pages of {} B",
        name,
        width * t.element_size(),
        t.buffer()->page_size());
}

void validate_runtime_args(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    // Shape is intentionally omitted from the program hash and patched through runtime args, so keep
    // these checks on both cache miss and cache hit.
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "topk_large_indices input must be on device");
    TT_FATAL(input.is_allocated(), "topk_large_indices input must have an allocated buffer");

    auto* device = input.device();
    const auto selected_subdevice_id = attrs.subdevice_id.value_or(device->get_sub_device_ids().at(0));
    const auto& active_subdevice_ids = device->get_sub_device_ids();
    TT_FATAL(
        std::find(active_subdevice_ids.begin(), active_subdevice_ids.end(), selected_subdevice_id) !=
            active_subdevice_ids.end(),
        "topk_large_indices subdevice_id {} is not part of active subdevice manager {}",
        selected_subdevice_id,
        attrs.subdevice_manager_id);
    const auto selected_subdevice_cores =
        device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, selected_subdevice_id);
    TT_FATAL(
        selected_subdevice_cores.contains(attrs.resolved_worker_core_grid),
        "topk_large_indices resolved worker grid {} must be fully contained in TENSIX subdevice {} cores {}",
        attrs.resolved_worker_core_grid,
        selected_subdevice_id,
        selected_subdevice_cores);
    TT_FATAL(
        attrs.resolved_worker_core_grid.num_cores() > 0, "topk_large_indices requires at least one TENSIX worker core");

    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() >= 1, "topk_large_indices input must have rank >= 1");
    const uint64_t num_rows = flattened_rows_excluding_last_dim(shape);
    TT_FATAL(num_rows > 0, "topk_large_indices input must have at least one row");
    const uint32_t n = shape[shape.rank() - 1];
    TT_FATAL(n >= attrs.k, "topk_large_indices input last dimension {} must be >= k {}", n, attrs.k);
    TT_FATAL(
        n <= max_row_elements,
        "topk_large_indices initial implementation supports at most {} elements in the last dimension; got {}",
        max_row_elements,
        n);
    const uint64_t input_row_bytes = static_cast<uint64_t>(n) * input.element_size();
    TT_FATAL(
        input_row_bytes <= std::numeric_limits<uint32_t>::max(),
        "topk_large_indices input row size must fit in uint32_t bytes; got {} bytes",
        input_row_bytes);

    // Optional bounded search width: top-k scans only the first valid_length columns of each row (the rest
    // of the physically-wider row is ignored, not read).  A short valid prefix is legal: inactive lanes
    // are already represented as -inf by the TopK XL tail path and are materialized as 0xFFFFFFFF indices.
    // This keeps a fixed worst-case k/output shape while a serving cache grows through its early prefixes.
    if (tensor_args.has_valid_length_metadata()) {
        // Mutually exclusive, rather than silently preferring one: the on-device value would win and the
        // caller's scalar would be quietly ignored.
        TT_FATAL(
            !attrs.valid_length.has_value(),
            "topk_large_indices: valid_length and valid_length_tensor are mutually exclusive (the tensor form "
            "is read on-device; a scalar supplied alongside it would be ignored)");
        const auto& m = *tensor_args.valid_length_tensor;
        TT_FATAL(
            m.storage_type() == StorageType::DEVICE && m.is_allocated(),
            "topk_large_indices valid_length_tensor must be allocated on device");
        TT_FATAL(
            m.device() == tensor_args.input_tensor.device(),
            "topk_large_indices valid_length_tensor must be on the same device as the input");
        TT_FATAL(m.dtype() == DataType::UINT32, "topk_large_indices valid_length_tensor must be UINT32");
        TT_FATAL(m.layout() == Layout::ROW_MAJOR, "topk_large_indices valid_length_tensor must be ROW_MAJOR");
        TT_FATAL(
            m.logical_volume() == 1,
            "topk_large_indices valid_length_tensor must hold exactly 1 element (got {})",
            m.logical_volume());
        TT_FATAL(
            m.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "topk_large_indices valid_length_tensor must be interleaved (the kernel reads page 0 at a single "
            "fixed address)");
        TT_FATAL(
            m.memory_config().buffer_type() == BufferType::DRAM,
            "topk_large_indices valid_length_tensor must be in DRAM");
        // The VALUE lives on device, so the <= n bound cannot be checked here; the offset alone can.
        TT_FATAL(
            attrs.valid_length_offset <= n,
            "topk_large_indices valid_length_offset {} must be <= the input last dimension {}",
            attrs.valid_length_offset,
            n);
    } else {
        TT_FATAL(
            attrs.valid_length_offset == 0, "topk_large_indices: valid_length_offset requires valid_length_tensor");
    }
    if (tensor_args.has_valid_end_metadata()) {
        TT_FATAL(
            tensor_args.has_valid_length_metadata(),
            "topk_large_indices: valid_end_tensor requires valid_length_tensor -- it only CAPS the bound the "
            "valid-length derivation produces");
        const auto& m = *tensor_args.valid_end_tensor;
        TT_FATAL(
            m.storage_type() == StorageType::DEVICE && m.is_allocated(),
            "topk_large_indices valid_end_tensor must be allocated on device");
        TT_FATAL(
            m.device() == tensor_args.input_tensor.device(),
            "topk_large_indices valid_end_tensor must be on the same device as the input");
        TT_FATAL(m.dtype() == DataType::UINT32, "topk_large_indices valid_end_tensor must be UINT32");
        TT_FATAL(m.layout() == Layout::ROW_MAJOR, "topk_large_indices valid_end_tensor must be ROW_MAJOR");
        TT_FATAL(
            m.logical_volume() == 1,
            "topk_large_indices valid_end_tensor must hold exactly 1 element (got {})",
            m.logical_volume());
        TT_FATAL(
            m.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "topk_large_indices valid_end_tensor must be interleaved (the kernel reads page 0 at a single "
            "fixed address)");
        TT_FATAL(
            m.memory_config().buffer_type() == BufferType::DRAM, "topk_large_indices valid_end_tensor must be in DRAM");
    }
    // block_select is not in the hash of the program it falls back to, so it is checked on every call.
    if (attrs.block_select.has_value()) {
        TT_FATAL(
            *attrs.block_select == program::two_level_block_keys,
            "topk_large_indices block_select supports {} only, got {}",
            program::two_level_block_keys,
            *attrs.block_select);
    }
    // Width-dependent, so checked on every call: a block_select program runs two-level only for these widths and
    // runs the single-level program otherwise.
    const bool two_level = program::two_level_enabled(attrs, n);
    if (!two_level) {
        TT_FATAL(
            !tensor_args.block_max_tensor.has_value() && !tensor_args.block_max_output.has_value() &&
                !tensor_args.mode_output.has_value() && !attrs.two_level_cutover.has_value(),
            "topk_large_indices block_max_tensor, block_max_output, mode_output and two_level_cutover need a "
            "two-level program: block_select=32, k = {}, and a last dimension in [{}, {}] that is a multiple of {} "
            "(got k = {}, last dimension {})",
            max_supported_k,
            program::two_level_min_width,
            program::two_level_max_width,
            program::two_level_block_keys,
            attrs.k,
            n);
    } else {
        TT_FATAL(input.buffer()->is_dram(), "topk_large_indices two-level selection requires the input in DRAM");
        // Phase 1 and the stage-3 gather address a row's keys as offsets into its page.
        TT_FATAL(
            input.buffer()->page_size() == input_row_bytes,
            "topk_large_indices two-level selection needs one packed input row per page ({} B), got pages of {} B",
            input_row_bytes,
            input.buffer()->page_size());
        // Readiness is counted only for rows whose maxima phase 1 computes.
        const uint32_t num_cores = attrs.resolved_worker_core_grid.num_cores();
        TT_FATAL(
            tensor_args.block_max_tensor.has_value() ||
                num_rows <= uint64_t{program::two_level_max_rows_per_core} * num_cores,
            "topk_large_indices two-level selection gives each core at most {} rows: {} rows over {} cores",
            program::two_level_max_rows_per_core,
            num_rows,
            num_cores);
        TT_FATAL(
            !(tensor_args.block_max_tensor.has_value() && tensor_args.block_max_output.has_value()),
            "topk_large_indices block_max_output receives computed maxima; with block_max_tensor none are computed");
        // One page per row: the reader and writer address these by row.
        const uint32_t blocks = tt::div_up(n, program::two_level_block_keys);
        validate_two_level_tensor(input, tensor_args.block_max_tensor, DataType::BFLOAT16, blocks, "block_max_tensor");
        validate_two_level_tensor(input, tensor_args.block_max_output, DataType::BFLOAT16, blocks, "block_max_output");
        validate_two_level_tensor(input, tensor_args.mode_output, DataType::UINT32, 1, "mode_output");
        // The mode is written while other cores still read their inputs, so it must not share a buffer with them.
        if (tensor_args.mode_output.has_value()) {
            const auto mode_address = tensor_args.mode_output->buffer()->address();
            for (const auto* other :
                 {&tensor_args.valid_length_tensor, &tensor_args.valid_end_tensor, &tensor_args.block_max_tensor}) {
                TT_FATAL(
                    !other->has_value() || (*other)->buffer()->address() != mode_address,
                    "topk_large_indices mode_output must not share a buffer with an input");
            }
        }
    }
    if (attrs.valid_length.has_value()) {
        const uint32_t valid_length = attrs.valid_length.value();
        TT_FATAL(valid_length > 0, "topk_large_indices valid_length must be > 0");
        TT_FATAL(
            valid_length <= n,
            "topk_large_indices valid_length {} must be <= the input last dimension {}",
            valid_length,
            n);
    }
}

}  // namespace

void TopkLargeIndicesDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_runtime_args(attrs, tensor_args);
}

void TopkLargeIndicesDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_static_args(attrs, tensor_args);
    validate_runtime_args(attrs, tensor_args);
}

ttsl::hash::hash_t TopkLargeIndicesDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const uint32_t num_rows = flattened_rows_excluding_last_dim(input.logical_shape());
    const uint32_t input_last_dim = input.logical_shape()[-1];
    const uint32_t num_cores = attrs.resolved_worker_core_grid.num_cores();
    const auto base = tt::tt_metal::operation::hash_operation<TopkLargeIndicesDeviceOperation>(
        // Metadata presence changes the reader/compute binaries (extra CBs + accessor args + the on-device
        // derivation) and the offset is baked in as a compile arg, so both must be hashed. The per-chunk
        // VALUE never enters the key -- that is what lets one captured program serve every chunk.
        tensor_args.has_valid_length_metadata(),
        // Presence only, and still required even though the reader no longer carries a valid_end flag: the
        // accessor compile args are built from this tensor's buffer (the input is the placeholder when
        // absent), so presence still changes the binary. The VALUE stays dynamic -- and so now does the
        // bound's on/off, read from the common arg -- so one captured program serves partial and full chunks.
        tensor_args.has_valid_end_metadata(),
        attrs.valid_length_offset,
        attrs.k,
        attrs.resolved_worker_core_grid,
        // Picks the reader's and writer's NoCs, so it changes the kernel binaries.
        static_cast<uint32_t>(attrs.dram_read_noc),
        input.dtype(),
        input.layout(),
        input.memory_config().memory_layout(),
        input.memory_config().buffer_type(),
        static_cast<uint32_t>(program::compute_body_mode(attrs.k, input_last_dim)),
        // The row count and width enter the key only through these classes of program.
        program::multi_chunk_rows(attrs.k, input_last_dim),
        program::split_whole_rows(attrs.k, num_rows, input_last_dim, num_cores),
        program::row_split_enabled(attrs.k, num_rows, input_last_dim, num_cores),
        program::long_rows(attrs.k, input_last_dim));
    if (!program::two_level_enabled(attrs, input_last_dim)) {
        return base;
    }
    // Two-level programs add their own kernels' structure; single-level keys take only the base.
    return tt::tt_metal::operation::hash_operation<TopkLargeIndicesDeviceOperation>(
        base,
        *attrs.block_select,
        // The block flags and the maxima row pitch are sized from the width, the program's own maxima tensor from the
        // width and the row count.
        input_last_dim,
        num_rows,
        tensor_args.block_max_tensor.has_value(),
        tensor_args.block_max_output.has_value(),
        tensor_args.mode_output.has_value());
}

spec_return_value_t TopkLargeIndicesDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    std::vector<uint32_t> output_shape_vec;
    output_shape_vec.reserve(input_shape.rank());
    for (uint32_t i = 0; i < input_shape.rank(); ++i) {
        output_shape_vec.push_back(input_shape[i]);
    }
    output_shape_vec.back() = attrs.k;

    const auto memory_config = tensor_args.input_tensor.memory_config();
    return tt::tt_metal::TensorSpec(
        ttnn::Shape(std::move(output_shape_vec)),
        tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config));
}

tensor_return_value_t TopkLargeIndicesDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<TopkLargeIndicesDeviceOperation::operation_attributes_t, TopkLargeIndicesDeviceOperation::tensor_args_t>
TopkLargeIndicesDeviceOperation::invoke(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    const std::optional<Tensor>& valid_length_tensor,
    uint32_t valid_length_offset,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const std::optional<Tensor>& valid_end_tensor,
    std::optional<tt::tt_metal::NOC> dram_read_noc,
    std::optional<uint32_t> block_select,
    std::optional<uint32_t> two_level_cutover,
    const std::optional<Tensor>& block_max_tensor,
    const std::optional<Tensor>& block_max_output,
    const std::optional<Tensor>& mode_output) {
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "topk_large_indices input must be on device");
    auto* device = input_tensor.device();
    TT_FATAL(device != nullptr, "topk_large_indices input must have a device");

    const auto subdevice_manager_id = device->get_active_sub_device_manager_id();
    const auto selected_subdevice_id = subdevice_id.value_or(device->get_sub_device_ids().at(0));
    const auto& active_subdevice_ids = device->get_sub_device_ids();
    TT_FATAL(
        std::find(active_subdevice_ids.begin(), active_subdevice_ids.end(), selected_subdevice_id) !=
            active_subdevice_ids.end(),
        "topk_large_indices subdevice_id {} is not part of active subdevice manager {}",
        selected_subdevice_id,
        subdevice_manager_id);
    const auto selected_subdevice_cores =
        device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, selected_subdevice_id);
    if (sub_core_grid.has_value()) {
        TT_FATAL(
            selected_subdevice_cores.contains(*sub_core_grid),
            "topk_large_indices sub_core_grid {} must be fully contained in TENSIX subdevice {} cores {}",
            *sub_core_grid,
            selected_subdevice_id,
            selected_subdevice_cores);
    }
    const auto resolved_worker_core_grid = sub_core_grid.value_or(selected_subdevice_cores);
    TT_FATAL(resolved_worker_core_grid.num_cores() > 0, "topk_large_indices requires at least one TENSIX worker core");
    const auto resolved_dram_read_noc =
        dram_read_noc.value_or(tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch()));

    return {
        operation_attributes_t{
            .k = k,
            .subdevice_id = subdevice_id,
            .sub_core_grid = sub_core_grid,
            .subdevice_manager_id = subdevice_manager_id,
            .resolved_worker_core_grid = resolved_worker_core_grid,
            .valid_length = valid_length,
            .valid_length_offset = valid_length_offset,
            .dram_read_noc = resolved_dram_read_noc,
            .block_select = block_select,
            .two_level_cutover = two_level_cutover},
        tensor_args_t{
            .input_tensor = input_tensor,
            .valid_length_tensor = valid_length_tensor,
            .valid_end_tensor = valid_end_tensor,
            .block_max_tensor = block_max_tensor,
            .block_max_output = block_max_output,
            .mode_output = mode_output}};
}

}  // namespace ttnn::operations::experimental::topk_large_indices

namespace ttnn::experimental {

Tensor topk_large_indices(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    const std::optional<Tensor>& valid_length_tensor,
    uint32_t valid_length_offset,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const std::optional<Tensor>& valid_end_tensor,
    std::optional<tt::tt_metal::NOC> dram_read_noc,
    std::optional<uint32_t> block_select,
    std::optional<uint32_t> two_level_cutover,
    const std::optional<Tensor>& block_max_tensor) {
    return topk_large_indices_with_test_outputs(
        input_tensor,
        k,
        valid_length,
        valid_length_tensor,
        valid_length_offset,
        subdevice_id,
        sub_core_grid,
        valid_end_tensor,
        dram_read_noc,
        block_select,
        two_level_cutover,
        block_max_tensor,
        std::nullopt,
        std::nullopt);
}

Tensor topk_large_indices_with_test_outputs(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    const std::optional<Tensor>& valid_length_tensor,
    uint32_t valid_length_offset,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const std::optional<Tensor>& valid_end_tensor,
    std::optional<tt::tt_metal::NOC> dram_read_noc,
    std::optional<uint32_t> block_select,
    std::optional<uint32_t> two_level_cutover,
    const std::optional<Tensor>& block_max_tensor,
    const std::optional<Tensor>& block_max_output,
    const std::optional<Tensor>& mode_output) {
    auto [operation_attributes, tensor_args] =
        operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation::invoke(
            input_tensor,
            k,
            valid_length,
            valid_length_tensor,
            valid_length_offset,
            subdevice_id,
            sub_core_grid,
            valid_end_tensor,
            dram_read_noc,
            block_select,
            two_level_cutover,
            block_max_tensor,
            block_max_output,
            mode_output);
    return ttnn::device_operation::launch<
        operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation>(
        operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
