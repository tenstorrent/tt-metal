// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "toy_scaled_add_device_operation.hpp"

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

#include "toy_scaled_add_common.hpp"

namespace ttnn::operations::toy_scaled_add {

using namespace tt::tt_metal;

namespace {

bool is_height_sharded(const MemoryConfig& memory_config) {
    return memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
}

bool is_supported_dtype(DataType dtype) { return dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32; }

void refuse_unless(bool supported, std::string_view axis, const auto& value, std::string_view allowed) {
    if (!supported) {
        throw UnsupportedAxisValue(fmt::format("toy_scaled_add: {}={} not in SUPPORTED [{}]", axis, value, allowed));
    }
}

// The axes one tensor operand contributes. The kernels walk full 32 x 32 tiles; the factories size every
// page from the tensor's own tile.
void check_operand(const Tensor& t, std::string_view dtype_axis) {
    refuse_unless(is_supported_dtype(t.dtype()), dtype_axis, t.dtype(), "BFLOAT16, FLOAT32");
    refuse_unless(t.layout() == Layout::TILE, "layout", t.layout(), "TILE");
    const auto tile = t.tensor_spec().tile().get_tile_shape();
    refuse_unless(tile == Tile().get_tile_shape(), "tile", fmt::format("{}x{}", tile[0], tile[1]), "32x32");
}

// The program is launched on a's device with the other tensors' addresses in it.
void validate_devices(const ToyScaledAddInputs& t) {
    TT_FATAL(t.b.device() == t.a.device(), "toy_scaled_add: b must be on a's device");
    TT_FATAL(!t.gamma.has_value() || t.gamma->device() == t.a.device(), "toy_scaled_add: gamma must be on a's device");
    TT_FATAL(
        !t.output.has_value() || t.output->device() == t.a.device(),
        "toy_scaled_add: output_tensor must be on a's device");
}

void validate_output(const ToyScaledAddParams& attrs, const ToyScaledAddInputs& t) {
    if (!t.output.has_value()) {
        return;
    }
    const Tensor& output = *t.output;
    TT_FATAL(
        output.tensor_spec() == ToyScaledAddDeviceOperation::compute_output_specs(attrs, t),
        "toy_scaled_add: output_tensor spec {} does not match {}",
        output.tensor_spec(),
        ToyScaledAddDeviceOperation::compute_output_specs(attrs, t));
}

}  // namespace

void check_support(const ToyScaledAddParams& attrs, const ToyScaledAddInputs& t) {
    check_operand(t.a, "dtype");
    check_operand(t.b, "b_dtype");
    if (t.gamma.has_value()) {
        check_operand(*t.gamma, "gamma_dtype");
    }
    refuse_unless(is_supported_dtype(attrs.output_dtype), "output_dtype", attrs.output_dtype, "BFLOAT16, FLOAT32");
    const uint32_t rank = t.a.logical_shape().rank();
    refuse_unless(rank >= 2 && rank <= 4, "rank", rank, "2, 3, 4");
    const MemoryConfig& out_mc = attrs.output_memory_config;
    refuse_unless(
        out_mc.memory_layout() == TensorMemoryLayout::INTERLEAVED || is_height_sharded(out_mc),
        "memory_layout",
        out_mc.memory_layout(),
        "INTERLEAVED, HEIGHT_SHARDED");
    // The sharded program backs its circular buffers with the shards, and circular buffers live in L1.
    if (is_height_sharded(out_mc) && out_mc.buffer_type() == BufferType::DRAM) {
        throw ExcludedCell("toy_scaled_add: unsupported combination: memory_layout=HEIGHT_SHARDED, buffer_type=DRAM");
    }
}

ToyScaledAddDeviceOperation::program_factory_t ToyScaledAddDeviceOperation::select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/) {
    if (is_height_sharded(attrs.output_memory_config)) {
        return HeightShardedProgramFactory{};
    }
    return InterleavedProgramFactory{};
}

void ToyScaledAddDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // The public entry already checked the support contract before launching; a direct prim caller has not.
    check_support(attrs, t);
    validate_devices(t);
    TT_FATAL(
        t.a.padded_shape() == t.b.padded_shape(),
        "toy_scaled_add: a {} and b {} must have the same padded shape",
        t.a.padded_shape(),
        t.b.padded_shape());

    const uint32_t width = t.a.padded_shape()[-1];
    if (t.gamma.has_value()) {
        const Tensor& gamma = *t.gamma;
        TT_FATAL(
            gamma.padded_shape()[-1] == width && gamma.padded_shape().volume() == tt::constants::TILE_HEIGHT * width,
            "toy_scaled_add: gamma {} must be one tile-row as wide as a {}",
            gamma.padded_shape(),
            t.a.padded_shape());
        TT_FATAL(!gamma.memory_config().is_sharded(), "toy_scaled_add: gamma must be interleaved");
    }

    const MemoryConfig& out_mc = attrs.output_memory_config;
    const bool sharded = is_height_sharded(out_mc);
    if (sharded) {
        TT_FATAL(
            is_height_sharded(t.a.memory_config()) && is_height_sharded(t.b.memory_config()),
            "toy_scaled_add: a, b and the output must be all interleaved or all height-sharded");
        const auto& shard_spec = t.a.memory_config().shard_spec();
        TT_FATAL(
            t.b.memory_config().shard_spec() == shard_spec && out_mc.shard_spec() == shard_spec,
            "toy_scaled_add: a, b and the output must share one shard spec");
        TT_FATAL(
            shard_spec->shape[1] == width && shard_spec->shape[0] % tt::constants::TILE_HEIGHT == 0,
            "toy_scaled_add: shard shape {} must span the full row and a whole number of tiles",
            shard_spec->shape);
    } else {
        TT_FATAL(
            !t.a.memory_config().is_sharded() && !t.b.memory_config().is_sharded() && !out_mc.is_sharded(),
            "toy_scaled_add: a, b and the output must be all interleaved or all height-sharded");
    }

    // The gamma row and the streaming buffers grow with the row width; reject what cannot fit rather
    // than fail while allocating the program.
    auto* device = t.a.device();
    const uint64_t available_l1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t out_tile_bytes =
        datum_size(datatype_to_dataformat_converter(attrs.output_dtype)) * tt::constants::TILE_HW;
    const uint32_t cb_bytes = detail::cb_bytes_per_core(t.a, t.b, t.gamma, out_tile_bytes, sharded);
    TT_FATAL(
        cb_bytes <= available_l1,
        "toy_scaled_add: rows of width {} need {} B of circular buffers per core; {} B of L1 are available",
        width,
        cb_bytes,
        available_l1);

    validate_output(attrs, t);
}

void ToyScaledAddDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    validate_devices(t);
    validate_output(attrs, t);
}

ttsl::hash::hash_t ToyScaledAddDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // alpha is why this op writes its own key and its own cache-hit hook. The default key hashes every
    // attribute, alpha included, and would compile a program per alpha; the default hit path patches
    // only buffer addresses. alpha is a runtime arg instead: it stays out of this key, and the
    // factories' override_runtime_arguments writes it, with the buffer addresses, on every hit. An op
    // whose calls differ only in their buffers keeps the default key and lets the framework patch the
    // Buffer* slots its descriptor declares.
    //
    // The key is everything that shapes the compiled program: the output's dtype and placement, the
    // compute kernel config, and the specs of a, b and gamma (shape, dtype, tile, layout, placement).
    // The shapes fix the work split, so a hit keeps every per-core argument as the miss wrote it. A
    // custom key gives up the framework's collision fallback, so it leaves nothing structural out.
    auto hash = tt::tt_metal::operation::hash_operation<ToyScaledAddDeviceOperation>(
        attrs.output_dtype,
        attrs.output_memory_config,
        attrs.compute_kernel_config,
        t.a.tensor_spec(),
        t.b.tensor_spec(),
        t.gamma.has_value());
    if (t.gamma.has_value()) {
        hash = ttsl::hash::hash_objects(hash, t.gamma->tensor_spec());
    }
    return hash;
}

ToyScaledAddDeviceOperation::spec_return_value_t ToyScaledAddDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    return TensorSpec(
        t.a.logical_shape(), TensorLayout(attrs.output_dtype, PageConfig(Layout::TILE), attrs.output_memory_config));
}

ToyScaledAddDeviceOperation::tensor_return_value_t ToyScaledAddDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output.has_value()) {
        return *t.output;
    }
    return create_device_tensor(compute_output_specs(attrs, t), t.a.device());
}

}  // namespace ttnn::operations::toy_scaled_add

namespace ttnn::prim {

Tensor toy_scaled_add(
    const Tensor& a,
    const Tensor& b,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& output,
    const ttnn::operations::toy_scaled_add::ToyScaledAddParams& params) {
    using OperationType = ttnn::operations::toy_scaled_add::ToyScaledAddDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.a = a, .b = b, .gamma = gamma, .output = output});
}

}  // namespace ttnn::prim
