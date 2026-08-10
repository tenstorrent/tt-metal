// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::pad_codegen {

namespace {

// RM: pad_rm_interleaved's cb_out page grows with W_out and is depth-batched
// (up to kPadReadBatch/kPadWriteBatch pages each way); ops/pad/spec.py's
// _rm_pad_batches_for_l1 self-adapts read_batch/write_batch down to fit, but
// if even the fully-clamped (1, 1) batch still overflows the usable L1
// budget there is nothing further to shrink -- the Python reference never
// reaches this because its sweep only runs shapes that already fit, but a
// port that skips this check would let an oversized W_out reach codegen and
// throw out of circular-buffer allocation at program-compile time instead of
// falling back to native. This mirrors PadCodegenProgramFactory's own
// batch-selection exactly (same helper), so the routing gate and the factory
// can never disagree on whether a shape fits.
bool rm_fits_in_l1(const Tensor& input, uint32_t W, uint32_t W_out, uint32_t front_w) {
    if (input.storage_type() != ttnn::StorageType::DEVICE) {
        // Not yet on device (e.g. host-side probing); nothing to bound against.
        return true;
    }
    const uint32_t elem_size = input.element_size();
    const uint32_t stick_size = W * elem_size;
    const uint32_t stick_size_out = W_out * elem_size;
    (void)front_w;
    const auto& allocator = input.device()->allocator();
    const uint32_t dram_alignment = allocator->get_alignment(tt::tt_metal::BufferType::DRAM);
    const uint32_t stick_size_out_aligned =
        tt::round_up(stick_size_out, std::max<uint32_t>(16, dram_alignment));
    const uint32_t stage_buf_size = tt::round_up(stick_size, dram_alignment);

    auto [read_batch, write_batch] =
        ttnn::prim::pad_rm_batches_for_l1(stage_buf_size, stick_size_out_aligned, operations::data_movement::get_max_l1_space(input));
    const uint64_t depth = static_cast<uint64_t>(std::max(read_batch, write_batch)) * 2;
    const uint64_t projected_cb_bytes =
        depth * stick_size_out_aligned + stick_size_out_aligned + stage_buf_size;
    const uint64_t budget = operations::data_movement::get_max_l1_space(input) > ttnn::prim::kPadL1SafetyMargin
        ? static_cast<uint64_t>(operations::data_movement::get_max_l1_space(input)) - ttnn::prim::kPadL1SafetyMargin
        : 0;
    return projected_cb_bytes <= budget;
}

}  // namespace

ImplementationSelector parse_implementation(const std::string& implementation) {
    if (implementation == "auto") {
        return ImplementationSelector::Auto;
    }
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_THROW("Unknown pad implementation '{}': expected 'auto', 'native', or 'codegen'", implementation);
}

bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& output_padded_shape,
    const std::array<uint32_t, 4>& front,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    // Sharded input/output: the manifest's hand-authored sharded case is
    // left out-of-scope for this port; neither kernel copied here reads a
    // ShardSpec.
    if (input.memory_config().is_sharded() || output_mem_config.is_sharded()) {
        return false;
    }
    const auto& in_shape = input.logical_shape();
    if (in_shape.rank() != 4 || output_padded_shape.rank() != 4) {
        return false;
    }
    const auto dtype = input.dtype();
    const auto layout = input.layout();

    const uint32_t N = in_shape[0];
    const uint32_t C = in_shape[1];
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];
    const uint32_t front_n = front[0];
    const uint32_t front_c = front[1];
    const uint32_t front_h = front[2];
    const uint32_t front_w = front[3];

    if (layout == ttnn::ROW_MAJOR_LAYOUT) {
        // codegen_pad.py::invalidate_vector: RM rejects only bfloat8_b (RM
        // storage is never tilized so bf8_b, a block-tiled format, has no
        // valid row-major representation to begin with).
        if (dtype == tt::tt_metal::DataType::BFLOAT8_B) {
            return false;
        }
        const uint32_t W_out = output_padded_shape[3];
        return rm_fits_in_l1(input, W, W_out, front_w);
    }
    if (layout == ttnn::TILE_LAYOUT) {
        // codegen_pad.py::invalidate_vector: TILE rejects bfloat8_b, rejects
        // any front-padding on any axis (the manifest's dropped_codegen_reasons
        // documents TILE front-padding as never generated -- the copied
        // reader_tile_interleaved_unified.cpp's SEQ_PAD path can address
        // front-padded tile coordinates, but the kernel/manifest pairing was
        // only ever swept and proven for back-only padding), and rejects any
        // H/W back-pad that isn't a whole-tile multiple (sub-tile back-pad is
        // the manifest's documented real-kernel-limit out-of-scope case: the
        // reader only ever emits whole pad tiles, never partial-tile pad
        // bytes within a boundary tile).
        if (dtype == tt::tt_metal::DataType::BFLOAT8_B) {
            return false;
        }
        if (front_n != 0 || front_c != 0 || front_h != 0 || front_w != 0) {
            return false;
        }
        if (output_padded_shape[2] % tt::constants::TILE_HEIGHT != 0 ||
            output_padded_shape[3] % tt::constants::TILE_WIDTH != 0) {
            return false;
        }
        (void)N;
        (void)C;
        (void)H;
        return true;
    }
    return false;
}

bool is_demoted(
    const Tensor& /*input*/,
    const ttnn::Shape& /*output_padded_shape*/,
    const std::array<uint32_t, 4>& /*front*/,
    const tt::tt_metal::MemoryConfig& /*output_mem_config*/) {
    // v1 stub: demotion analysis is out of scope for this port. The routing
    // extension point stays here for a future perf-regression finding.
    return false;
}

}  // namespace ttnn::operations::data_movement::pad_codegen
