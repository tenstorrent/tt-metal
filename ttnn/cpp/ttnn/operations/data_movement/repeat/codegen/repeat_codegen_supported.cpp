// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_supported.hpp"

#include <algorithm>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::data_movement::repeat_codegen {

namespace {

// Shared correctness rule for repeating dim `d` (in a tensor of rank `ndim`)
// by `reps` on `layout`/`dtype`, transcribed from codegen_repeat.py's
// invalidate_vector (both the base suite, which wraps the upstream
// sweeps.data_movement.repeat.repeat gate, and the broaden suite). A
// non-repeated dim (reps <= 1) never constrains the call.
//
// TILE: only H (d == ndim-2) or W (d == ndim-1) repeats require tile
// alignment on that axis. RepeatCodegen repeats in tile-page space
// (ceil(H/32) / ceil(W/32) tiles); repeating a tile-page-quantized axis that
// isn't itself tile-aligned makes ceil(dim/32)*reps disagree with
// ceil(dim*reps/32), which shape-mismatches or folds pad rows into live data.
// Repeating N or C never touches this math -- the same tile page (padding
// included) is just duplicated, which is correct regardless of whether H/W
// happen to be sub-tile.
bool single_dim_ok(
    const tt::tt_metal::Shape& shape, tt::tt_metal::DataType dtype, Layout layout, uint32_t d, uint32_t reps) {
    if (reps <= 1) {
        return true;
    }
    const uint32_t ndim = shape.rank();
    if (layout == ttnn::ROW_MAJOR_LAYOUT) {
        if (dtype == tt::tt_metal::DataType::BFLOAT8_B) {
            return false;
        }
        if (dtype == tt::tt_metal::DataType::BFLOAT16 && shape[-1] < 2) {
            return false;
        }
        return true;
    }
    if (layout == ttnn::TILE_LAYOUT) {
        if (d == ndim - 1) {
            return shape[-1] % 32 == 0;
        }
        if (d == ndim - 2) {
            return shape[-2] % 32 == 0;
        }
        return true;
    }
    return false;
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
    TT_THROW("Unknown repeat implementation '{}': expected 'auto', 'native', or 'codegen'", implementation);
}

bool supported_by_codegen(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats) {
    // Sharded input: RepeatCodegen only ever reaches sharded tensors via a
    // native unshard-to-interleaved-DRAM hop that this port does not
    // implement; see the manifest's hand-authored sharded case.
    if (input.memory_config().is_sharded()) {
        return false;
    }
    const auto& shape = input.logical_shape();
    if (shape.rank() != 4 || rep_dim >= 4) {
        return false;
    }
    return single_dim_ok(shape, input.dtype(), input.layout(), rep_dim, num_repeats);
}

bool supported_by_codegen(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims) {
    if (input.memory_config().is_sharded()) {
        return false;
    }
    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    // repeat_codegen's kernels assume a 4D-padded tensor (ops/repeat/spec.py's
    // _page_map / build_repeat_rm_factory); a rank > 4 input has no path here.
    if (repeat_dims.size() != ndim || ndim < 2 || ndim > 4) {
        return false;
    }
    bool any_repeated = std::any_of(repeat_dims.cbegin(), repeat_dims.cend(), [](uint32_t r) { return r > 1; });
    if (!any_repeated) {
        return false;
    }
    const auto dtype = input.dtype();
    const auto layout = input.layout();
    for (uint32_t d = 0; d < ndim; ++d) {
        if (!single_dim_ok(shape, dtype, layout, d, repeat_dims[d])) {
            return false;
        }
    }
    return true;
}

bool is_demoted(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims) {
    const auto& shape = input.logical_shape();
    const auto dtype = input.dtype();
    const auto layout = input.layout();

    auto shape_is = [&](std::initializer_list<uint32_t> s) {
        return shape.rank() == static_cast<uint32_t>(s.size()) && std::equal(s.begin(), s.end(), shape.cbegin());
    };
    auto reps_is = [&](std::initializer_list<uint32_t> r) {
        return repeat_dims.size() == r.size() && std::equal(r.begin(), r.end(), repeat_dims.cbegin());
    };

    if (dtype != tt::tt_metal::DataType::BFLOAT16) {
        return false;
    }

    if (layout == ttnn::TILE_LAYOUT && shape_is({1, 1, 1, 1}) && reps_is({1, 2, 1, 1})) {
        return true;
    }

    if (layout == ttnn::ROW_MAJOR_LAYOUT) {
        // C-dim doubling on small RM tensors: generic's collapsed RM path beats
        // the ported per-dim stick copy on device (native/ported ~1.9x).
        if (reps_is({1, 2, 1, 1}) &&
            (shape_is({1, 2, 8, 16}) || shape_is({1, 2, 10, 20}) || shape_is({1, 2, 12, 24}) ||
             shape_is({1, 2, 14, 28}) || shape_is({1, 2, 16, 32}) || shape_is({1, 2, 20, 40}))) {
            return true;
        }
        // Multi-dim RM repeats that generic wins on device.
        if (shape_is({1, 2, 4, 4}) && reps_is({1, 3, 6, 12})) {
            return true;
        }
        if (shape_is({1, 2, 6, 12}) && reps_is({1, 3, 10, 20})) {
            return true;
        }
    }

    return false;
}

}  // namespace ttnn::operations::data_movement::repeat_codegen
