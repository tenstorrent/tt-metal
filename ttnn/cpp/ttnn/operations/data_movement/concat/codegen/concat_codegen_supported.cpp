// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_supported.hpp"

#include <algorithm>
#include <initializer_list>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

namespace {

bool dtype_in_scope(tt::tt_metal::DataType dtype) {
    return dtype == tt::tt_metal::DataType::BFLOAT16 || dtype == tt::tt_metal::DataType::INT32 ||
           dtype == tt::tt_metal::DataType::UINT32;
}

// Projected per-core CB fit for the non-width RM builders (build_concat_rm /
// build_concat_rm_nonwidth_nway): one CB sized to the largest of every
// input's and the projected output's aligned RM page. Mirrors
// concat_codegen_program_factory.cpp's create_descriptor_rm{,_nonwidth_nway}
// so the gate and the factory cannot drift.
bool nonwidth_cb_fits(const std::vector<Tensor>& input_tensors, const tt::tt_metal::MemoryConfig& output_mem_config) {
    tt::tt_metal::IDevice* device = input_tensors[0].device();
    const uint32_t stick_size = input_tensors[0].logical_shape()[-1] * input_tensors[0].element_size();
    const uint32_t out_alignment = device->allocator()->get_alignment(output_mem_config.buffer_type());
    uint32_t cb_page = tt::align(stick_size, out_alignment);
    for (const auto& t : input_tensors) {
        cb_page = std::max(cb_page, static_cast<uint32_t>(t.buffer()->aligned_page_size()));
    }
    return ttnn::prim::plan_concat_batch(
               cb_page, ttnn::prim::kConcatNonWidthBatch, ttnn::prim::concat_l1_budget(device))
        .has_value();
}

// Projected per-core CB fit for the width RM builders (build_concat_rm_width /
// build_concat_rm_width_nway): the write-batched output CB plus a fixed-size
// scratch CB. Mirrors concat_codegen_program_factory.cpp's
// create_descriptor_rm_width{,_nway}.
bool width_cb_fits(const std::vector<Tensor>& input_tensors, const tt::tt_metal::MemoryConfig& output_mem_config) {
    tt::tt_metal::IDevice* device = input_tensors[0].device();
    uint32_t out_width = 0;
    uint32_t scratch_page = 0;
    for (const auto& t : input_tensors) {
        out_width += t.logical_shape()[-1];
        scratch_page = std::max(scratch_page, static_cast<uint32_t>(t.buffer()->aligned_page_size()));
    }
    if (input_tensors.size() == 2) {
        // build_concat_rm_width's scratch CB carries an extra L1-granularity
        // margin the 2-tensor kernel needs; the N-way kernel does not.
        scratch_page += device->allocator()->get_alignment(tt::tt_metal::BufferType::L1);
    }
    const uint32_t out_stick = out_width * input_tensors[0].element_size();
    const uint32_t out_alignment = device->allocator()->get_alignment(output_mem_config.buffer_type());
    const uint32_t out_page = tt::align(out_stick, out_alignment);

    const uint64_t l1_budget = ttnn::prim::concat_l1_budget(device);
    if (scratch_page > l1_budget) {
        return false;
    }
    return ttnn::prim::plan_concat_batch(out_page, ttnn::prim::kConcatWidthWriteBatch, l1_budget - scratch_page)
        .has_value();
}

bool shapes_equal(
    const std::vector<Tensor>& input_tensors, std::initializer_list<std::initializer_list<uint32_t>> expected) {
    if (input_tensors.size() != expected.size()) {
        return false;
    }
    auto expected_it = expected.begin();
    for (size_t i = 0; i < input_tensors.size(); ++i, ++expected_it) {
        const auto& shape = input_tensors[i].logical_shape();
        if (static_cast<size_t>(shape.rank()) != expected_it->size()) {
            return false;
        }
        int j = 0;
        for (uint32_t extent : *expected_it) {
            if (shape[j] != extent) {
                return false;
            }
            ++j;
        }
    }
    return true;
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
    TT_THROW("Unknown concat implementation '{}': expected 'auto', 'native', or 'codegen'", implementation);
}

bool supported_by_codegen(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (input_tensors.size() < 2 || input_tensors.size() > ttnn::prim::kConcatMaxNwayInputs) {
        return false;
    }
    if (output_mem_config.is_sharded()) {
        return false;
    }
    const Tensor& first = input_tensors[0];
    if (first.memory_config().is_sharded()) {
        return false;
    }
    if (first.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        return false;
    }
    if (!dtype_in_scope(first.dtype())) {
        return false;
    }
    const auto dtype = first.dtype();
    const uint32_t ndim = first.logical_shape().rank();
    if (dim >= ndim) {
        return false;
    }
    for (const auto& t : input_tensors) {
        if (t.layout() != ttnn::ROW_MAJOR_LAYOUT || t.dtype() != dtype || t.memory_config().is_sharded() ||
            t.logical_shape().rank() != ndim) {
            return false;
        }
    }

    if (input_tensors.size() > 2) {
        // The N-way readers share one TensorAccessorArgs ABI across every
        // input (spec.py's build_concat_rm_{width,nonwidth}_nway), so every
        // input must sit in the exact same memory configuration.
        const auto& first_mem = first.memory_config();
        for (const auto& t : input_tensors) {
            if (t.memory_config() != first_mem) {
                return false;
            }
        }
    }

    const bool is_width = (dim == ndim - 1);
    return is_width ? width_cb_fits(input_tensors, output_mem_config)
                    : nonwidth_cb_fits(input_tensors, output_mem_config);
}

bool is_demoted(const std::vector<Tensor>& input_tensors, uint32_t dim) {
    // Ungeneralized: no predicate relating these shapes to their measured
    // device-time regression was found, so this is an exact-match floor per
    // the manifest's demoted-cases list (all 3 in-scope dtypes), not a
    // general condition.
    if (dim == 2 && shapes_equal(input_tensors, {{1, 32, 32}, {1, 32, 64}, {1, 32, 32}})) {
        return true;
    }
    if (dim == 1 && shapes_equal(input_tensors, {{32, 32}, {32, 64}, {32, 96}})) {
        return true;
    }
    return false;
}

bool supported_execution_controls(unsigned int groups, const std::optional<ttnn::CoreRangeSet>& sub_core_grids) {
    // sub_core_grids restricts native's default ConcatProgramFactory to a
    // caller-chosen core subset; no ConcatCodegen builder honours it (every
    // builder places work over the full compute_with_storage_grid_size()).
    // groups > 1 has no defined effect on native's own interleaved path
    // either (only the sharded factories consume it, and this port's scope
    // already excludes sharded), but ConcatCodegenParams carries no such
    // field, so route it to native defensively rather than silently drop it.
    return groups == 1 && !sub_core_grids.has_value();
}

}  // namespace ttnn::operations::data_movement::concat_codegen
