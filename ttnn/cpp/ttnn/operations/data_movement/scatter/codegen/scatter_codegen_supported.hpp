// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::scatter {

// Correctness gate: true iff the codegen prim can produce a result matching native's contract for
// this input/index/src combination, scatter axis and reduction mode. Consulted by ttnn::scatter()'s
// routing, by scatter_force_codegen(), and by the codegen prim's own validation step.
//
// `dim` is the (possibly negative) scatter axis in the tensors' OWN rank -- the router and
// scatter_force_codegen() call this ahead of the transpose-to-last-dim / 4D-fold normalization, with
// the caller's raw dim, so an out-of-range dim falls through to native's error instead of indexing
// out of bounds here. The codegen prim's validate calls this on the already-normalized tensors it
// actually holds, where the scatter axis is always the last dim; it passes -1 to say so.
//
// reduction_mode is 0 (no reduction), 1 (add) or 2 (multiply): ttnn::scatter()'s own validate_inputs
// rejects any other value before routing is ever consulted, so max/min are never observed here.
bool supported_by_codegen(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    uint32_t reduction_mode);

// The half of the call contract supported_by_codegen() cannot see: the caller-controlled output
// placement and, when the caller preallocates a destination, the spec that destination carries.
// False means the codegen factories cannot honour what the caller asked for -- auto must route to
// native and forced codegen must fail rather than silently write through the mismatch. Not a perf
// question, so it is kept out of is_demoted() too.
bool supported_execution_controls(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor);

// The shape the codegen dispatch scatters over: the caller's logical shape with the scatter axis
// swapped to the last position (the pre-scatter transpose) and the rank padded up to 4, exactly as
// pre_scatter_transform_tensor() shapes each operand. Shape-only, so routing can ask without
// moving data.
ttnn::Shape codegen_working_shape(const ttnn::Shape& logical_shape, int32_t dim);

// Whether a TILE call takes the untilize -> per-stick ROW_MAJOR scatter -> tilize detour instead of
// the TILE factory. The TILE factories split per-core work by tile row, so a low tile-row count
// leaves most of the grid idle whatever the row width; below kRowMajorRerouteMaxHt the RM factory's
// per-stick split reaches far more cores, provided the untilized stick is bounded, NOC-aligned and
// fits L1. `working_input`/`working_index` are the operands' codegen_working_shape()s; the tensors
// supply dtype, element size, device and placement, none of which the transpose changes. The one
// definition both the dispatch and is_demoted() read: a case this detour serves never pays the
// TILE factories' padded-row cost that is_demoted() otherwise guards against.
bool prefers_row_major_strategy(
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    const ttnn::Shape& working_input,
    const ttnn::Shape& working_index);

// Perf gate consulted only by ttnn::scatter()'s routing, and only when supported_by_codegen() is
// true: true means fall back to native despite codegen support. `dim` is the caller's raw
// (pre-normalization) scatter axis, matching supported_by_codegen()'s convention.
bool is_demoted(const Tensor& input_tensor, int32_t dim, const Tensor& index_tensor, const Tensor& src_tensor);

}  // namespace ttnn::operations::data_movement::scatter
