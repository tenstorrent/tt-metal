// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file buffer_helpers.hpp
 * @brief Buffer-type adapter for kernel-side helpers that need to work
 *        uniformly with circular buffers (Gen1) and dataflow buffers (Gen1/Gen2).
 *
 * Background:
 *   - Gen1 (WH/BH) has only circular buffers (CBs). Kernels operate on them
 *     either via raw `uint32_t` CB ids or via the `experimental::CircularBuffer`
 *     wrapper (defined in tt_metal/hw/inc/experimental/circular_buffer.h).
 *   - Metal 2.0 / Quasar introduces dataflow buffers (DFBs). Kernels access
 *     them via the `experimental::DataflowBuffer` wrapper. On Gen1 a DFB id IS
 *     the underlying CB id; on Gen2 it indexes a distinct DFB hardware unit.
 *
 * The helper libraries in `kernel_lib/` (e.g. `compute_kernel_lib::reduce`,
 * `dataflow_kernel_lib::prepare_reduce_scaler`) historically took raw CB ids and
 * built `experimental::CircularBuffer` wrappers internally. That made them
 * Gen1-only. To support Quasar without scattering `#ifdef ARCH_QUASAR`
 * branches across kernel sources and the helpers themselves, callers now pass
 * either:
 *   - a raw `uint32_t` (CB id, for legacy kernels),
 *   - an `experimental::CircularBuffer` reference, or
 *   - an `experimental::DataflowBuffer` reference,
 * and the helpers normalize internally via `BufferRef<T>` below. The wrapper
 * exposes a uniform sync API (wait_front / pop_front / reserve_back /
 * push_back) and an `id()` accessor that returns a `uint32_t` for use with
 * the LLK calls (`reduce_init`, `reduce_tile`, `pack_tile`, ...) that still
 * take raw ids.
 *
 * IMPORTANT: this is a kernel-side header. It is included from .cpp files
 * that compile for BRISC / NCRISC / TRISC. We deliberately avoid std::*
 * machinery beyond <type_traits>.
 */

#include <cstdint>
#include <type_traits>

#include "experimental/circular_buffer.h"
#include "experimental/dataflow_buffer.h"

// Defensive fallback: ALWI is normally pulled in via compute_kernel_api.h
// (compute kernels) or one of the dataflow-side helpers (DM kernels). This
// header is included from both, so we provide the same definition if it
// hasn't been defined yet.
#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

namespace kernel_lib {

namespace detail {

// Tag the argument category at compile time so BufferRef<T> can pick the
// right specialization without partial specialization on cv/ref qualifiers.
enum class BufferKind { ID, CIRCULAR_BUFFER, DATAFLOW_BUFFER };

template <typename T>
constexpr BufferKind buffer_kind_v =
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, ::experimental::CircularBuffer>
        ? BufferKind::CIRCULAR_BUFFER
        : (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, ::experimental::DataflowBuffer>
               ? BufferKind::DATAFLOW_BUFFER
               : BufferKind::ID);

}  // namespace detail

/**
 * @brief Uniform handle wrapping any buffer-like argument passed to a kernel helper.
 *
 * Specialized for three input categories:
 *   - `uint32_t` (CB id): owns a `CircularBuffer` instance constructed from the id.
 *     Cheap — `CircularBuffer` is just a `uint32_t` member with inlined methods.
 *   - `experimental::CircularBuffer&`: holds a reference to the caller's wrapper.
 *   - `experimental::DataflowBuffer&`: holds a reference to the caller's wrapper.
 *
 * Methods (all inline-forwarded to the underlying buffer):
 *   - `id()` -> `uint32_t`        — buffer id for LLK calls
 *   - `wait_front(uint32_t n)`
 *   - `pop_front(uint32_t n)`
 *   - `reserve_back(uint32_t n)`
 *   - `push_back(uint32_t n)`
 *
 * Usage in a helper:
 *   template <typename InputBuf, typename ScalerBuf, typename OutputBuf, ...>
 *   void reduce(InputBuf input, ScalerBuf scaler, OutputBuf output, ...) {
 *       BufferRef in{input};
 *       BufferRef sc{scaler};
 *       BufferRef ou{output};
 *       const uint32_t in_id = in.id();
 *       in.wait_front(1);
 *       reduce_tile<...>(in_id, sc.id(), 0, 0, 0);  // raw-id LLK call
 *       in.pop_front(1);
 *   }
 */
template <typename T, detail::BufferKind Kind = detail::buffer_kind_v<T>>
class BufferRef;

// Specialization: raw uint32_t CB id. Constructs and owns a CircularBuffer.
template <typename T>
class BufferRef<T, detail::BufferKind::ID> {
public:
    ALWI BufferRef(uint32_t cb_id) : cb_(cb_id) {}

    ALWI uint32_t id() const { return cb_.get_cb_id(); }
    ALWI void wait_front(uint32_t num_pages) { cb_.wait_front(num_pages); }
    ALWI void pop_front(uint32_t num_pages) { cb_.pop_front(num_pages); }
    ALWI void reserve_back(uint32_t num_pages) { cb_.reserve_back(num_pages); }
    ALWI void push_back(uint32_t num_pages) { cb_.push_back(num_pages); }
    // L1 byte-address accessors (UNPACK uses read; PACK uses write).
    ALWI uint32_t get_read_ptr() const { return cb_.get_read_ptr(); }
    ALWI uint32_t get_write_ptr() const { return cb_.get_write_ptr(); }

private:
    ::experimental::CircularBuffer cb_;
};

// Specialization: existing CircularBuffer instance (e.g. user code on Gen1).
template <typename T>
class BufferRef<T, detail::BufferKind::CIRCULAR_BUFFER> {
public:
    ALWI BufferRef(::experimental::CircularBuffer& cb) : cb_(cb) {}

    ALWI uint32_t id() const { return cb_.get_cb_id(); }
    ALWI void wait_front(uint32_t num_pages) { cb_.wait_front(num_pages); }
    ALWI void pop_front(uint32_t num_pages) { cb_.pop_front(num_pages); }
    ALWI void reserve_back(uint32_t num_pages) { cb_.reserve_back(num_pages); }
    ALWI void push_back(uint32_t num_pages) { cb_.push_back(num_pages); }
    ALWI uint32_t get_read_ptr() const { return cb_.get_read_ptr(); }
    ALWI uint32_t get_write_ptr() const { return cb_.get_write_ptr(); }

private:
    ::experimental::CircularBuffer& cb_;
};

// Specialization: existing DataflowBuffer instance (Gen1 with DFB id == CB id, Gen2 real DFB).
template <typename T>
class BufferRef<T, detail::BufferKind::DATAFLOW_BUFFER> {
public:
    ALWI BufferRef(::experimental::DataflowBuffer& dfb) : dfb_(dfb) {}

    ALWI uint32_t id() const { return dfb_.get_id(); }
    ALWI void wait_front(uint32_t num_pages) { dfb_.wait_front(num_pages); }
    ALWI void pop_front(uint32_t num_pages) { dfb_.pop_front(num_pages); }
    ALWI void reserve_back(uint32_t num_pages) { dfb_.reserve_back(num_pages); }
    ALWI void push_back(uint32_t num_pages) { dfb_.push_back(num_pages); }
    ALWI uint32_t get_read_ptr() const { return dfb_.get_read_ptr(); }
    ALWI uint32_t get_write_ptr() const { return dfb_.get_write_ptr(); }

private:
    ::experimental::DataflowBuffer& dfb_;
};

// Deduction guide so callers can write `BufferRef ref{arg}` without naming the type.
template <typename T>
BufferRef(T&&) -> BufferRef<std::remove_reference_t<T>>;

}  // namespace kernel_lib
