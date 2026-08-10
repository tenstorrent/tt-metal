// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

using namespace tt;

namespace {

// TILE layout stores a 32x32 tile as four 16x16 faces:
// face 0 = rows 0-15, cols 0-15; face 1 = rows 0-15, cols 16-31;
// face 2 = rows 16-31, cols 0-15; face 3 = rows 16-31, cols 16-31.
// Within a face, elements are row-major.
static inline uint32_t tile_idx(uint32_t row, uint32_t col) {
    return ((row >> 4) << 1 | (col >> 4)) << 8 | ((row & 15) << 4) | (col & 15);
}

// fp32 sqrt without libm (not linked into dataflow kernels): bit-level seed
// plus Newton iterations. Relative error < 1e-10 for normal floats after 4
// iterations from the ~3.5%-accurate seed.
static inline float qr_sqrt(float x) {
    if (x == 0.0f) {
        return 0.0f;
    }
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    v.u = (v.u + 0x3F800000u) >> 1;
    for (uint32_t i = 0; i < 4; ++i) {
        v.f = 0.5f * (v.f + x / v.f);
    }
    return v.f;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t r_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t q_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t v_cb_id = get_compile_time_arg_val(3);

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t q_addr = get_arg_val<uint32_t>(1);
    uint32_t r_addr = get_arg_val<uint32_t>(2);
    uint32_t m = get_arg_val<uint32_t>(3);
    uint32_t n = get_arg_val<uint32_t>(4);
    const uint32_t k = m < n ? m : n;

    constexpr auto input_args = TensorAccessorArgs<4>();
    const auto input_addrg = TensorAccessor(input_args, input_addr);
    constexpr auto q_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto q_addrg = TensorAccessor(q_args, q_addr);
    constexpr auto r_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    const auto r_addrg = TensorAccessor(r_args, r_addr);

    Noc noc;
    CircularBuffer cb_in(in_cb_id);
    CircularBuffer cb_r(r_cb_id);
    CircularBuffer cb_q(q_cb_id);
    CircularBuffer cb_v(v_cb_id);

    const uint32_t page_bytes = get_local_cb_interface(in_cb_id).fifo_page_size;
    const uint32_t scratch_bytes = get_local_cb_interface(r_cb_id).fifo_page_size;

    cb_in.wait_front(1);
    cb_r.reserve_back(1);
    cb_q.reserve_back(1);
    cb_v.reserve_back(1);

    const float* in_ptr = reinterpret_cast<const float*>(cb_in.get_read_ptr());
    float* r_ptr = reinterpret_cast<float*>(cb_r.get_write_ptr());
    float* q_ptr = reinterpret_cast<float*>(cb_q.get_write_ptr());
    float* v_ptr = reinterpret_cast<float*>(cb_v.get_write_ptr());

    // Copy the input tile into the R scratch, zeroing the padded region.
    for (uint32_t i = 0; i < 1024; ++i) {
        r_ptr[i] = 0.0f;
    }
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            r_ptr[tile_idx(i, j)] = in_ptr[tile_idx(i, j)];
        }
    }

    // Q = identity (m x k).
    for (uint32_t i = 0; i < 1024; ++i) {
        q_ptr[i] = 0.0f;
    }
    for (uint32_t i = 0; i < k; ++i) {
        q_ptr[tile_idx(i, i)] = 1.0f;
    }

    // Forward pass: Householder reflectors that reduce R to upper triangular.
    // v for step j is stored at v_ptr[j * m + i], i in [j, m).
    for (uint32_t step = 0; step < k; ++step) {
        float norm2 = 0.0f;
        for (uint32_t i = step; i < m; ++i) {
            float x = r_ptr[tile_idx(i, step)];
            norm2 += x * x;
        }
        float norm = qr_sqrt(norm2);
        float x0 = r_ptr[tile_idx(step, step)];
        // sign(x0), with sign(0) = 1.
        float s = (x0 >= 0.0f) ? 1.0f : -1.0f;
        float alpha = -s * norm;
        float v0 = x0 - alpha;
        v_ptr[step * m + step] = v0;
        for (uint32_t i = step + 1; i < m; ++i) {
            v_ptr[step * m + i] = r_ptr[tile_idx(i, step)];
        }
        // beta = 2 / vTv, with a zero guard so an all-zero reflector gives
        // H = I (v = 0, so the update is identically zero either way).
        float vTv = v0 * v0;
        for (uint32_t i = step + 1; i < m; ++i) {
            float vv = v_ptr[step * m + i];
            vTv += vv * vv;
        }
        float beta = 2.0f / (vTv == 0.0f ? 1.0f : vTv);

        // R[step:, step:] -= beta * v * (v^T R[step:, step:]).
        for (uint32_t c = step; c < n; ++c) {
            float w = v0 * r_ptr[tile_idx(step, c)];
            for (uint32_t i = step + 1; i < m; ++i) {
                w += v_ptr[step * m + i] * r_ptr[tile_idx(i, c)];
            }
            r_ptr[tile_idx(step, c)] -= beta * v0 * w;
            for (uint32_t i = step + 1; i < m; ++i) {
                r_ptr[tile_idx(i, c)] -= beta * v_ptr[step * m + i] * w;
            }
        }
    }

    // Backward pass: Q = H_{k-1} ... H_0 applied to the identity columns.
    for (uint32_t step = k; step-- > 0;) {
        float v0 = v_ptr[step * m + step];
        float vTv = v0 * v0;
        for (uint32_t i = step + 1; i < m; ++i) {
            float vv = v_ptr[step * m + i];
            vTv += vv * vv;
        }
        float beta = 2.0f / (vTv == 0.0f ? 1.0f : vTv);

        for (uint32_t c = step; c < k; ++c) {
            float w = v0 * q_ptr[tile_idx(step, c)];
            for (uint32_t i = step + 1; i < m; ++i) {
                w += v_ptr[step * m + i] * q_ptr[tile_idx(i, c)];
            }
            q_ptr[tile_idx(step, c)] -= beta * v0 * w;
            for (uint32_t i = step + 1; i < m; ++i) {
                q_ptr[tile_idx(i, c)] -= beta * v_ptr[step * m + i] * w;
            }
        }
    }

    // Write both output tiles to DRAM.
    noc.async_write(
        CoreLocalMem<uint32_t>(cb_q.get_write_ptr()), q_addrg, page_bytes, {}, {.page_id = 0});
    noc.async_write(
        CoreLocalMem<uint32_t>(cb_r.get_write_ptr()), r_addrg, page_bytes, {}, {.page_id = 0});
    noc.async_write_barrier();

    cb_in.pop_front(1);
}
