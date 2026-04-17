// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Where<df, In0, In1, In2, Out>::init() const { where_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Where<df, In0, In1, In2, Out>::call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const { where_tile<df>(a, b, c, d); }

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Lerp<df, In0, In1, In2, Out>::init() const { lerp_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Lerp<df, In0, In1, In2, Out>::call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const { lerp_tile<df>(a, b, c, d); }

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcmul<df, In0, In1, In2, Out>::init() const { addcmul_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcmul<df, In0, In1, In2, Out>::call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const { addcmul_tile<df>(a, b, c, d, value); }

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcdiv<df, In0, In1, In2, Out>::init() const { addcdiv_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcdiv<df, In0, In1, In2, Out>::call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const { addcdiv_tile<df>(a, b, c, d, value); }

}  // namespace compute_kernel_lib
