// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#if defined(CKL_ELTWISE_BINARY_SFPU_BASIC)
#include "api/compute/eltwise_binary_sfpu.h"
#endif

#if defined(CKL_ELTWISE_BINARY_SFPU_MINMAX)
#include "api/compute/binary_max_min.h"
#endif

#if defined(CKL_ELTWISE_BINARY_SFPU_INT)
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/div_int32_floor.h"
#endif

#if defined(CKL_ELTWISE_BINARY_SFPU_EXTENDED)
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/binary_shift.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_comp.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/atan2.h"
#endif

namespace compute_kernel_lib {

#if defined(CKL_ELTWISE_BINARY_SFPU_BASIC)
template <Dst In0, Dst In1, Dst Out>
struct AddBinary : BinaryOp<AddBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { add_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        add_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct SubBinary : BinaryOp<SubBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { sub_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        sub_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct MulBinary : BinaryOp<MulBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { mul_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        mul_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct DivBinary : BinaryOp<DivBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { div_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        div_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};
#endif

#if defined(CKL_ELTWISE_BINARY_SFPU_MINMAX)
// binary_max_tile / binary_min_tile — SFPU two-DEST max/min into third DEST slot.
template <Dst In0, Dst In1, Dst Out>
struct BinaryMax : BinaryOp<BinaryMax<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_max_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_max_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct BinaryMin : BinaryOp<BinaryMin<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_min_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_min_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};
#endif

#if defined(CKL_ELTWISE_BINARY_SFPU_INT)
// ============================================================================
// Integer-typed binary ops (DataFormat-templated).
// One struct per OpConfig::SfpuBinaryOp integer variant.
// ============================================================================

template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct AddIntBinary : BinaryOp<AddIntBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { add_int_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        add_int_tile<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct SubIntBinary : BinaryOp<SubIntBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { sub_int_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        sub_int_tile<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct MulIntBinary : BinaryOp<MulIntBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { mul_int_tile_init<DF>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        mul_int_tile<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct RsubIntBinary : BinaryOp<RsubIntBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { rsub_int_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        rsub_int_tile<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// Integer division family (int32 only — no DataFormat template).
template <Dst In0, Dst In1, Dst Out>
struct DivInt32Binary : BinaryOp<DivInt32Binary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { div_int32_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        div_int32_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct DivInt32FloorBinary : BinaryOp<DivInt32FloorBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { div_int32_floor_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        div_int32_floor_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct DivInt32TruncBinary : BinaryOp<DivInt32TruncBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { div_int32_trunc_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        div_int32_trunc_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};
#endif

#if defined(CKL_ELTWISE_BINARY_SFPU_EXTENDED)
// Remainder / fmod (float + int32 variants).
template <Dst In0, Dst In1, Dst Out>
struct RemainderBinary : BinaryOp<RemainderBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { remainder_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        remainder_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct RemainderInt32Binary : BinaryOp<RemainderInt32Binary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { remainder_int32_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        remainder_int32_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct FmodBinary : BinaryOp<FmodBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { fmod_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        fmod_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct FmodInt32Binary : BinaryOp<FmodInt32Binary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { fmod_int32_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        fmod_int32_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// Power (float pow), RsubBinary (float reverse subtract).
template <Dst In0, Dst In1, Dst Out>
struct PowerBinary : BinaryOp<PowerBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { power_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        power_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct RsubBinary : BinaryOp<RsubBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { rsub_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        rsub_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// GCD / LCM (int).
template <Dst In0, Dst In1, Dst Out>
struct GcdBinary : BinaryOp<GcdBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { gcd_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        gcd_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct LcmBinary : BinaryOp<LcmBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { lcm_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        lcm_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// Bit shift (DataFormat-templated).
template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct LeftShiftBinary : BinaryOp<LeftShiftBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_shift_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_left_shift_tile<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct RightShiftBinary : BinaryOp<RightShiftBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_shift_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_right_shift_tile<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct LogicalRightShiftBinary : BinaryOp<LogicalRightShiftBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_shift_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_logical_right_shift_tile<DF>(
            to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// Bitwise (DataFormat-templated).
template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct BitwiseAndBinary : BinaryOp<BitwiseAndBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_bitwise_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        bitwise_and_binary_tile<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct BitwiseOrBinary : BinaryOp<BitwiseOrBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_bitwise_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        bitwise_or_binary_tile<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <DataFormat DF, Dst In0, Dst In1, Dst Out>
struct BitwiseXorBinary : BinaryOp<BitwiseXorBinary<DF, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_bitwise_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        bitwise_xor_binary_tile<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};
#endif

#if defined(CKL_ELTWISE_BINARY_SFPU_MINMAX)
// Integer max/min variants (BinaryMax/BinaryMin above cover float).
template <Dst In0, Dst In1, Dst Out>
struct BinaryMaxInt32 : BinaryOp<BinaryMaxInt32<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_max_int32_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_max_int32_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct BinaryMaxUint32 : BinaryOp<BinaryMaxUint32<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_max_uint32_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_max_uint32_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct BinaryMinInt32 : BinaryOp<BinaryMinInt32<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_min_int32_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_min_int32_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct BinaryMinUint32 : BinaryOp<BinaryMinUint32<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_min_uint32_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_min_uint32_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};
#endif

#if defined(CKL_ELTWISE_BINARY_SFPU_EXTENDED)
// Misc float ops.
template <Dst In0, Dst In1, Dst Out>
struct XlogyBinary : BinaryOp<XlogyBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { xlogy_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        xlogy_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0, Dst In1, Dst Out>
struct Atan2Binary : BinaryOp<Atan2Binary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { atan2_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        atan2_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// Comparison ops — float-only variants (output 0/1 mask).
#define _CKL_BIN_CMP_FLOAT(Name, llk_call, init_call)                                                  \
    template <Dst In0, Dst In1, Dst Out>                                                               \
    struct Name : BinaryOp<Name<In0, In1, Out>, In0, In1, Out> {                                       \
        static ALWI void init() { init_call(); }                                                       \
        static ALWI void exec_impl(uint32_t slot_offset) {                                             \
            llk_call(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset); \
        }                                                                                              \
    };

_CKL_BIN_CMP_FLOAT(LtBinary, lt_binary_tile, lt_binary_tile_init)
_CKL_BIN_CMP_FLOAT(GtBinary, gt_binary_tile, gt_binary_tile_init)
_CKL_BIN_CMP_FLOAT(LeBinary, le_binary_tile, le_binary_tile_init)
_CKL_BIN_CMP_FLOAT(GeBinary, ge_binary_tile, ge_binary_tile_init)
_CKL_BIN_CMP_FLOAT(EqBinary, eq_binary_tile, eq_binary_tile_init)
_CKL_BIN_CMP_FLOAT(NeBinary, ne_binary_tile, ne_binary_tile_init)

#undef _CKL_BIN_CMP_FLOAT

// Comparison ops — integer (DataFormat-templated init+call).
#define _CKL_BIN_CMP_INT(Name, llk_call, init_call)                                                        \
    template <DataFormat DF, Dst In0, Dst In1, Dst Out>                                                    \
    struct Name : BinaryOp<Name<DF, In0, In1, Out>, In0, In1, Out> {                                       \
        static ALWI void init() { init_call<DF>(); }                                                       \
        static ALWI void exec_impl(uint32_t slot_offset) {                                                 \
            llk_call<DF>(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset); \
        }                                                                                                  \
    };

_CKL_BIN_CMP_INT(LtIntBinary, lt_int_tile, lt_int_tile_init)
_CKL_BIN_CMP_INT(GtIntBinary, gt_int_tile, gt_int_tile_init)
_CKL_BIN_CMP_INT(LeIntBinary, le_int_tile, le_int_tile_init)
_CKL_BIN_CMP_INT(GeIntBinary, ge_int_tile, ge_int_tile_init)

#undef _CKL_BIN_CMP_INT

// Quant / Requant / Dequant are not currently included. (Their init() takes a runtime zero_point,
// but that is NOT the blocker — the chain dispatches init on the element instance, so a
// runtime-stateful init is already supported; Dropout does exactly this with its seed.)
#endif

}  // namespace compute_kernel_lib
