// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "sfpi.h"
#include "tensor_shape.h"

namespace llk::sfpu
{

using namespace ckernel;

namespace detail
{

/**
 * @brief Whether DEST can be read through sfpi vector type V.
 *
 * Mirrors the `operator T()` overloads on sfpi::impl_::vReg_; keep in sync with it. Not derived
 * from std::is_base_of: the narrow types derive from the full-width ones, so vUInt8 IS-A vUInt.
 */
template <typename V>
constexpr bool is_vtype_readable = std::is_same_v<V, sfpi::vFloat> || std::is_same_v<V, sfpi::vInt> || std::is_same_v<V, sfpi::vUInt> ||
                                   std::is_same_v<V, sfpi::vSMag> || std::is_same_v<V, sfpi::vFloat16a> || std::is_same_v<V, sfpi::vFloat16b> ||
                                   std::is_same_v<V, sfpi::vMag> || std::is_same_v<V, sfpi::vUInt16> || std::is_same_v<V, sfpi::vSMag16>;

/**
 * @brief Whether DEST can be written through sfpi vector type V.
 *
 * A strict superset of @ref is_vtype_readable: vUInt8 and vSMag8 are write-only.
 */
template <typename V>
constexpr bool is_vtype_writable = is_vtype_readable<V> || std::is_same_v<V, sfpi::vUInt8> || std::is_same_v<V, sfpi::vSMag8>;

/**
 * @brief Whether V is a valid SFPI vector type (readable or writable).
 */
template <typename V>
constexpr bool is_vtype = is_vtype_readable<V> || is_vtype_writable<V>;

} // namespace detail

/**
 * @brief A tensor resident in DEST: the vector type to access it through, its geometry, its index.
 *
 * @tparam V: SFPI vector type.
 * @tparam TS: Tensor shape.
 */
template <typename V, const TensorShape& TS>
struct DestTensor
{
    static_assert(detail::is_vtype<V>, "V must be a valid SFPI vector type");

    using vType                               = V;
    static constexpr const TensorShape& shape = TS;

    std::uint32_t index;

    constexpr DestTensor(const std::uint32_t idx) : index(idx)
    {
    }

    /// Whether Dest can be read through vType.
    static constexpr bool readable()
    {
        return detail::is_vtype_readable<vType>;
    }

    /// Whether Dest can be written through vType.
    static constexpr bool writable()
    {
        return detail::is_vtype_writable<vType>;
    }
};

namespace detail
{

/// Whether T is a @ref DestTensor specialization.
template <typename T>
struct is_dest_tensor : std::false_type
{
};

template <typename V, const TensorShape& TS>
struct is_dest_tensor<DestTensor<V, TS>> : std::true_type
{
};

template <typename T>
constexpr bool is_dest_tensor_v = is_dest_tensor<T>::value;

} // namespace detail

constexpr std::uint32_t DEST_VECTOR_ROWS     = 4;
constexpr std::uint32_t DEST_VECTOR_ROW_STEP = 1;

constexpr std::uint32_t DEST_VECTOR_COLS     = 8;
constexpr std::uint32_t DEST_VECTOR_COL_STEP = 2;

} // namespace llk::sfpu
