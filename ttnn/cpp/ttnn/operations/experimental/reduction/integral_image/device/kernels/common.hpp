// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint32_t ONE_TILE{1};
constexpr uint32_t FIRST_TILE{0};
constexpr uint32_t WORKING_REG{0};

// choose the right C++ POD type at compile-time
template <DataFormat df>
struct df_to_std {
    using std_type = void;
};

template <>
struct df_to_std<DataFormat::Float32> {
    using std_type = uint32_t;
};

template <>
struct df_to_std<DataFormat::Float16_b> {
    using std_type = uint16_t;
};

template <>
struct df_to_std<DataFormat::Int32> {
    using std_type = uint32_t;
};

template <>
struct df_to_std<DataFormat::UInt32> {
    using std_type = uint32_t;
};

template <>
struct df_to_std<DataFormat::UInt16> {
    using std_type = uint16_t;
};

template <>
struct df_to_std<DataFormat::UInt8> {
    using std_type = uint8_t;
};

template <DataFormat df>
using std_type_t = typename df_to_std<df>::std_type;

/**
 * @brief RAII guard for safely reading from a Circular Buffer (CB).
 *
 * `ReadCBGuard` automatically manages CB read synchronization in a scoped manner.
 * When constructed, it waits for `tiles` tiles to be available at the CB front
 * (`cb_wait_front`), ensuring data readiness before access. Upon destruction,
 * it automatically pops those tiles from the CB front (`cb_pop_front`),
 * signaling that the tiles have been consumed.
 *
 * This guarantees balanced `cb_wait_front`/`cb_pop_front` calls, even in the
 * presence of early returns or exceptions, preventing CB underflows and race
 * conditions.
 *
 * @note This class is strictly non-copyable and non-movable to prevent any
 *       double-pop or premature release of CB resources.
 *
 * Example usage:
 * @code
 * {
 *     ReadCBGuard guard(cb_id, num_tiles);
 *     // Safely read from CB: data is guaranteed ready.
 *     // ...
 * } // Automatically pops the tiles on scope exit.
 * @endcode
 */
class ReadCBGuard {
    uint32_t cb;
    uint32_t tiles;

public:
    ReadCBGuard(uint32_t cb, uint32_t tiles) : cb{cb}, tiles{tiles} { cb_wait_front(cb, tiles); }
    ~ReadCBGuard() { cb_pop_front(cb, tiles); }

    ReadCBGuard(const ReadCBGuard&) = delete;  // can not allow to touch the object anyhow
    ReadCBGuard(ReadCBGuard&&) = delete;
    ReadCBGuard& operator=(const ReadCBGuard&) = delete;
    ReadCBGuard& operator=(ReadCBGuard&&) = delete;
    template <typename any_type_t>
    operator any_type_t() = delete;
};

/**
 * @brief RAII guard for safely writing to a Circular Buffer (CB).
 *
 * `WriteCBGuard` automatically manages CB write synchronization in a scoped manner.
 * When constructed, it reserves space for `tiles` tiles at the CB back
 * (`cb_reserve_back`), ensuring sufficient room for writing. Upon destruction,
 * it automatically pushes those tiles to the CB back (`cb_push_back`),
 * making them visible to downstream consumers.
 *
 * This ensures balanced `cb_reserve_back`/`cb_push_back` calls and prevents
 * buffer overflows or mismatched producer-consumer behavior.
 *
 * @note Like `ReadCBGuard`, this class is non-copyable and non-movable to ensure
 *       one-to-one ownership of the CB reservation.
 *
 * Example usage:
 * @code
 * {
 *     WriteCBGuard guard(cb_id, num_tiles);
 *     // Safely write to CB: space is guaranteed reserved.
 *     // ...
 * } // Automatically pushes tiles to the CB on scope exit.
 * @endcode
 */
class WriteCBGuard {
    uint32_t cb;
    uint32_t tiles;

public:
    WriteCBGuard(uint32_t cb, uint32_t tiles) : cb{cb}, tiles{tiles} { cb_reserve_back(cb, tiles); }
    ~WriteCBGuard() { cb_push_back(cb, tiles); }

    WriteCBGuard(const WriteCBGuard&) = delete;  // can not allow to touch the object anyhow
    WriteCBGuard(WriteCBGuard&&) = delete;
    WriteCBGuard& operator=(const WriteCBGuard&) = delete;
    WriteCBGuard& operator=(WriteCBGuard&&) = delete;
    template <typename any_type_t>
    operator any_type_t() = delete;
};

// this function targets a correct index within a tile given the X (read_i) and Y (write_i) coordinates
FORCE_INLINE uint32_t get_coord_from_tile_xy(uint32_t read_i, uint32_t write_i) {
    return ((write_i & 0x10) << 5)    // y_hi * 512
           | ((read_i & 0x10) << 4)   // x_hi * 256
           | ((write_i & 0x0F) << 4)  // y_lo * 16
           | (read_i & 0x0F);
}

FORCE_INLINE uint32_t get_tile_id(
    uint32_t height_blocks_num,
    uint32_t channels_blocks_num,
    uint32_t inner_tile_stride,
    uint32_t channels_slice_i,
    uint32_t row_block_i,
    uint32_t column_block_i,
    uint32_t generic_block_depth = 48) {
    const uint32_t tensor_face_block_size = channels_blocks_num * height_blocks_num;
    const uint32_t tensor_face_thickness = tensor_face_block_size * generic_block_depth;
    const uint32_t block_first_tile_id =
        tensor_face_thickness * column_block_i + row_block_i * channels_blocks_num + channels_slice_i;
    const uint32_t tile_id = block_first_tile_id + inner_tile_stride * tensor_face_block_size;
    return tile_id;
}

FORCE_INLINE constexpr uint32_t ceil(uint32_t value, uint32_t block_depth) {
    return (value + block_depth - 1) / block_depth;
}
