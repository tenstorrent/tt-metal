#pragma once

#include <cmath>

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
    using std_type = float;
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

FORCE_INLINE uint32_t portable_ilogb(uint32_t x) {
    // using std::isinf;
    // using std::isnan;
    using std::frexp;

    // if (isnan(x))  return FP_ILOGBNAN;                     // NaN
    // if (isinf(x))  return std::numeric_limits<int>::max(); // +/-inf
    // if (x == uint32_t(0)) return FP_ILOGB0;                       // +/-0

    int exp = 0;
    // frexp: x = m * 2^(exp), with 0.5 <= |m| < 1 (if x!=0)
    // ilogb (base 2) = exp - 1
    (void)frexp(x > uint32_t(0) ? x : -x, &exp);
    return exp - 1;
}

FORCE_INLINE uint32_t get_coord_from_tile_xy(uint32_t read_i, uint32_t write_i) {
    return ((write_i & 0x10) << 5)    // y_hi * 512
           | ((read_i & 0x10) << 4)   // x_hi * 256
           | ((write_i & 0x0F) << 4)  // y_lo * 16
           | (read_i & 0x0F);
}

FORCE_INLINE uint32_t get_tile_id(
    uint32_t depth_blocks_num,
    uint32_t height_blocks_num,
    uint32_t channels_blocks_num,
    uint32_t inner_tile_stride,
    uint32_t channels_slice_i,
    uint32_t row_block_i,
    uint32_t column_block_i,
    uint32_t block_depth = 32) {
    const uint32_t tensor_face_block_size = channels_blocks_num * height_blocks_num;
    const uint32_t tensor_face_thickness = tensor_face_block_size * block_depth;
    const uint32_t block_first_tile_id =
        tensor_face_thickness * column_block_i + row_block_i * channels_blocks_num + channels_slice_i;
    const uint32_t tile_id = block_first_tile_id + inner_tile_stride * tensor_face_block_size;
    // DPRINT << "TENSOR_FACE_BLOCK_SIZE: " << tensor_face_block_size << ", BLOCK_FIRST_TILE_ID: " <<
    // block_first_tile_id << ", TILE_ID: " << tile_id << ENDL(); DPRINT << "TILE_ID: " << tile_id << ENDL();
    return tile_id;
}

FORCE_INLINE uint32_t block_depth_ceil(uint32_t value, uint32_t block_depth = 32) {
    return (value + block_depth - 1) / block_depth;
}

// total tiles with double buffering (apart from 32t CBs, for which there is not enough memory) and default 32
// block_size: 204, each tile: 4 KB, total: 808 KB for 4-byte types per core.
template <typename InputAccessorArgs, typename OutputAccessorArgs>
struct IntImgCTAs {
    const uint32_t start_cb;            // 2 tiles
    const uint32_t input_cb;            // 2 tiles
    const uint32_t acc_cb;              // 2 tiles
    const uint32_t cumsum_stage_0_cb;   // `block_size` tiles
    const uint32_t cumsum_stage_1_cb;   // `block_size` tiles
    const uint32_t cumsum_stage_2_cb;   // `block_size` tiles
    const uint32_t cumsum_stage_3_cb;   // `block_size` tiles
    const uint32_t output_cb;           // 2 tiles
    const uint32_t axis_2_buffer_cb;    // 2 tiles: covers entire propagation
    const uint32_t axis_3_buffer_0_cb;  // `block_size` tiles: each tile is spawned from broadcasting the last row of
                                        // upper block across all rows of a given tile - for the time being, their
                                        // spawning is forced to be done in the reader kernel.
    const uint32_t axis_3_buffer_1_cb;  // dual channel communication with the writer kernel is comprehensive and
                                        // properly synchronizes writer and compute kernels.
    const uint32_t tile_height;
    const uint32_t tile_width;
    const uint32_t block_depth;   // usually 32
    const uint32_t num_channels;  // axis 4/4
    const uint32_t input_height;  // axis 3/4
    const uint32_t input_depth;   // axis 2/4
    const uint32_t num_batches;   // axis 1/4
    const InputAccessorArgs input_args;
    const OutputAccessorArgs output_args;  // reused for reading upper block for propagation.
};

FORCE_INLINE constexpr auto get_ctas() {
    constexpr auto input_args = TensorAccessorArgs<18>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    return IntImgCTAs<decltype(input_args), decltype(output_args)>{
        get_compile_time_arg_val(0),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(17),
        input_args,
        output_args,
    };
}
