#pragma once

constexpr uint32_t ONE_TILE{1};
constexpr uint32_t FIRST_TILE{0};
constexpr uint32_t WORKING_REG{0};
constexpr uint32_t SEMAPHORE_BUSY{0};
constexpr uint32_t SEMAPHORE_READY{1};

constexpr uint32_t FIRST_ROW_ORD = 0;
constexpr uint32_t LAST_ROW_ORD = 32 - 1;

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

// aggressively cutting lines in the main logic (kek DABlyu)
// CB read RAII
class ReadCBGuard {
    uint32_t cb;
    uint32_t tiles;

public:
    ReadCBGuard(uint32_t cb, uint32_t tiles) : cb{cb}, tiles{tiles} { cb_wait_front(cb, tiles); }
    ~ReadCBGuard() { cb_pop_front(cb, tiles); }

    ReadCBGuard(const ReadCBGuard&) = delete;  // do not allow to touch the object anyhow
    ReadCBGuard(ReadCBGuard&&) = delete;
    ReadCBGuard& operator=(const ReadCBGuard&) = delete;
    ReadCBGuard& operator=(ReadCBGuard&&) = delete;
    template <typename any_type_t>
    operator any_type_t() = delete;
};

// CB write RAII
class WriteCBGuard {
    uint32_t cb;
    uint32_t tiles;

public:
    WriteCBGuard(uint32_t cb, uint32_t tiles) : cb{cb}, tiles{tiles} { cb_reserve_back(cb, tiles); }
    ~WriteCBGuard() { cb_push_back(cb, tiles); }

    WriteCBGuard(const WriteCBGuard&) = delete;  // do not allow to touch the object anyhow
    WriteCBGuard(WriteCBGuard&&) = delete;
    WriteCBGuard& operator=(const WriteCBGuard&) = delete;
    WriteCBGuard& operator=(WriteCBGuard&&) = delete;
    template <typename any_type_t>
    operator any_type_t() = delete;
};

// L1 -> DRAM
template <typename addr_gen_t>
FORCE_INLINE void write_to_dram(
    uint32_t cb, const addr_gen_t& addr_gtor, uint32_t write_tile_id, uint32_t num_tiles = ONE_TILE) {
    ReadCBGuard read_guard{cb, num_tiles};

    uint32_t l1_read_addr{get_read_ptr(cb)};
    noc_async_write_tile(write_tile_id, addr_gtor, l1_read_addr);
    noc_async_write_barrier();
}

// DRAM -> L1
template <typename addr_gen_type>
FORCE_INLINE void load_to_cb(
    uint32_t cb, const addr_gen_type& addr_gtor, uint32_t read_tile_id, uint32_t num_tiles = ONE_TILE) {
    WriteCBGuard write_guard{cb, num_tiles};

    uint32_t l1_write_addr{get_write_ptr(cb)};
    noc_async_read_tile(read_tile_id, addr_gtor, l1_write_addr);
    noc_async_read_barrier();
}

FORCE_INLINE uint32_t get_coord_from_tile_xy(uint32_t read_i, uint32_t write_i) {
    return ((write_i & 0x10) << 5)    // y_hi * 512
           | ((read_i & 0x10) << 4)   // x_hi * 256
           | ((write_i & 0x0F) << 4)  // y_lo * 16
           | (read_i & 0x0F);
}

FORCE_INLINE uint32_t get_tile_id(
    // uint32_t depth_blocks_num,
    uint32_t num_tiles_in_column,
    uint32_t num_tiles_along_channels,
    uint32_t row_tile_stride,
    uint32_t channels_slice_i,
    uint32_t row_chunk_i) {
    const uint32_t tensor_face_tile_size = num_tiles_along_channels * num_tiles_in_column;
    // const uint32_t tensor_face_thickness = tensor_face_block_size * generic_block_depth;
    // const uint32_t block_first_tile_id =
    //     tensor_face_thickness * column_block_i + row_block_i * channels_blocks_num + channels_slice_i;
    const uint32_t tile_id =
        row_tile_stride * tensor_face_tile_size + row_chunk_i * num_tiles_along_channels + channels_slice_i;
    // DPRINT << "TENSOR_FACE_BLOCK_SIZE: " << tensor_face_block_size << ", BLOCK_FIRST_TILE_ID: " <<
    // block_first_tile_id << ", TILE_ID: " << tile_id << ENDL(); DPRINT << "TILE_ID: " << tile_id << ENDL();
    return tile_id;
}

// all static data
// template <typename InputAccessorArgs, typename ZeroTileAccessorArgs, typename OutputAccessorArgs>
template <typename InputAccessorArgs, typename OutputAccessorArgs>
struct IntImgCTAs {
    const uint32_t start_cb;
    const uint32_t input_cb;
    const uint32_t acc_cb;
    const uint32_t cumsum_axis_3;
    const uint32_t before_adder_propagation_stage_cb;
    const uint32_t output_cb;
    const uint32_t to_bot_tile_cb;
    // const uint32_t from_top_tile_cb;
    const uint32_t axis_3_buffer_0_cb;
    const uint32_t axis_3_buffer_1_cb;  // dual channel communication with the writer kernel is comprehensive and
                                        // properly synchronizes writer and compute kernels.
    const uint32_t tile_height;
    const uint32_t tile_width;
    const uint32_t num_batches;
    const uint32_t input_depth;
    const uint32_t input_height;
    const uint32_t input_width;
    const uint32_t num_tiles_along_channels;
    const uint32_t num_tiles_along_height;
    const uint32_t top_semaphore;
    const uint32_t bot_semaphore;
    const InputAccessorArgs input_args;
    // const ZeroTileAccessorArgs zero_tile_args;
    const OutputAccessorArgs output_args;  // reused for reading upper block for propagation.
};

// all per-core runtime data
struct IntImgRTAs {
    const uint32_t input_base_addr;
    // const uint32_t zero_tile_base_addr;
    const uint32_t output_base_addr;
    const uint32_t starting_tile_along_channels;
    const uint32_t num_tiles_along_channels_per_core;
    const uint32_t starting_tile_along_height;
    const uint32_t num_tiles_along_height_per_core;
};

FORCE_INLINE constexpr auto get_ctas() {
    constexpr auto input_args = TensorAccessorArgs<19>();
    // constexpr auto zero_tile_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    // constexpr auto output_args = TensorAccessorArgs<zero_tile_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    // return IntImgCTAs<decltype(input_args), decltype(zero_tile_args), decltype(output_args)>{
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
        get_compile_time_arg_val(18),
        input_args,
        // zero_tile_args,
        output_args,
    };
}

FORCE_INLINE const auto get_rtas() {
    return IntImgRTAs{
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5),
        // get_arg_val<uint32_t>(6),
    };
}
