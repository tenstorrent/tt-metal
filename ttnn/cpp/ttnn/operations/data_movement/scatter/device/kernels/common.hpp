
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint32_t ONE_PAGE = 1;

// supported reduction methods for scatter to be applied for source values coming from recurring indices
enum class ScatterReductionType : uint8_t { INVALID, ADD, MULTIPLY, AMIN, AMAX };

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

template <int32_t N>
FORCE_INLINE std::array<uint32_t, N> make_strides(const std::array<uint32_t, N>& dims) {
    std::array<uint32_t, N> s{};
    uint32_t acc = 1;
    for (int32_t i = N - 1; i >= 0; --i) {
        s[i] = acc;
        acc *= dims[i];
    }
    return s;
}

template <int32_t N>
FORCE_INLINE bool in_bounds(const std::array<uint32_t, N>& idx, const std::array<uint32_t, N>& dims) {
    for (int32_t i = 0; i < N; ++i) {
        if (idx[i] >= dims[i]) {
            return false;
        }
    }
    return true;
}

template <int32_t N>
FORCE_INLINE bool next_inplace(std::array<uint32_t, N>& idx, const std::array<uint32_t, N>& dims) {
    // last axis fastest
    for (int32_t i = N - 1; i >= 0; --i) {
        if (++idx[i] < dims[i]) {
            return true;  // normal increment without carry
        }
        idx[i] = 0;  // carry and continue
    }
    return false;  // overflow past most significant digit
}

template <int32_t N>
FORCE_INLINE uint32_t to_id(const std::array<uint32_t, N>& idx, const std::array<uint32_t, N>& strides) {
    uint32_t id = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(N); ++i) {
        id += idx[i] * strides[i];
    }
    return id;
}

// Convert linear id -> coordinates (row-major, last axis fastest).
template <int32_t N>
std::array<uint32_t, N> from_id(int32_t id, const std::array<uint32_t, N>& dims) {
    std::array<uint32_t, N> coord{};
    // Go left to right: for [d0, d1, ..., dN-1], last axis fastest
    for (int32_t i = N - 1; i >= 0; --i) {
        coord[i] = id % dims[i];
        id /= dims[i];
    }
    return coord;
}

template <uint32_t N>
std::array<uint32_t, N> make_shape_array_from_runtime_args(const uint32_t& C) {
    std::array<uint32_t, N> ret{};
    for (uint32_t i = C; i < C + N; ++i) {
        ret[i - C] = get_arg_val<uint32_t>(i);
    }

    return ret;
}

// this function is supposed to load either a whole stick or part of it
template <typename AddrGen>
FORCE_INLINE void load_to_cb(
    const uint32_t& cb,
    const AddrGen& addr_gtor,
    const uint32_t& offset_bytes,
    const uint32_t& chunk_size_bytes,
    const uint32_t& stick_id) {
    cb_reserve_back(cb, ONE_PAGE);
    const uint64_t source_noc_address = get_noc_addr(stick_id, addr_gtor);
    const uint32_t l1_write_address = get_write_ptr(cb);

    noc_async_read(source_noc_address + offset_bytes, l1_write_address, chunk_size_bytes);
    noc_async_read_barrier();

    cb_push_back(cb, ONE_PAGE);
}

// this function is supposed to write either a whole stick or part of it (76800 elements)
template <typename AddrGen>
FORCE_INLINE void write_to_output(
    const uint32_t& cb,
    const AddrGen& addr_gtor,
    const uint32_t& offset_bytes,
    const uint32_t& chunk_size_bytes,
    const uint32_t& stick_id) {
    cb_wait_front(cb, ONE_PAGE);
    const uint64_t destination_noc_address = get_noc_addr(stick_id, addr_gtor);
    const uint32_t l1_read_address = get_read_ptr(cb);

    noc_async_write(l1_read_address, destination_noc_address + offset_bytes, chunk_size_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb, ONE_PAGE);
}
