// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../isunique_cbs.hpp"

#include <utility>

constexpr uint32_t ONE_PAGE = 1;

template <
    typename input_accessor_args_type,
    typename index_hint_accessor_args_type,
    typename first_occurrences_accessor_args_type,
    typename output_accessor_args_type>
struct IsUniqueCTAs {
    const bool input_tensor_is_dram;
    const bool index_hint_tensor_is_dram;
    const bool first_occurrences_tensor_is_dram;
    const bool output_tensor_is_dram;
    const uint32_t input_tensor_addr;
    const uint32_t index_hint_tensor_addr;
    const uint32_t first_occurrences_tensor_addr;
    const uint32_t output_tensor_addr;
    const uint32_t input_cb;
    const uint32_t index_hint_cb;
    const uint32_t first_occurrences_cb;
    const uint32_t output_cb;
    const uint32_t rearrange_phase_stick_size;
    const uint32_t operation_phase_stick_size;
    const uint32_t input_datum_size;
    const uint32_t index_hint_datum_size;
    const uint32_t first_occurrences_datum_size;
    const uint32_t output_datum_size;
    // const uint32_t input_stick_size_bytes;
    // const uint32_t index_hint_stick_size_bytes;
    // const uint32_t first_occurrences_stick_size_bytes;
    // const uint32_t output_stick_size_bytes;
    // const uint32_t input_stick_size_bytes_log2;
    // const uint32_t index_hint_stick_size_bytes_log2;
    // const uint32_t first_occurrences_stick_size_bytes_log2;
    // const uint32_t output_stick_size_bytes_log2;
    // const bool is_input_stick_size_bytes_pow2_min_32;   // necessary for InterleavedAddrGen
    // const bool is_index_hint_stick_size_bytes_pow2_min_32;   // necessary for InterleavedAddrGen
    // const bool is_first_occurrences_stick_size_bytes_pow2_min_32;  // necessary for InterleavedAddrGen
    // const bool is_output_stick_size_bytes_pow2_min_32;  // necessary for InterleavedAddrGen
    const bool first_occurrences_tensor_engaged;
    const uint32_t num_rows;
    const bool invert;
    const uint32_t dim;
    const uint32_t num_cores;
    const input_accessor_args_type input_accessor_args;
    const index_hint_accessor_args_type index_hint_accessor_args;
    // const first_occurrences_accessor_args_type first_occurrences_accessor_args;
    const output_accessor_args_type output_accessor_args;
};

FORCE_INLINE constexpr IsUniqueCTAs get_ctas() {
    constexpr auto input_args = TensorAccessorArgs<16>();
    constexpr auto index_hint_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    // constexpr auto fi_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    return {
        get_compile_time_arg_val(0) != 0,
        get_compile_time_arg_val(1) != 0,
        get_compile_time_arg_val(2) != 0,
        get_compile_time_arg_val(3) != 0,
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
        get_compile_time_arg_val(18) != 0,
        get_compile_time_arg_val(19),
        get_compile_time_arg_val(20) != 0,
        get_compile_time_arg_val(21),
        get_compile_time_arg_val(22),
        input_args,
        index_hint_args,
        output_args
        // get_compile_time_arg_val(23),
        // get_compile_time_arg_val(24) != 0,
        // get_compile_time_arg_val(25) != 0,
        // get_compile_time_arg_val(26) != 0,
        // get_compile_time_arg_val(27) != 0,
        // get_compile_time_arg_val(28) != 0,
        // get_compile_time_arg_val(29),
        // get_compile_time_arg_val(30) != 0,
        // get_compile_time_arg_val(31),
    };
}

using ChunkIDPair = std::pair<uint32_t, uint32_t>;

// class RoundRobinScheduler {
// public:
//   constexpr RoundRobinScheduler(uint32_t N_, uint32_t M_)
//     : N(N_), M(M_)
//   {
//     // pad to even
//     N0 = N + (N & 1);
//     // number of base matchings
//     L  = N0 - 1;
//     // real pairs per matching
//     P  = N >> 1;
//     // start at matching 0, sub-round 0
//     br = 0;
//     sr = 0;
//     // initial dummy‐edge index for odd N
//     k = 1;
//     dummyIdx = std::min(k, L - k);
//   }

//   // Call once *before* each barrier
//   constexpr void next_round() {
//     // advance to the next chunk in this matching
//     sr += 1;
//     // if we've now stepped past all P real pairs,
//     // wrap sr→0 and move to the next base‐matching
//     if (sr * M >= P) {
//       sr = 0;
//       br += 1;
//       // advance k∈[1..L] without / or %
//       k += 1;
//       if (k > L) k = 1;
//       dummyIdx = std::min(k, L - k);
//     }
//   }

//   // On each core (0 ≤ cid < M), *after* next_round():
//   // returns the unique (i,j) to compare, or (-1,-1).
//   constexpr ChunkIDPair get_pair(uint32_t cid) const {
//     // global index within this base‐matching
//     uint32_t idx = sr * M + cid;
//     if (idx >= P)
//       return {-1, -1};    // idle slot

//     // if N is odd, skip the dummy pair
//     uint32_t origIdx = idx + ((N & 1) && idx >= dummyIdx ? 1 : 0);

//     // two “slots” in the perfect matching on [0..N0-1]
//     uint32_t i = origIdx;
//     uint32_t j = (N0 - 1) - origIdx;

//     // closed‐form circle‐rotation of players[]
//     auto rotated = [&](uint32_t x) -> uint32_t {
//       if (x == 0) return 0;
//       uint32_t y = x - 1 - br;     // now in [-(L-1)..L-1]
//       if (y < 0) y += L;       // wrap into [0..L-1]
//       return y + 1;
//     };

//     return { rotated(i), rotated(j) };
//   }

// private:
//   uint32_t N, M;
//   uint32_t N0, L, P;   // derived constants
//   uint32_t br, sr;     // matching index & sub-round index
//   uint32_t k, dummyIdx;
// };

// template<uint32_t Rows, uint32_t Cores>
// struct EulerScheduler {
//     // static_assert(Rows > 1, "Rows must be > 1");
//     // static_assert(Cores > 0, "Cores must be > 0");

//     // Pad to odd degree: if Rows is even, use Rows0 = Rows+1; else Rows0 = Rows
//     static constexpr uint32_t Rows0 = (Rows % 2 == 0 ? Rows + 1 : Rows);
//     // Total real edges in complete graph K_Rows
//     static constexpr uint32_t TotalEdges = Rows * (Rows - 1) / 2;
//     // Circuit length = edges in K_Rows0 + 1
//     static constexpr uint32_t CircuitLen = (Rows0 * (Rows0 - 1)) / 2 + 1;

//     // 1) Build the Eulerian vertex circuit on K_{Rows0}
//     static constexpr std::array<uint32_t, CircuitLen> buildCircuit() {
//         bool adj[Rows0][Rows0] = {};
//         for (uint32_t u = 0; u < Rows0; ++u)
//             for (uint32_t v = u + 1; v < Rows0; ++v)
//                 adj[u][v] = adj[v][u] = true;

//         uint32_t stack_arr[CircuitLen], sp = 1;
//         stack_arr[0] = 0;
//         uint32_t out[CircuitLen], op = 0;
//         uint32_t iter[Rows0] = {};

//         while (sp) {
//             uint32_t u = stack_arr[sp - 1];
//             bool found = false;
//             for (uint32_t &v = iter[u]; v < Rows0; ++v) {
//                 if (v == u) continue;
//                 if (adj[u][v]) {
//                     adj[u][v] = adj[v][u] = false;
//                     stack_arr[sp++] = v;
//                     ++v;
//                     found = true;
//                     break;
//                 }
//             }
//             if (!found) {
//                 out[op++] = u;
//                 --sp;
//             }
//         }

//         std::array<uint32_t, CircuitLen> circuit{};
//         for (uint32_t i = 0; i < CircuitLen; ++i)
//             circuit[i] = out[i];
//         return circuit;
//     }

//     // 2) Extract and normalize unordered edges, skipping dummy Rows0-1
//     static constexpr std::array<ChunkIDPair, TotalEdges> buildEdges() {
//         auto V = buildCircuit();
//         std::array<ChunkIDPair, TotalEdges> edges{};
//         uint32_t idx = 0;
//         for (uint32_t i = 0; i + 1 < CircuitLen; ++i) {
//             uint32_t u = V[i], v = V[i+1];
//             if (u < Rows && v < Rows) {
//                 // normalize ordering
//                 if (u < v) edges[idx++] = {u, v};
//                 else        edges[idx++] = {v, u};
//             }
//         }
//         return edges; // idx must equal TotalEdges
//     }

//     // 3) Compute offsets & lengths for slicing into Cores chunks
//     static constexpr std::array<uint32_t, Cores> buildOffsets() {
//         std::array<uint32_t, Cores> offs{};
//         uint32_t base = TotalEdges / Cores;
//         uint32_t rem  = TotalEdges % Cores;
//         uint32_t cur  = 0;
//         for (uint32_t c = 0; c < Cores; ++c) {
//             offs[c] = cur;
//             cur += base + (c < rem ? 1 : 0);
//         }
//         return offs;
//     }

//     static constexpr std::array<uint32_t, Cores> buildLengths() {
//         std::array<uint32_t, Cores> lens{};
//         uint32_t base = TotalEdges / Cores;
//         uint32_t rem  = TotalEdges % Cores;
//         for (uint32_t c = 0; c < Cores; ++c)
//             lens[c] = base + (c < rem ? 1 : 0);
//         return lens;
//     }

//     // Compile-time data
//     static constexpr auto circuitEdges = buildEdges();
//     static constexpr auto offsets      = buildOffsets();
//     static constexpr auto lengths      = buildLengths();

//     // 4) O(1) lookup: get the k-th pair for core c, or (-1,-1)
//     static constexpr ChunkIDPair get_pair(uint32_t c, uint32_t k) {
//         if (c < 0 || c >= Cores || k < 0 || k >= lengths[c])
//             return {-1, -1};
//         return circuitEdges[offsets[c] + k];
//     }
//   };

namespace pair_scheduling {
// On startup (in ROM/Flash), you pre-compute nothing except two constants:
// constexpr int64_t N    = /* number of rows, e.g. 16384 */;
// constexpr int64_t M    = /* number of cores, e.g. 1024 */;
// constexpr int64_t E    = N*(N-1)/2;          // total pairs
// constexpr int64_t base = E / M;              // floor
// constexpr int64_t rem  = E % M;              // leftover

template <uint32_t num_rows>
constexpr inline uint32_t get_E() {
    return num_rows * (num_rows - 1) / 2;
}

template <uint32_t num_cores, uint32_t num_rows>
constexpr inline uint32_t get_base() {
    return E(num_rows) >> (static_cast<uint32_t>(std::log2(num_cores)));
}

template <uint32_t num_cores, uint32_t num_rows>
constexpr inline uint32_t get_rem() {
    return E(num_rows) & (num_cores - 1);
}

// Given a core c∈[0..M) and a local step k∈[0..len(c)), get your global index p:
template <uint32_t num_cores, uint32_t num_rows>
constexpr inline uint32_t offset_of(const uint32_t& core_id) {
    // first `rem` cores get base+1, the rest get base
    // p = c*(base+1)           for c<rem
    // p = rem*(base+1) + (c-rem)*base   otherwise
    //
    constexpr uint32_t base = get_base<num_cores, num_rows>();
    constexpr uint32_t rem = get_rem<num_cores, num_rows>();
    if (core_id < rem) {
        return core_id * (base + 1);
    } else {
        return rem * (base + 1) + (core_id - rem) * base;
    }
}

template <uint32_t num_cores, uint32_t num_rows>
constexpr inline uint32_t length_of(const uint32_t& core_id) {
    constexpr uint32_t base = get_base<num_cores, num_rows>();
    constexpr uint32_t rem = get_rem<num_cores, num_rows>();
    return core_id < rem ? base + 1 : base;
}

// Map lex-order index p→(u,v) with 0 ≤ u < v < N in O(1):
// We solve p = C(u) + (v-u-1), where C(u)=sum_{t< u}(N-1-t)=u*N - u*(u+1)/2

template <uint32_t num_cores, uint32_t num_rows>
constexpr inline std::pair<int, int> lex_pair(const uint32_t& p) {
    // 1) find u via inversion of C(u) ≤ p < C(u+1)
    //    Solve u*N - u*(u+1)/2 ≤ p  < ...
    //    quadratic:   u^2 - (2N-1)u + 2p ≤ 0
    //    root ≈ [ (2N-1) - sqrt((2N-1)^2 - 8p) ] / 2
    constexpr uint32_t E = get_E<num_rows>();
    constexpr uint32_t base = get_base<num_cores, num_rows>();
    constexpr uint32_t rem = get_rem<num_cores, num_rows>();
    double D = double(2 * N - 1);
    double S = std::sqrt(D * D - 8 * double(p));
    int64_t u = int64_t(std::floor((D - S) / 2.0));
    // clamp just in case
    if (u < 0) {
        u = 0;
    } else if (u >= N - 1) {
        u = N - 2;
    }

    // 2) compute how many edges come before u
    int64_t Cu = u * N - (u * (u + 1)) / 2;
    // 3) then p - Cu = (v - u - 1)  ⇒  v = (p - Cu) + u + 1
    int64_t v = (p - Cu) + u + 1;
    return {int(u), int(v)};
}

// Finally, at the top of round `k` on core `c`:
std::pair<int, int> get_pair(
    const uint32_t& num_cores, const uint32_t& num_rows, const uint32_t& c, const uint32_t& k) {
    int64_t len = length_of(c);
    if (k < 0 || k >= len) {
        return {-1, -1};
    }
    int64_t p = offset_of(c) + k;
    return lex_pair(num_cores, num_rows, p);
}
}  // namespace pair_scheduling
