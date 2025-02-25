#include <stdint.h>
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

struct PermuteReaderStruct {
    bool src_is_dram;
    uint32_t input_rm_page_size;

    constexpr PermuteReaderStruct(bool src_is_dram, uint32_t input_rm_page_size) :
        src_is_dram(src_is_dram), input_rm_page_size(input_rm_page_size) {}
};

// The struct you want to be constexpr-initializable:
template <uint32_t N>
struct PermuteWriterStruct {
    bool dst_is_dram;
    uint32_t rank;  // = N;
    uint32_t output_rm_page_size;
    uint32_t num_rows;
    uint32_t input_shape[N];
    uint32_t perm[N];
    uint32_t dest_strides[N];

    // This "inner" constructor does all the real initialization in its
    // member-initializer list. We expand the array references via I... .
    template <size_t... I>
    constexpr PermuteWriterStruct(
        bool dst_is_dram_,
        uint32_t out_page_size,
        uint32_t rows,
        const uint32_t (&inp)[N],
        const uint32_t (&p)[N],
        const uint32_t (&d)[N],
        tt::data_movement::common ::index_sequence<I...> /*unused*/
        ) :
        dst_is_dram(dst_is_dram_),
        rank(N),
        output_rm_page_size(out_page_size),
        num_rows(rows),
        input_shape{inp[I]...}  // expand each element
        ,
        perm{p[I]...},
        dest_strides{d[I]...} {
        // No loop here. Everything is done in the initializer list
    }

    // A "convenience" constructor that automatically creates the index_sequence.
    constexpr PermuteWriterStruct(
        bool dst_is_dram_,
        uint32_t out_page_size,
        uint32_t rows,
        const uint32_t (&inp)[N],
        const uint32_t (&p)[N],
        const uint32_t (&d)[N]) :
        PermuteWriterStruct(
            dst_is_dram_, out_page_size, rows, inp, p, d, tt::data_movement::common::make_index_sequence<N>{}) {}
};
