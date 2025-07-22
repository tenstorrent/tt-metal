
#include "dprint_pages.h"

#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

inline void print_uint16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << *ptr << " ";
        }
        DPRINT << ENDL();
    }
}

namespace detail {

template <uint32_t Size, class Enable = void>
struct DataTypeHolder {
    typedef void type;
};

template <uint32_t Size>
struct DataTypeHolder<Size, typename std::enable_if<Size == 2>::type> {
    typedef uint16_t type;
};

template <uint32_t Size>
struct DataTypeHolder<Size, typename std::enable_if<Size == 4>::type> {
    typedef uint32_t type;
};

}  // namespace detail

void kernel_main() {
    constexpr uint32_t local_experts_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t data_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(4);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(5);
    constexpr uint32_t output_page_size_bytes = get_compile_time_arg_val(6);  // num_local_experts * datum size
    constexpr uint32_t output_is_dram = get_compile_time_arg_val(7);
    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(8);

    using data_addr_t = detail::DataTypeHolder<datum_size_bytes>::type;

    const auto output_base_addr = get_arg_val<uint32_t>(0);
    const auto start_idx = get_arg_val<uint32_t>(1);
    const auto end_idx = get_arg_val<uint32_t>(2);

    // DPRINT << "WRITER 0 "<<"\n";

    InterleavedAddrGen<output_is_dram> output_addrgen{
        .bank_base_address = output_base_addr, .page_size = output_page_size_bytes};

    // DPRINT << "WRITER 0.5 "<<"\n";

    // scratch space
    cb_reserve_back(output_cb_id, 1);
    const uint32_t output_l1_addr = get_write_ptr(output_cb_id);
    cb_push_back(output_cb_id, 1);

    cb_wait_front(local_experts_cb_id, 1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(local_experts_cb_id));

    // DPRINT << "WRITER 0.75 "<<"\n";

    for (uint32_t bs = start_idx; bs < end_idx; ++bs) {
        // DPRINT << "WRITER 1: "<<bs<<"\n";
        cb_wait_front(metadata_cb_id, 1);
        const uint32_t metadata_l1_addr = get_write_ptr(metadata_cb_id);
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        print_uint16_pages(metadata_l1_addr, selected_experts_k, 1);

        tt::data_movement::common::fill_with_val<data_addr_t>(output_l1_addr, num_local_experts, 0);

        bool found = false;
        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto& expert_idx = local_experts_ptr[e];
            if (ttnn::operations::ccl::common::find_if<uint16_t, selected_experts_k, false>(metadata_ptr, expert_idx)) {
                if (!found) {
                    // DPRINT << "WRITER 2 bs: "<<bs<<" e: "<<e<<"\n";
                    cb_wait_front(data_cb_id, 1);
                    const uint32_t data_l1_addr = get_read_ptr(data_cb_id);

                    tt::data_movement::common::print_bf16_pages(data_l1_addr, num_local_experts * 8, 1);

                    // DPRINT << "WRITER 2.5 bs: "<<bs<<" e: "<<e<<"\n";
                    found = true;
                }

                const uint32_t topk_l1_addr = get_read_ptr(data_cb_id) + expert_idx * datum_size_bytes;
                const uint32_t output_l1_element_addr = output_l1_addr + e * datum_size_bytes;
                tt::data_movement::common::tt_memmove<false, false, false, datum_size_bytes>(
                    output_l1_element_addr, topk_l1_addr, datum_size_bytes);

                DPRINT << "bs: " << bs << " e: " << e << " expert: " << expert_idx << "\n";
                tt::data_movement::common::print_bf16_pages(output_l1_addr, num_local_experts, 1);
            }
            const uint64_t output_noc_addr = get_noc_addr(bs, output_addrgen);
            noc_async_write(output_l1_addr, output_noc_addr, output_page_size_bytes);
            noc_async_write_barrier();
            // DPRINT << "WRITER 4 "<<"\n";
        }
        if (found) {
            cb_pop_front(data_cb_id, 1);
            found = false;
            // DPRINT << "WRITER 3 "<<"\n";
        }

        cb_pop_front(metadata_cb_id, 1);
        // DPRINT << "WRITER 5 "<<bs<<"\n";
    }
    cb_pop_front(local_experts_cb_id, 1);
}
