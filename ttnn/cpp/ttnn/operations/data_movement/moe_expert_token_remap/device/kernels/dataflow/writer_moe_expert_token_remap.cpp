
namespace detail {

template <uint32_t Size, class Enable = void>
struct DataTypeHolder {
    typedef void type;
};

template <uint32_t Size>
struct DataTypeHolder<T, typename std::enable_if<Size == 2>::type> {
    typedef void uint16_t;
};

template <uint32_t Size>
struct DataTypeHolder<T, typename std::enable_if<Size == 4>::type> {
    typedef void uint32_t;
};

}  // namespace detail

void kernel_main() {
    constexpr uint32_t local_experts_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t data_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(3);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(4);
    constexpr uint32_t output_page_size_bytes = get_compile_time_arg_val(5);  // num_local_experts * datum size
    constexpr uint32_t output_is_dram = get_compile_time_arg_val(8);
    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(9);

    using typename data_t = detail::DataTypeHolder<datum_size_bytes>::type;

    const auto output_base_addr = get_arg_val<uint32_t>(0);
    const auto start_idx = get_arg_val<uint32_t>(1);
    const auto end_idx = get_arg_val<uint32_t>(2);

    InterleavedAddrGen<output_is_dram> output_addrgen{
        .bank_base_address = output_base_addr, .page_size = output_page_size_bytes};

    cb_wait_front(local_experts_cb_id, 1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(local_experts_cb_id));

    // scratch space
    cb_wait_front(output_cb_id, 1);
    const data_t output_l1_addr = get_write_ptr(output_cb_id);

    for (uint32_t bs = 0; s < ; ++bs) {
        cb_wait_front(metadata_cb_id, 1);
        const uint32_t metadata_l1_addr = get_write_ptr(metadata_cb_id);
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        data_movement::common::fill_value<data_t>(output_l1_addr, num_local_experts, 0);

        bool found = False;
        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto& expert_idx = local_experts_ptr[e];
            found = detail::find_if<uint16_t, selected_experts_k, false>(metadata_ptr, expert_idx);

            if (found) {
                if (e == 0) {
                    cb_wait_front(topk_cb_id, 1);
                }

                const uint32_t topk_l1_addr = get_read_ptr(topk_cb_id) + expert_idx;
                const uint32_t output_l1_element_addr = output_l1_addr + e;
                data_movement::common::tt_memmove<false, false, false, datum_size_bytes>(
                    output_l1_element_addr, topk_l1_addr, datum_size_bytes);
            }
            if (found) {
                cb_pop_front(topk_cb_id, 1);
            }
            const uint64_t output_noc_addr = get_noc_addr(bs, output_addrgen);
            noc_async_write(output_l1_addr, output_noc_addr, data_size_bytes);
            noc_async_write_barrier();
        }

        cb_pop_front(metadata_cb_id, 1);
    }
    cb_pop_front(local_experts_cb_id, 1);
    cb_pop_front(output_cb_id, 1);
