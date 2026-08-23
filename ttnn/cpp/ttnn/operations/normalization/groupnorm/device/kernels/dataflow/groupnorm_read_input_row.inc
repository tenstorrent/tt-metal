for (std::uint32_t nt = 0; nt < per_core_N; nt += max_read_batch) {
    const std::uint32_t read_batch = nt + max_read_batch <= per_core_N ? max_read_batch : per_core_N - nt;
    dfb_in0.reserve_back(read_batch);
    if constexpr (welford_fp32_alias) {
        dfb_in0_welford.reserve_back(read_batch);
    }
    for (std::uint32_t r = 0; r < read_batch; ++r) {
        noc.async_read(
            src_a,
            dfb_in0,
            src0_tile_bytes,
            {.page_id = start_id + index_b_offset + mt_offset + nt + r},
            {.offset_bytes = r * src0_tile_bytes});
    }
    noc.async_read_barrier();
    dfb_in0.push_back(read_batch);
    if constexpr (welford_fp32_alias) {
        // The alias shares SRAM with dfb_in0; only advance its CB state.
        dfb_in0_welford.push_back(read_batch);
    }
}
