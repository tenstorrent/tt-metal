void kernel_main() {
  uint32_t num_tiles = get_arg_val<uint32_t>(0);
  uint32_t start_id = get_arg_val<uint32_t>(1);

  uint32_t end_id = start_id + num_tiles;
  for (uint32_t i = start_id; i < end_id; ++i) {
    cb_wait_front(tt::CBIndex::c_4, 1);
    cb_pop_front(tt::CBIndex::c_4, 1);

    cb_wait_front(tt::CBIndex::c_5, 1);
    cb_pop_front(tt::CBIndex::c_5, 1);

    cb_wait_front(tt::CBIndex::c_6, 1);
    cb_pop_front(tt::CBIndex::c_6, 1);
  }
}
