#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/adam_w.h"

ALWI void pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb) {
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(icb);
#endif
    pack_tile(ifrom_dst, icb);
}

ALWI void copy_tile_init_with_dt(uint32_t icb, uint32_t transpose = 0) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format_srca(icb);
#endif
    copy_tile_to_dst_init_short(icb, transpose);
}

namespace NAMESPACE {
void MAIN {
  uint32_t per_core_block_cnt = get_compile_time_arg_val(0);

  float lr = 0.2;
  float weight_decay = 0.3;
  float beta1 = 0.5;
  float beta2 = 0.555;
  float recip_bias_correction1 = 1 / (1 - 0.5); // assume step 1
  float recip_bias_correction2 = 1 / (1 - 0.555); // assume step 1
  float eps = 1e-8;

  init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_4);
  for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
    cb_wait_front(tt::CBIndex::c_0, 1);
    cb_wait_front(tt::CBIndex::c_1, 1);
    cb_wait_front(tt::CBIndex::c_2, 1);
    cb_wait_front(tt::CBIndex::c_3, 1);
    tile_regs_acquire();
    copy_tile_init_with_dt(tt::CBIndex::c_0);
    copy_tile(tt::CBIndex::c_0, 0, 0);
    copy_tile_init_with_dt(tt::CBIndex::c_1);
    copy_tile(tt::CBIndex::c_1, 0, 1);
    copy_tile_init_with_dt(tt::CBIndex::c_2);
    copy_tile(tt::CBIndex::c_2, 0, 2);
    copy_tile_init_with_dt(tt::CBIndex::c_3);
    copy_tile(tt::CBIndex::c_3, 0, 3);
    moreh::adam_w_init();
    moreh::adam_w(0, lr, weight_decay, beta1, beta2, recip_bias_correction1, recip_bias_correction2, eps);
    tile_regs_commit();
    cb_pop_front(tt::CBIndex::c_0, 1);
    cb_pop_front(tt::CBIndex::c_1, 1);
    cb_pop_front(tt::CBIndex::c_2, 1);
    cb_pop_front(tt::CBIndex::c_3, 1);

    tile_regs_wait();
    cb_reserve_back(tt::CBIndex::c_4, 1);
    cb_reserve_back(tt::CBIndex::c_5, 1);
    cb_reserve_back(tt::CBIndex::c_6, 1);
    pack_tile_with_dt(0, tt::CBIndex::c_4);
    pack_tile_with_dt(1, tt::CBIndex::c_5);
    pack_tile_with_dt(2, tt::CBIndex::c_6);
    cb_push_back(tt::CBIndex::c_4, 1);
    cb_push_back(tt::CBIndex::c_5, 1);
    cb_push_back(tt::CBIndex::c_6, 1);
    tile_regs_release();
  }
}
}  // namespace NAMESPACE
