# usage: mk_split.py <src_dir> <dst_dir>: copy an op copy and add the fine kfz_* zones around the final pass (measurement only)
import re, shutil, sys

src, dst = sys.argv[1:3]
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
p = f"{dst}/kernels/conv2d_3x3_kw_fold.hpp"
s = open(p).read()


def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, 1)


rep(
    """                acc_ready.wait_front(1);
                const uint32_t n_first = g * gw;""",
    """                {
                    MaybeDeviceZoneScope("kfz_accw");
                    acc_ready.wait_front(1);
                }
                const uint32_t n_first = g * gw;""",
)
rep(
    """                UNPACK(({
                    const uint32_t acc_rd""",
    """                {
                MaybeDeviceZoneScope("kfz_rest");
                UNPACK(({
                    const uint32_t acc_rd""",
)
rep(
    """                if (g + 1 == NG) {
                    out.push_back(bm * N);
                }
                acc_ready.pop_front(1);""",
    """                }
                {
                MaybeDeviceZoneScope("kfz_uninit");
                if (g + 1 == NG) {
                    out.push_back(bm * N);
                }
                acc_ready.pop_front(1);""",
)
rep(
    """                if constexpr (relu) {
                    pack_relu_config(ReluConfig::none());
                }
            }
        }
""",
    """                if constexpr (relu) {
                    pack_relu_config(ReluConfig::none());
                }
                }
            }
        }
""",
)
rep(
    """                if (blk == 0 && g == 0) {
                    bias_rows.wait_front(N);
                }""",
    """                if (blk == 0 && g == 0) {
                    MaybeDeviceZoneScope("kfz_biasw");
                    bias_rows.wait_front(N);
                }""",
)
# init block: from reconfig_data_format(cb_acc, cb_bias_rows) through the reserve
i = s.index("                reconfig_data_format(cb_acc, cb_bias_rows);\n")
j = s.index("                if (g == 0) {\n                    out.reserve_back(bm * N);\n                }\n", i)
j_end = j + len("                if (g == 0) {\n                    out.reserve_back(bm * N);\n                }\n")
block = s[i:j_end]
rows = "                constexpr uint32_t ROWS_BYTES = 4 * 32 * 4;  // one column tile's 4 fp32 bias rows\n"
block_wo = block.replace(rows, "")
s = (
    s[:i]
    + (rows if rows in block else "")
    + '                {\n                MaybeDeviceZoneScope("kfz_init");\n'
    + block_wo
    + "                }\n"
    + s[j_end:]
)
rep(
    """                pack_untilize_uninit(cb_acc);
                pack_reconfig_l1_acc(0);""",
    """                {
                MaybeDeviceZoneScope("kfz_handoff");
                pack_untilize_uninit(cb_acc);
                pack_reconfig_l1_acc(0);""",
)
rep(
    """                acc_ready.reserve_back(1);
                acc_ready.push_back(1);
            }""",
    """                acc_ready.reserve_back(1);
                acc_ready.push_back(1);
                }
            }""",
)
rep(
    """        in.wait_front(slab_pages);
        in.pop_front(slab_pages);
    }
    if constexpr (!kblocked) {""",
    """        MaybeDeviceZoneScope("kfz_tail");
        in.wait_front(slab_pages);
        in.pop_front(slab_pages);
    }
    MaybeDeviceZoneScope("kfz_tail2");
    if constexpr (!kblocked) {""",
)
open(p, "w").write(s)
