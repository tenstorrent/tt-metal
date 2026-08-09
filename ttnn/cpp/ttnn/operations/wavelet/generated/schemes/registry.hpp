// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "bior1_1.hpp"
#include "bior1_3.hpp"
#include "bior1_5.hpp"
#include "bior2_2.hpp"
#include "bior2_4.hpp"
#include "bior2_6.hpp"
#include "bior2_8.hpp"
#include "bior3_1.hpp"
#include "bior3_3.hpp"
#include "bior3_5.hpp"
#include "bior3_7.hpp"
#include "bior3_9.hpp"
#include "bior4_4.hpp"
#include "bior5_5.hpp"
#include "bior6_8.hpp"
#include "coif1.hpp"
#include "coif10.hpp"
#include "coif11.hpp"
#include "coif12.hpp"
#include "coif13.hpp"
#include "coif14.hpp"
#include "coif15.hpp"
#include "coif16.hpp"
#include "coif17.hpp"
#include "coif2.hpp"
#include "coif3.hpp"
#include "coif4.hpp"
#include "coif5.hpp"
#include "coif6.hpp"
#include "coif7.hpp"
#include "coif8.hpp"
#include "coif9.hpp"
#include "db1.hpp"
#include "db10.hpp"
#include "db11.hpp"
#include "db12.hpp"
#include "db13.hpp"
#include "db14.hpp"
#include "db15.hpp"
#include "db16.hpp"
#include "db17.hpp"
#include "db18.hpp"
#include "db19.hpp"
#include "db2.hpp"
#include "db20.hpp"
#include "db21.hpp"
#include "db22.hpp"
#include "db23.hpp"
#include "db24.hpp"
#include "db25.hpp"
#include "db26.hpp"
#include "db27.hpp"
#include "db28.hpp"
#include "db29.hpp"
#include "db3.hpp"
#include "db30.hpp"
#include "db31.hpp"
#include "db32.hpp"
#include "db33.hpp"
#include "db34.hpp"
#include "db35.hpp"
#include "db36.hpp"
#include "db37.hpp"
#include "db38.hpp"
#include "db4.hpp"
#include "db5.hpp"
#include "db6.hpp"
#include "db7.hpp"
#include "db8.hpp"
#include "db9.hpp"
#include "dmey.hpp"
#include "haar.hpp"
#include "rbio1_1.hpp"
#include "rbio1_3.hpp"
#include "rbio1_5.hpp"
#include "rbio2_2.hpp"
#include "rbio2_4.hpp"
#include "rbio2_6.hpp"
#include "rbio2_8.hpp"
#include "rbio3_1.hpp"
#include "rbio3_3.hpp"
#include "rbio3_5.hpp"
#include "rbio3_7.hpp"
#include "rbio3_9.hpp"
#include "rbio4_4.hpp"
#include "rbio5_5.hpp"
#include "rbio6_8.hpp"
#include "sym10.hpp"
#include "sym11.hpp"
#include "sym12.hpp"
#include "sym13.hpp"
#include "sym14.hpp"
#include "sym15.hpp"
#include "sym16.hpp"
#include "sym17.hpp"
#include "sym18.hpp"
#include "sym19.hpp"
#include "sym2.hpp"
#include "sym20.hpp"
#include "sym3.hpp"
#include "sym4.hpp"
#include "sym5.hpp"
#include "sym6.hpp"
#include "sym7.hpp"
#include "sym8.hpp"
#include "sym9.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::wavelet {

struct SchemeInfo {
    std::string_view name;
    uint32_t tap_size;
    int32_t delay_even;
    int32_t delay_odd;
    uint32_t num_steps;
};

enum class SchemeId : uint32_t {
    kbior1_1,
    kbior1_3,
    kbior1_5,
    kbior2_2,
    kbior2_4,
    kbior2_6,
    kbior2_8,
    kbior3_1,
    kbior3_3,
    kbior3_5,
    kbior3_7,
    kbior3_9,
    kbior4_4,
    kbior5_5,
    kbior6_8,
    kcoif1,
    kcoif10,
    kcoif11,
    kcoif12,
    kcoif13,
    kcoif14,
    kcoif15,
    kcoif16,
    kcoif17,
    kcoif2,
    kcoif3,
    kcoif4,
    kcoif5,
    kcoif6,
    kcoif7,
    kcoif8,
    kcoif9,
    kdb1,
    kdb10,
    kdb11,
    kdb12,
    kdb13,
    kdb14,
    kdb15,
    kdb16,
    kdb17,
    kdb18,
    kdb19,
    kdb2,
    kdb20,
    kdb21,
    kdb22,
    kdb23,
    kdb24,
    kdb25,
    kdb26,
    kdb27,
    kdb28,
    kdb29,
    kdb3,
    kdb30,
    kdb31,
    kdb32,
    kdb33,
    kdb34,
    kdb35,
    kdb36,
    kdb37,
    kdb38,
    kdb4,
    kdb5,
    kdb6,
    kdb7,
    kdb8,
    kdb9,
    kdmey,
    khaar,
    krbio1_1,
    krbio1_3,
    krbio1_5,
    krbio2_2,
    krbio2_4,
    krbio2_6,
    krbio2_8,
    krbio3_1,
    krbio3_3,
    krbio3_5,
    krbio3_7,
    krbio3_9,
    krbio4_4,
    krbio5_5,
    krbio6_8,
    ksym10,
    ksym11,
    ksym12,
    ksym13,
    ksym14,
    ksym15,
    ksym16,
    ksym17,
    ksym18,
    ksym19,
    ksym2,
    ksym20,
    ksym3,
    ksym4,
    ksym5,
    ksym6,
    ksym7,
    ksym8,
    ksym9,
    kUnknown,
};

inline constexpr std::array<SchemeInfo, 106> kSchemeInfos = {
    SchemeInfo{"bior1.1", 2U, 0, 1, 5U},    SchemeInfo{"bior1.3", 6U, 1, 2, 4U},
    SchemeInfo{"bior1.5", 10U, 2, 3, 4U},   SchemeInfo{"bior2.2", 6U, 1, 2, 5U},
    SchemeInfo{"bior2.4", 10U, 2, 3, 5U},   SchemeInfo{"bior2.6", 14U, 3, 4, 5U},
    SchemeInfo{"bior2.8", 18U, 4, 5, 5U},   SchemeInfo{"bior3.1", 4U, 1, 1, 5U},
    SchemeInfo{"bior3.3", 8U, 2, 2, 6U},    SchemeInfo{"bior3.5", 12U, 3, 3, 6U},
    SchemeInfo{"bior3.7", 16U, 4, 4, 6U},   SchemeInfo{"bior3.9", 20U, 5, 5, 6U},
    SchemeInfo{"bior4.4", 10U, 2, 3, 7U},   SchemeInfo{"bior5.5", 12U, 3, 3, 7U},
    SchemeInfo{"bior6.8", 18U, 4, 5, 9U},   SchemeInfo{"coif1", 6U, 1, 2, 7U},
    SchemeInfo{"coif10", 60U, 15, 15, 33U}, SchemeInfo{"coif11", 66U, 16, 17, 37U},
    SchemeInfo{"coif12", 72U, 18, 18, 39U}, SchemeInfo{"coif13", 78U, 19, 20, 43U},
    SchemeInfo{"coif14", 84U, 21, 21, 45U}, SchemeInfo{"coif15", 90U, 22, 23, 49U},
    SchemeInfo{"coif16", 96U, 24, 24, 51U}, SchemeInfo{"coif17", 102U, 25, 26, 55U},
    SchemeInfo{"coif2", 12U, 3, 3, 9U},     SchemeInfo{"coif3", 18U, 4, 5, 13U},
    SchemeInfo{"coif4", 24U, 6, 6, 15U},    SchemeInfo{"coif5", 30U, 7, 8, 19U},
    SchemeInfo{"coif6", 36U, 9, 9, 21U},    SchemeInfo{"coif7", 42U, 10, 11, 25U},
    SchemeInfo{"coif8", 48U, 12, 12, 27U},  SchemeInfo{"coif9", 54U, 13, 14, 31U},
    SchemeInfo{"db1", 2U, 0, 1, 5U},        SchemeInfo{"db10", 20U, 5, 5, 13U},
    SchemeInfo{"db11", 22U, 5, 6, 15U},     SchemeInfo{"db12", 24U, 6, 6, 15U},
    SchemeInfo{"db13", 26U, 6, 7, 17U},     SchemeInfo{"db14", 28U, 7, 7, 17U},
    SchemeInfo{"db15", 30U, 7, 8, 19U},     SchemeInfo{"db16", 32U, 8, 8, 19U},
    SchemeInfo{"db17", 34U, 8, 9, 21U},     SchemeInfo{"db18", 36U, 9, 9, 21U},
    SchemeInfo{"db19", 38U, 9, 10, 23U},    SchemeInfo{"db2", 4U, 1, 1, 5U},
    SchemeInfo{"db20", 40U, 10, 10, 23U},   SchemeInfo{"db21", 42U, 10, 11, 25U},
    SchemeInfo{"db22", 44U, 11, 11, 25U},   SchemeInfo{"db23", 46U, 11, 12, 27U},
    SchemeInfo{"db24", 48U, 12, 12, 27U},   SchemeInfo{"db25", 50U, 12, 13, 29U},
    SchemeInfo{"db26", 52U, 13, 13, 29U},   SchemeInfo{"db27", 54U, 13, 14, 31U},
    SchemeInfo{"db28", 56U, 14, 14, 31U},   SchemeInfo{"db29", 58U, 14, 15, 33U},
    SchemeInfo{"db3", 6U, 1, 2, 7U},        SchemeInfo{"db30", 60U, 15, 15, 33U},
    SchemeInfo{"db31", 62U, 15, 16, 35U},   SchemeInfo{"db32", 64U, 16, 16, 35U},
    SchemeInfo{"db33", 66U, 16, 17, 37U},   SchemeInfo{"db34", 68U, 17, 17, 37U},
    SchemeInfo{"db35", 70U, 17, 18, 39U},   SchemeInfo{"db36", 72U, 18, 18, 39U},
    SchemeInfo{"db37", 74U, 18, 19, 41U},   SchemeInfo{"db38", 76U, 19, 19, 41U},
    SchemeInfo{"db4", 8U, 2, 2, 7U},        SchemeInfo{"db5", 10U, 2, 3, 9U},
    SchemeInfo{"db6", 12U, 3, 3, 9U},       SchemeInfo{"db7", 14U, 3, 4, 11U},
    SchemeInfo{"db8", 16U, 4, 4, 11U},      SchemeInfo{"db9", 18U, 4, 5, 13U},
    SchemeInfo{"dmey", 62U, 15, 16, 33U},   SchemeInfo{"haar", 2U, 0, 1, 5U},
    SchemeInfo{"rbio1.1", 2U, 0, 1, 5U},    SchemeInfo{"rbio1.3", 6U, 1, 2, 5U},
    SchemeInfo{"rbio1.5", 10U, 2, 3, 5U},   SchemeInfo{"rbio2.2", 6U, 1, 2, 5U},
    SchemeInfo{"rbio2.4", 10U, 2, 3, 5U},   SchemeInfo{"rbio2.6", 14U, 3, 4, 5U},
    SchemeInfo{"rbio2.8", 18U, 4, 5, 5U},   SchemeInfo{"rbio3.1", 4U, 1, 1, 5U},
    SchemeInfo{"rbio3.3", 8U, 2, 2, 5U},    SchemeInfo{"rbio3.5", 12U, 3, 3, 5U},
    SchemeInfo{"rbio3.7", 16U, 4, 4, 5U},   SchemeInfo{"rbio3.9", 20U, 5, 5, 5U},
    SchemeInfo{"rbio4.4", 10U, 2, 3, 7U},   SchemeInfo{"rbio5.5", 12U, 3, 3, 7U},
    SchemeInfo{"rbio6.8", 18U, 4, 5, 9U},   SchemeInfo{"sym10", 20U, 5, 5, 13U},
    SchemeInfo{"sym11", 22U, 5, 6, 15U},    SchemeInfo{"sym12", 24U, 6, 6, 15U},
    SchemeInfo{"sym13", 26U, 6, 7, 17U},    SchemeInfo{"sym14", 28U, 7, 7, 17U},
    SchemeInfo{"sym15", 30U, 7, 8, 19U},    SchemeInfo{"sym16", 32U, 8, 8, 19U},
    SchemeInfo{"sym17", 34U, 8, 9, 21U},    SchemeInfo{"sym18", 36U, 9, 9, 21U},
    SchemeInfo{"sym19", 38U, 9, 10, 23U},   SchemeInfo{"sym2", 4U, 1, 1, 5U},
    SchemeInfo{"sym20", 40U, 10, 10, 23U},  SchemeInfo{"sym3", 6U, 1, 2, 7U},
    SchemeInfo{"sym4", 8U, 2, 2, 7U},       SchemeInfo{"sym5", 10U, 2, 3, 9U},
    SchemeInfo{"sym6", 12U, 3, 3, 9U},      SchemeInfo{"sym7", 14U, 3, 4, 11U},
    SchemeInfo{"sym8", 16U, 4, 4, 11U},     SchemeInfo{"sym9", 18U, 4, 5, 13U},
};

[[nodiscard]] inline std::span<const SchemeInfo> available_wavelets() noexcept { return kSchemeInfos; }

[[nodiscard]] inline SchemeId scheme_id(std::string_view name) noexcept {
    if (name == "bior1.1") {
        return SchemeId::kbior1_1;
    }
    if (name == "bior1.3") {
        return SchemeId::kbior1_3;
    }
    if (name == "bior1.5") {
        return SchemeId::kbior1_5;
    }
    if (name == "bior2.2") {
        return SchemeId::kbior2_2;
    }
    if (name == "bior2.4") {
        return SchemeId::kbior2_4;
    }
    if (name == "bior2.6") {
        return SchemeId::kbior2_6;
    }
    if (name == "bior2.8") {
        return SchemeId::kbior2_8;
    }
    if (name == "bior3.1") {
        return SchemeId::kbior3_1;
    }
    if (name == "bior3.3") {
        return SchemeId::kbior3_3;
    }
    if (name == "bior3.5") {
        return SchemeId::kbior3_5;
    }
    if (name == "bior3.7") {
        return SchemeId::kbior3_7;
    }
    if (name == "bior3.9") {
        return SchemeId::kbior3_9;
    }
    if (name == "bior4.4") {
        return SchemeId::kbior4_4;
    }
    if (name == "bior5.5") {
        return SchemeId::kbior5_5;
    }
    if (name == "bior6.8") {
        return SchemeId::kbior6_8;
    }
    if (name == "coif1") {
        return SchemeId::kcoif1;
    }
    if (name == "coif10") {
        return SchemeId::kcoif10;
    }
    if (name == "coif11") {
        return SchemeId::kcoif11;
    }
    if (name == "coif12") {
        return SchemeId::kcoif12;
    }
    if (name == "coif13") {
        return SchemeId::kcoif13;
    }
    if (name == "coif14") {
        return SchemeId::kcoif14;
    }
    if (name == "coif15") {
        return SchemeId::kcoif15;
    }
    if (name == "coif16") {
        return SchemeId::kcoif16;
    }
    if (name == "coif17") {
        return SchemeId::kcoif17;
    }
    if (name == "coif2") {
        return SchemeId::kcoif2;
    }
    if (name == "coif3") {
        return SchemeId::kcoif3;
    }
    if (name == "coif4") {
        return SchemeId::kcoif4;
    }
    if (name == "coif5") {
        return SchemeId::kcoif5;
    }
    if (name == "coif6") {
        return SchemeId::kcoif6;
    }
    if (name == "coif7") {
        return SchemeId::kcoif7;
    }
    if (name == "coif8") {
        return SchemeId::kcoif8;
    }
    if (name == "coif9") {
        return SchemeId::kcoif9;
    }
    if (name == "db1") {
        return SchemeId::kdb1;
    }
    if (name == "db10") {
        return SchemeId::kdb10;
    }
    if (name == "db11") {
        return SchemeId::kdb11;
    }
    if (name == "db12") {
        return SchemeId::kdb12;
    }
    if (name == "db13") {
        return SchemeId::kdb13;
    }
    if (name == "db14") {
        return SchemeId::kdb14;
    }
    if (name == "db15") {
        return SchemeId::kdb15;
    }
    if (name == "db16") {
        return SchemeId::kdb16;
    }
    if (name == "db17") {
        return SchemeId::kdb17;
    }
    if (name == "db18") {
        return SchemeId::kdb18;
    }
    if (name == "db19") {
        return SchemeId::kdb19;
    }
    if (name == "db2") {
        return SchemeId::kdb2;
    }
    if (name == "db20") {
        return SchemeId::kdb20;
    }
    if (name == "db21") {
        return SchemeId::kdb21;
    }
    if (name == "db22") {
        return SchemeId::kdb22;
    }
    if (name == "db23") {
        return SchemeId::kdb23;
    }
    if (name == "db24") {
        return SchemeId::kdb24;
    }
    if (name == "db25") {
        return SchemeId::kdb25;
    }
    if (name == "db26") {
        return SchemeId::kdb26;
    }
    if (name == "db27") {
        return SchemeId::kdb27;
    }
    if (name == "db28") {
        return SchemeId::kdb28;
    }
    if (name == "db29") {
        return SchemeId::kdb29;
    }
    if (name == "db3") {
        return SchemeId::kdb3;
    }
    if (name == "db30") {
        return SchemeId::kdb30;
    }
    if (name == "db31") {
        return SchemeId::kdb31;
    }
    if (name == "db32") {
        return SchemeId::kdb32;
    }
    if (name == "db33") {
        return SchemeId::kdb33;
    }
    if (name == "db34") {
        return SchemeId::kdb34;
    }
    if (name == "db35") {
        return SchemeId::kdb35;
    }
    if (name == "db36") {
        return SchemeId::kdb36;
    }
    if (name == "db37") {
        return SchemeId::kdb37;
    }
    if (name == "db38") {
        return SchemeId::kdb38;
    }
    if (name == "db4") {
        return SchemeId::kdb4;
    }
    if (name == "db5") {
        return SchemeId::kdb5;
    }
    if (name == "db6") {
        return SchemeId::kdb6;
    }
    if (name == "db7") {
        return SchemeId::kdb7;
    }
    if (name == "db8") {
        return SchemeId::kdb8;
    }
    if (name == "db9") {
        return SchemeId::kdb9;
    }
    if (name == "dmey") {
        return SchemeId::kdmey;
    }
    if (name == "haar") {
        return SchemeId::khaar;
    }
    if (name == "rbio1.1") {
        return SchemeId::krbio1_1;
    }
    if (name == "rbio1.3") {
        return SchemeId::krbio1_3;
    }
    if (name == "rbio1.5") {
        return SchemeId::krbio1_5;
    }
    if (name == "rbio2.2") {
        return SchemeId::krbio2_2;
    }
    if (name == "rbio2.4") {
        return SchemeId::krbio2_4;
    }
    if (name == "rbio2.6") {
        return SchemeId::krbio2_6;
    }
    if (name == "rbio2.8") {
        return SchemeId::krbio2_8;
    }
    if (name == "rbio3.1") {
        return SchemeId::krbio3_1;
    }
    if (name == "rbio3.3") {
        return SchemeId::krbio3_3;
    }
    if (name == "rbio3.5") {
        return SchemeId::krbio3_5;
    }
    if (name == "rbio3.7") {
        return SchemeId::krbio3_7;
    }
    if (name == "rbio3.9") {
        return SchemeId::krbio3_9;
    }
    if (name == "rbio4.4") {
        return SchemeId::krbio4_4;
    }
    if (name == "rbio5.5") {
        return SchemeId::krbio5_5;
    }
    if (name == "rbio6.8") {
        return SchemeId::krbio6_8;
    }
    if (name == "sym10") {
        return SchemeId::ksym10;
    }
    if (name == "sym11") {
        return SchemeId::ksym11;
    }
    if (name == "sym12") {
        return SchemeId::ksym12;
    }
    if (name == "sym13") {
        return SchemeId::ksym13;
    }
    if (name == "sym14") {
        return SchemeId::ksym14;
    }
    if (name == "sym15") {
        return SchemeId::ksym15;
    }
    if (name == "sym16") {
        return SchemeId::ksym16;
    }
    if (name == "sym17") {
        return SchemeId::ksym17;
    }
    if (name == "sym18") {
        return SchemeId::ksym18;
    }
    if (name == "sym19") {
        return SchemeId::ksym19;
    }
    if (name == "sym2") {
        return SchemeId::ksym2;
    }
    if (name == "sym20") {
        return SchemeId::ksym20;
    }
    if (name == "sym3") {
        return SchemeId::ksym3;
    }
    if (name == "sym4") {
        return SchemeId::ksym4;
    }
    if (name == "sym5") {
        return SchemeId::ksym5;
    }
    if (name == "sym6") {
        return SchemeId::ksym6;
    }
    if (name == "sym7") {
        return SchemeId::ksym7;
    }
    if (name == "sym8") {
        return SchemeId::ksym8;
    }
    if (name == "sym9") {
        return SchemeId::ksym9;
    }
    return SchemeId::kUnknown;
}

template <typename Fn>
decltype(auto) dispatch_scheme(const SchemeId id, Fn&& fn) {
    switch (id) {
        case SchemeId::kbior1_1: return fn.template operator()<schemes::bior1_1>();
        case SchemeId::kbior1_3: return fn.template operator()<schemes::bior1_3>();
        case SchemeId::kbior1_5: return fn.template operator()<schemes::bior1_5>();
        case SchemeId::kbior2_2: return fn.template operator()<schemes::bior2_2>();
        case SchemeId::kbior2_4: return fn.template operator()<schemes::bior2_4>();
        case SchemeId::kbior2_6: return fn.template operator()<schemes::bior2_6>();
        case SchemeId::kbior2_8: return fn.template operator()<schemes::bior2_8>();
        case SchemeId::kbior3_1: return fn.template operator()<schemes::bior3_1>();
        case SchemeId::kbior3_3: return fn.template operator()<schemes::bior3_3>();
        case SchemeId::kbior3_5: return fn.template operator()<schemes::bior3_5>();
        case SchemeId::kbior3_7: return fn.template operator()<schemes::bior3_7>();
        case SchemeId::kbior3_9: return fn.template operator()<schemes::bior3_9>();
        case SchemeId::kbior4_4: return fn.template operator()<schemes::bior4_4>();
        case SchemeId::kbior5_5: return fn.template operator()<schemes::bior5_5>();
        case SchemeId::kbior6_8: return fn.template operator()<schemes::bior6_8>();
        case SchemeId::kcoif1: return fn.template operator()<schemes::coif1>();
        case SchemeId::kcoif10: return fn.template operator()<schemes::coif10>();
        case SchemeId::kcoif11: return fn.template operator()<schemes::coif11>();
        case SchemeId::kcoif12: return fn.template operator()<schemes::coif12>();
        case SchemeId::kcoif13: return fn.template operator()<schemes::coif13>();
        case SchemeId::kcoif14: return fn.template operator()<schemes::coif14>();
        case SchemeId::kcoif15: return fn.template operator()<schemes::coif15>();
        case SchemeId::kcoif16: return fn.template operator()<schemes::coif16>();
        case SchemeId::kcoif17: return fn.template operator()<schemes::coif17>();
        case SchemeId::kcoif2: return fn.template operator()<schemes::coif2>();
        case SchemeId::kcoif3: return fn.template operator()<schemes::coif3>();
        case SchemeId::kcoif4: return fn.template operator()<schemes::coif4>();
        case SchemeId::kcoif5: return fn.template operator()<schemes::coif5>();
        case SchemeId::kcoif6: return fn.template operator()<schemes::coif6>();
        case SchemeId::kcoif7: return fn.template operator()<schemes::coif7>();
        case SchemeId::kcoif8: return fn.template operator()<schemes::coif8>();
        case SchemeId::kcoif9: return fn.template operator()<schemes::coif9>();
        case SchemeId::kdb1: return fn.template operator()<schemes::db1>();
        case SchemeId::kdb10: return fn.template operator()<schemes::db10>();
        case SchemeId::kdb11: return fn.template operator()<schemes::db11>();
        case SchemeId::kdb12: return fn.template operator()<schemes::db12>();
        case SchemeId::kdb13: return fn.template operator()<schemes::db13>();
        case SchemeId::kdb14: return fn.template operator()<schemes::db14>();
        case SchemeId::kdb15: return fn.template operator()<schemes::db15>();
        case SchemeId::kdb16: return fn.template operator()<schemes::db16>();
        case SchemeId::kdb17: return fn.template operator()<schemes::db17>();
        case SchemeId::kdb18: return fn.template operator()<schemes::db18>();
        case SchemeId::kdb19: return fn.template operator()<schemes::db19>();
        case SchemeId::kdb2: return fn.template operator()<schemes::db2>();
        case SchemeId::kdb20: return fn.template operator()<schemes::db20>();
        case SchemeId::kdb21: return fn.template operator()<schemes::db21>();
        case SchemeId::kdb22: return fn.template operator()<schemes::db22>();
        case SchemeId::kdb23: return fn.template operator()<schemes::db23>();
        case SchemeId::kdb24: return fn.template operator()<schemes::db24>();
        case SchemeId::kdb25: return fn.template operator()<schemes::db25>();
        case SchemeId::kdb26: return fn.template operator()<schemes::db26>();
        case SchemeId::kdb27: return fn.template operator()<schemes::db27>();
        case SchemeId::kdb28: return fn.template operator()<schemes::db28>();
        case SchemeId::kdb29: return fn.template operator()<schemes::db29>();
        case SchemeId::kdb3: return fn.template operator()<schemes::db3>();
        case SchemeId::kdb30: return fn.template operator()<schemes::db30>();
        case SchemeId::kdb31: return fn.template operator()<schemes::db31>();
        case SchemeId::kdb32: return fn.template operator()<schemes::db32>();
        case SchemeId::kdb33: return fn.template operator()<schemes::db33>();
        case SchemeId::kdb34: return fn.template operator()<schemes::db34>();
        case SchemeId::kdb35: return fn.template operator()<schemes::db35>();
        case SchemeId::kdb36: return fn.template operator()<schemes::db36>();
        case SchemeId::kdb37: return fn.template operator()<schemes::db37>();
        case SchemeId::kdb38: return fn.template operator()<schemes::db38>();
        case SchemeId::kdb4: return fn.template operator()<schemes::db4>();
        case SchemeId::kdb5: return fn.template operator()<schemes::db5>();
        case SchemeId::kdb6: return fn.template operator()<schemes::db6>();
        case SchemeId::kdb7: return fn.template operator()<schemes::db7>();
        case SchemeId::kdb8: return fn.template operator()<schemes::db8>();
        case SchemeId::kdb9: return fn.template operator()<schemes::db9>();
        case SchemeId::kdmey: return fn.template operator()<schemes::dmey>();
        case SchemeId::khaar: return fn.template operator()<schemes::haar>();
        case SchemeId::krbio1_1: return fn.template operator()<schemes::rbio1_1>();
        case SchemeId::krbio1_3: return fn.template operator()<schemes::rbio1_3>();
        case SchemeId::krbio1_5: return fn.template operator()<schemes::rbio1_5>();
        case SchemeId::krbio2_2: return fn.template operator()<schemes::rbio2_2>();
        case SchemeId::krbio2_4: return fn.template operator()<schemes::rbio2_4>();
        case SchemeId::krbio2_6: return fn.template operator()<schemes::rbio2_6>();
        case SchemeId::krbio2_8: return fn.template operator()<schemes::rbio2_8>();
        case SchemeId::krbio3_1: return fn.template operator()<schemes::rbio3_1>();
        case SchemeId::krbio3_3: return fn.template operator()<schemes::rbio3_3>();
        case SchemeId::krbio3_5: return fn.template operator()<schemes::rbio3_5>();
        case SchemeId::krbio3_7: return fn.template operator()<schemes::rbio3_7>();
        case SchemeId::krbio3_9: return fn.template operator()<schemes::rbio3_9>();
        case SchemeId::krbio4_4: return fn.template operator()<schemes::rbio4_4>();
        case SchemeId::krbio5_5: return fn.template operator()<schemes::rbio5_5>();
        case SchemeId::krbio6_8: return fn.template operator()<schemes::rbio6_8>();
        case SchemeId::ksym10: return fn.template operator()<schemes::sym10>();
        case SchemeId::ksym11: return fn.template operator()<schemes::sym11>();
        case SchemeId::ksym12: return fn.template operator()<schemes::sym12>();
        case SchemeId::ksym13: return fn.template operator()<schemes::sym13>();
        case SchemeId::ksym14: return fn.template operator()<schemes::sym14>();
        case SchemeId::ksym15: return fn.template operator()<schemes::sym15>();
        case SchemeId::ksym16: return fn.template operator()<schemes::sym16>();
        case SchemeId::ksym17: return fn.template operator()<schemes::sym17>();
        case SchemeId::ksym18: return fn.template operator()<schemes::sym18>();
        case SchemeId::ksym19: return fn.template operator()<schemes::sym19>();
        case SchemeId::ksym2: return fn.template operator()<schemes::sym2>();
        case SchemeId::ksym20: return fn.template operator()<schemes::sym20>();
        case SchemeId::ksym3: return fn.template operator()<schemes::sym3>();
        case SchemeId::ksym4: return fn.template operator()<schemes::sym4>();
        case SchemeId::ksym5: return fn.template operator()<schemes::sym5>();
        case SchemeId::ksym6: return fn.template operator()<schemes::sym6>();
        case SchemeId::ksym7: return fn.template operator()<schemes::sym7>();
        case SchemeId::ksym8: return fn.template operator()<schemes::sym8>();
        case SchemeId::ksym9: return fn.template operator()<schemes::sym9>();
        case SchemeId::kUnknown: break;
    }
    TT_THROW("Unsupported wavelet scheme id: {}", static_cast<uint32_t>(id));
    return fn.template operator()<schemes::bior1_1>();
}

template <typename Fn>
decltype(auto) dispatch_scheme(const std::string_view name, Fn&& fn) {
    return dispatch_scheme(scheme_id(name), std::forward<Fn>(fn));
}

}  // namespace ttnn::operations::wavelet
