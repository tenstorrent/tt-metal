// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/model/matmul_model_device_params.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::matmul::model {
namespace {

// L1 compute model tables, measured on the device; values are exact (hex floats), with the shortest round-trip decimal
// alongside. Rows: h, w, fidelity, calls, Kref, F1, F2(kw=1), F2(kw=4), XP points (kw, value), XR points.
// clang-format off
// o_c, pack = p0 + p1 * h * w (cycles)
constexpr double kL1OverheadPerSubblock = 0x1.51344d1344d20p+4;  // 21.075268817204346
constexpr double kL1PackP0 = 0x1.37bd80e865ac9p+6;  // 77.9350620567376
constexpr double kL1PackP1 = 0x1.4f1a538489fc6p+4;  // 20.94392730496454
constexpr L1ShapeCosts kBlackholeL1Shapes[] = {
    {1, 1, Fidelity::kLoFi, 0x1.861ee6dabc715p+4, 0x1.54c92a1c7a653p+9, true, 0x1.c84cccccccce0p+5, true, 0x1.c599999999998p+6, false, 0.0, 4, {1, 2, 4, 8}, {0x1.74da506e2ab70p+6, 0x1.135c7e8ef1007p+7, 0x1.017b2b3ea84aep+7, 0x1.3c38849e16debp+7}, 4, {1, 2, 4, 8}, {0x1.e0c2664f0ca48p+7, 0x1.eec4efda1b7cap+7, 0x1.071501781c96bp+8, 0x1.f534291c748e4p+7}},
    {1, 1, Fidelity::kHiFi2, 0x1.0e6f7bdef7bdep+5, 0x1.60e22af122af6p+9, true, 0x1.b480000000000p+6, true, 0x1.baccccccccca0p+4, true, 0x1.a533333333338p+6, 4, {1, 2, 4, 8}, {0x1.6210b29bc45a8p+6, 0x1.fe728e45e215ap+6, 0x1.0ddb22cd0ec56p+7, 0x1.dcbdb44294758p+6}, 4, {1, 2, 4, 8}, {0x1.dd70ca990ca9ap+7, 0x1.d334eba14eba1p+7, 0x1.cce9fa7e9fa7ep+7, 0x1.74ba7e9fa7ea4p+7}},
    {1, 1, Fidelity::kHiFi4, 0x1.0a0034da034dcp+6, 0x1.6928d3680d36cp+9, false, 0.0, false, 0.0, true, 0x1.28fffffffffc0p+5, 4, {1, 2, 4, 8}, {0x1.4dfb6ee4701f2p+6, 0x1.41c806d7399e0p+6, 0x1.a8479d2333030p+6, 0x1.61e06354bf660p+6}, 4, {1, 2, 4, 8}, {0x1.99595bf095beep+7, 0x1.59dfa7e9fa7e8p+7, 0x1.c7654c8654c80p+6, 0x1.14fe12b7e12a8p+6}},
    {1, 2, Fidelity::kLoFi, 0x1.1cc34da034da1p+5, 0x1.6cd501a6d01acp+9, true, 0x1.3da2be2be2becp+6, false, 0.0, true, 0x1.18cccccccccd0p+6, 4, {1, 2, 4, 8}, {0x1.a66d01a6d01acp+6, 0x1.0178e09e8e09ap+7, 0x1.c8fb4069b406ap+6, 0x1.ff27d85c7d858p+6}, 4, {1, 2, 4, 8}, {0x1.0f5ab7e12b7e0p+8, 0x1.0dd8b4938b494p+8, 0x1.020e4791e4792p+8, 0x1.ef460e5060e52p+7}},
    {1, 2, Fidelity::kHiFi2, 0x1.1597891578915p+6, 0x1.7b01578915790p+9, true, 0x1.bb2f8af8af880p+5, false, 0.0, true, 0x1.9a666666666c0p+4, 4, {1, 2, 4, 8}, {0x1.4fd0b8fb0b8f8p+6, 0x1.0d6964bf964bdp+7, 0x1.d2f0842108418p+6, 0x1.9178c6318c640p+6}, 4, {1, 2, 4, 8}, {0x1.f2ba7e9fa7ea2p+7, 0x1.a55b86e1b86e3p+7, 0x1.f787fb987fb9cp+6, 0x1.35d03da903db0p+6}},
    {1, 2, Fidelity::kHiFi4, 0x1.09f4210842109p+7, 0x1.9cf5d50edd516p+9, true, 0x1.81541d41d41c0p+5, false, 0.0, true, 0x1.9a66666666660p+6, 4, {1, 2, 4, 8}, {0x1.124cccccccccdp+7, 0x1.27e578915788cp+7, 0x1.0a3069b4069b0p+7, 0x1.a2f2fe592fe38p+6}, 4, {1, 2, 4, 8}, {0x1.4bbeeeeeeeeebp+7, 0x1.4b62689a26894p+6, 0x1.41eb7e12b7e10p+6, 0x1.20240f6a40f58p+6}},
    {1, 3, Fidelity::kLoFi, 0x1.abf6ea876ea82p+5, 0x1.7fc43b7543b7ap+9, true, 0x1.05e6666666660p+6, true, 0x1.0333333333338p+6, false, 0.0, 4, {1, 2, 4, 8}, {0x1.3e79825c3b8d0p+7, 0x1.48eefaed93167p+7, 0x1.51b385a9dbc22p+7, 0x1.3abc9b226d1a0p+7}, 4, {1, 2, 4, 8}, {0x1.1b7dddddddddep+8, 0x1.071233c0233c0p+8, 0x1.05b7ac517ac55p+8, 0x1.d56ba14eba158p+7}},
    {1, 3, Fidelity::kHiFi2, 0x1.a448cb04e5bbbp+6, 0x1.891674a59ef0fp+9, true, 0x1.8af8af8af8b00p+6, false, 0.0, true, 0x1.bacccccccccc0p+5, 4, {1, 2, 4, 8}, {0x1.df3f4890e24c4p+6, 0x1.66e1a52c64ae9p+7, 0x1.35d8da277ef36p+7, 0x1.515410ea80488p+7}, 4, {1, 2, 4, 8}, {0x1.ea28440e57bb2p+7, 0x1.73d0ab58b1aa1p+7, 0x1.51c2f3dacb104p+6, 0x1.3a6c949400ee0p+6}},
    {1, 3, Fidelity::kHiFi4, 0x1.8efb0601f901dp+7, 0x1.cb625a905d418p+9, false, 0.0, false, 0.0, true, 0x1.0e00000000000p+4, 4, {1, 2, 4, 8}, {0x1.462f6a2f5168fp+7, 0x1.6f4dfdc6f2002p+7, 0x1.6c7e582966638p+7, 0x1.1078a687e8c24p+7}, 4, {1, 2, 4, 8}, {0x1.fed67a50d661ep+6, 0x1.956d3b19b12a4p+6, 0x1.39678978338a0p+6, 0x1.745c263538468p+6}},
    {1, 4, Fidelity::kLoFi, 0x1.1a56bf8926ff1p+6, 0x1.88c7b179c40bap+9, true, 0x1.558ccccccccc0p+6, true, 0x1.7f66666666670p+5, false, 0.0, 4, {1, 2, 4, 8}, {0x1.d2822c628c120p+7, 0x1.9e706637922c4p+7, 0x1.9219a6ae6b2cap+7, 0x1.6e9f5acf50620p+7}, 4, {1, 2, 4, 8}, {0x1.1efd71b2e8598p+8, 0x1.14d15b6a38336p+8, 0x1.fd925db1afce2p+7, 0x1.c3d1ab6c2e9d0p+7}},
    {1, 4, Fidelity::kHiFi2, 0x1.12b52b52b52b7p+7, 0x1.ba31b91b91b92p+9, true, 0x1.9d19999999980p+6, true, 0x1.7f66666666680p+5, true, 0x1.7499999999960p+5, 4, {1, 2, 4, 8}, {0x1.8b11fa6e03ffcp+7, 0x1.91299be81ba0ep+7, 0x1.64a5aba917b10p+7, 0x1.9ab764c4a968cp+7}, 4, {1, 2, 4, 8}, {0x1.cf644b0ae23a5p+7, 0x1.038f1fb82d0eep+7, 0x1.d949922585700p+6, 0x1.cea0378fdc138p+6}},
    {1, 4, Fidelity::kHiFi4, 0x1.07dedd50edd53p+8, 0x1.fbbdd50edd518p+9, true, 0x1.5433333333330p+6, false, 0.0, true, 0x1.b000000000000p+4, 4, {1, 2, 4, 8}, {0x1.74e304b8771a6p+7, 0x1.ce187d49cea30p+7, 0x1.79436e6c7db4cp+7, 0x1.b7cc83e50f0acp+7}, 4, {1, 2, 4, 8}, {0x1.c92aaaaaaaa9cp+6, 0x1.e1af3566f3550p+6, 0x1.d1eb7e12b7df8p+6, 0x1.0b253ae853abcp+7}},
    {1, 5, Fidelity::kLoFi, 0x1.5958915789157p+6, 0x1.a810abc48abcbp+9, true, 0x1.7f66666666660p+5, true, 0x1.64666666666a0p+4, true, 0x1.23999999999c0p+5, 4, {1, 2, 4, 8}, {0x1.ea512c3dfcfa8p+7, 0x1.ebcb49f89ed64p+7, 0x1.fd991f077c26ep+7, 0x1.f09b2f8b9d2f8p+7}, 4, {1, 2, 4, 8}, {0x1.459363a5b63a4p+8, 0x1.135a0c1ca0c1ep+8, 0x1.039dc370dc36fp+8, 0x1.ce5d9765d9768p+7}},
    {1, 5, Fidelity::kHiFi2, 0x1.55cdd50edd511p+7, 0x1.d0bea876ea87bp+9, true, 0x1.cb62be2be2bc0p+6, true, 0x1.e099999999990p+6, true, 0x1.7f66666666650p+6, 4, {1, 2, 4, 8}, {0x1.fa22d30e17674p+7, 0x1.050a7eff9d0adp+8, 0x1.041ca9f0bfba0p+8, 0x1.f19b993fa3ca4p+7}, 4, {1, 2, 4, 8}, {0x1.b5786e1b86e17p+7, 0x1.643765d9765d2p+7, 0x1.5022222222220p+7, 0x1.4bbe0119e0104p+7}},
    {1, 5, Fidelity::kHiFi4, 0x1.4ab46318c631cp+8, 0x1.106be592fe594p+10, true, 0x1.abf3333333310p+6, false, 0.0, true, 0x1.71e6666666680p+6, 4, {1, 2, 4, 8}, {0x1.bb07e1eb68544p+7, 0x1.0562c110212bep+8, 0x1.fd73f5bd2990cp+7, 0x1.fc77435d5e69cp+7}, 3, {1, 2, 4}, {0x1.51ea49c5a49c0p+7, 0x1.3d14b6c74b6b4p+7, 0x1.4d765d9765d7cp+7}},
    {1, 6, Fidelity::kLoFi, 0x1.a0eea876ea879p+6, 0x1.c2076ea876ea7p+9, true, 0x1.f06a0ea0ea0f0p+5, true, 0x1.61b3333333330p+6, true, 0x1.9d19999999990p+6, 4, {1, 2, 4, 8}, {0x1.13d7d18543e74p+8, 0x1.27cf5a9abc786p+8, 0x1.2d519ff8e0ce9p+8, 0x1.256fc44ec3138p+8}, 4, {1, 2, 4, 8}, {0x1.41daaaaaaaaaap+8, 0x1.1cc89a2689a24p+8, 0x1.e76f58a2f589ap+7, 0x1.c6cba14eba140p+7}},
    {1, 6, Fidelity::kHiFi2, 0x1.9b2c48abc48acp+7, 0x1.eeb7122af1228p+9, true, 0x1.cf6f8af8af8b0p+6, true, 0x1.6c80000000008p+7, true, 0x1.0e00000000000p+6, 4, {1, 2, 4, 8}, {0x1.0ba3bdb382aa0p+8, 0x1.0ea0cc90d3982p+8, 0x1.1b5aea4b7573ap+8, 0x1.17c8bf5a52c4ap+8}, 4, {1, 2, 4, 8}, {0x1.c526c74b6c750p+7, 0x1.9a874b6c74b6cp+7, 0x1.820853ae853acp+7, 0x1.8f23fdcc3fdccp+7}},
    {1, 6, Fidelity::kHiFi4, 0x1.8dc7b0b8fb0bbp+8, 0x1.2887ec2e3ec2fp+10, true, 0x1.7e0cccccccca0p+6, false, 0.0, true, 0x1.4eccccccccc80p+6, 4, {1, 2, 4, 8}, {0x1.1c0564839d177p+8, 0x1.274a80976ed84p+8, 0x1.24db1f2578c16p+8, 0x1.17e2c2a7f2f96p+8}, 4, {1, 2, 4, 8}, {0x1.8aca14eba14e6p+7, 0x1.97344d1344d10p+7, 0x1.6bdbf095bf084p+7, 0x1.5d64d1344d10cp+7}},
    {1, 7, Fidelity::kLoFi, 0x1.dd4c2e3ec2e42p+6, 0x1.d338b8fb0b8fbp+9, true, 0x1.f4457c57c57d0p+6, true, 0x1.46b3333333320p+6, true, 0x1.d3199999999a0p+6, 4, {1, 2, 4, 8}, {0x1.478ecadb05306p+8, 0x1.51d558e4ee111p+8, 0x1.4095a82bf3054p+8, 0x1.448fe0539687ap+8}, 4, {1, 2, 4, 8}, {0x1.46ace2d24e2d2p+8, 0x1.1489d7429d73fp+8, 0x1.1fa3c0233c020p+8, 0x1.09a12b7e12b7ap+8}},
    {1, 7, Fidelity::kHiFi2, 0x1.dea2108421085p+7, 0x1.0a037543b7545p+10, true, 0x1.5401d41d41d30p+6, true, 0x1.27a6666666660p+7, true, 0x1.1a26666666640p+7, 4, {1, 2, 4, 8}, {0x1.47b7348f0bcbcp+8, 0x1.4d6c92b361ae1p+8, 0x1.4b574efc0d724p+8, 0x1.514cc78d64fbap+8}, 4, {1, 2, 4, 8}, {0x1.00f87fb987fbap+8, 0x1.d4a888888888ap+7, 0x1.e4be0119e0120p+7, 0x1.dfc8f23c8f234p+7}},
    {1, 7, Fidelity::kHiFi4, 0x1.cf1957891578ap+8, 0x1.4119ef7bdef82p+10, true, 0x1.73a2be2be2ba0p+6, false, 0.0, true, 0x1.b2b33333333c0p+6, 4, {1, 2, 4, 8}, {0x1.24eee54806d78p+8, 0x1.4ebbf42557c52p+8, 0x1.3fc945132cd36p+8, 0x1.50771a220a23ep+8}, 4, {1, 2, 4, 8}, {0x1.c68060e5060e0p+7, 0x1.dd5a7e9fa7e9cp+7, 0x1.f6ceba14eba1cp+7, 0x1.cb3730ff7311cp+7}},
    {1, 8, Fidelity::kLoFi, 0x1.0d83827a3827cp+7, 0x1.e1c5ad6b5ad75p+9, true, 0x1.d538af8af8af0p+6, true, 0x1.28ffffffffff0p+6, true, 0x1.0e00000000000p+5, 4, {1, 2, 4, 8}, {0x1.717b3cc21e028p+8, 0x1.7673151e9b882p+8, 0x1.67692c3dfcfa6p+8, 0x1.74888daff311ap+8}, 4, {1, 2, 4, 8}, {0x1.6421605816056p+8, 0x1.316938b4938b2p+8, 0x1.2512830728302p+8, 0x1.31db7e12b7e0ep+8}},
    {1, 8, Fidelity::kHiFi2, 0x1.0cab827a3827bp+8, 0x1.17b1ad6b5ad6fp+10, true, 0x1.4e9b6db6db6d0p+6, false, 0.0, true, 0x1.9fccccccccd00p+5, 4, {1, 2, 4, 8}, {0x1.6717e1eb68553p+8, 0x1.8f92c5d79693cp+8, 0x1.840ef41659776p+8, 0x1.94da83c71272ep+8}, 4, {1, 2, 4, 8}, {0x1.1d546be7c6be5p+8, 0x1.21b2830728308p+8, 0x1.2661e4791e47ap+8, 0x1.28aa40f6a40eep+8}},
    {1, 8, Fidelity::kHiFi4, 0x1.06cc55e2455e5p+9, 0x1.57f091578915ep+10, true, 0x1.b65d41d41d420p+6, false, 0.0, true, 0x1.0899999999940p+6, 4, {1, 2, 4, 8}, {0x1.6f711f077c26cp+8, 0x1.808ba676249cep+8, 0x1.78e71bb9dbf0ep+8, 0x1.94d139747dcbep+8}, 4, {1, 2, 4, 8}, {0x1.116da903da900p+8, 0x1.10a4fd3f4fd32p+8, 0x1.1783a5b63a59ep+8, 0x1.341a903da900ep+8}},
    {2, 1, Fidelity::kLoFi, 0x1.195294a5294a2p+5, 0x1.6e385c7d85c81p+9, true, 0x1.1335075075080p+6, true, 0x1.6c80000000000p+6, true, 0x1.08999999999a0p+6, 4, {1, 2, 4, 8}, {0x1.b03ef7bdef7bcp+6, 0x1.2397a3827a385p+7, 0x1.8c5cb25fcb268p+6, 0x1.129bc48abc492p+7}, 4, {1, 2, 4, 8}, {0x1.0cbf3566f3566p+8, 0x1.1c8b4938b493ap+8, 0x1.07d370dc370ddp+8, 0x1.15108cf008cf2p+8}},
    {2, 1, Fidelity::kHiFi2, 0x1.09cfe592fe593p+6, 0x1.81b7964bf9651p+9, true, 0x1.123e2be2be2c0p+6, true, 0x1.0b4ccccccccd0p+7, false, 0.0, 4, {1, 2, 4, 8}, {0x1.86cb8fb0b8fa8p+6, 0x1.2c176ea876eabp+7, 0x1.0bd455e2455e0p+7, 0x1.6127bdef7bde8p+7}, 4, {1, 2, 4, 8}, {0x1.e257e9fa7e9f6p+7, 0x1.9b8990ca990cdp+7, 0x1.1f79ab379ab3ap+7, 0x1.ca0d59bcd59c0p+6}},
    {2, 1, Fidelity::kHiFi4, 0x1.0b8ec2e3ec2e5p+7, 0x1.95e14a5294a57p+9, true, 0x1.97e4924924940p+6, false, 0.0, true, 0x1.774ccccccccd0p+6, 4, {1, 2, 4, 8}, {0x1.bd7def7bdef72p+6, 0x1.d546d01a6d008p+6, 0x1.5abef7bdef7c0p+6, 0x1.b1c8e09e8e078p+6}, 4, {1, 2, 4, 8}, {0x1.6a8ab379ab377p+7, 0x1.336b145eb145cp+6, 0x1.0c966f3566f30p+6, 0x1.2179f1af9f1a8p+6}},
    {2, 2, Fidelity::kLoFi, 0x1.256bf964bf965p+6, 0x1.90fb18c6318ccp+9, true, 0x1.186a0ea0ea0e4p+7, true, 0x1.69cccccccccd0p+6, false, 0.0, 4, {1, 2, 4, 8}, {0x1.5e445c418c92ep+7, 0x1.920192c25ffb0p+7, 0x1.5a2f32f739feap+7, 0x1.be7da69421388p+7}, 4, {1, 2, 4, 8}, {0x1.594e89a2689a3p+8, 0x1.4eb9f1af9f1b0p+8, 0x1.5afa5b63a5b65p+8, 0x1.4e04c8654c864p+8}},
    {2, 2, Fidelity::kHiFi2, 0x1.155313a31acfbp+7, 0x1.b73093c589761p+9, false, 0.0, true, 0x1.0899999999980p+5, false, 0.0, 4, {1, 2, 4, 8}, {0x1.ace74550d18e0p+7, 0x1.c5fa98141d24ep+7, 0x1.c787a4011ab98p+7, 0x1.a4d4ef0e4913cp+7}, 4, {1, 2, 4, 8}, {0x1.15f997c3a4b18p+8, 0x1.8c134f1761c66p+7, 0x1.0eb9f49df8f40p+7, 0x1.01a0d944c0e84p+7}},
    {2, 2, Fidelity::kHiFi4, 0x1.0bef14ecdb0c2p+8, 0x1.ec0a2f316b4b9p+9, true, 0x1.417e2be2be2b0p+6, false, 0.0, true, 0x1.0e00000000000p+4, 4, {1, 2, 4, 8}, {0x1.a828fbe703128p+7, 0x1.bb246ba6e693cp+7, 0x1.6641b18d13fd4p+7, 0x1.972f708ca203cp+7}, 4, {1, 2, 4, 8}, {0x1.f5a9cc3af5ce8p+6, 0x1.afa0abbabcd18p+6, 0x1.a934d120b13d8p+6, 0x1.6fd04f1fcd498p+6}},
    {2, 4, Fidelity::kLoFi, 0x1.16b0e9da29119p+7, 0x1.df3e7ca331c4ap+9, false, 0.0, true, 0x1.db33333333380p+3, true, 0x1.69ccccccccd00p+5, 4, {1, 2, 4, 8}, {0x1.6577bc4558c12p+8, 0x1.67327a8b776b5p+8, 0x1.6e07f717b4c04p+8, 0x1.60062363629cep+8}, 4, {1, 2, 4, 8}, {0x1.9c4aaca81d90ap+8, 0x1.a10f0487d5d49p+8, 0x1.8067b447465ccp+8, 0x1.6185e092f439ap+8}},
    {2, 4, Fidelity::kHiFi2, 0x1.148413d1c13d3p+8, 0x1.10efd1c13d1c5p+10, true, 0x1.24c1d41d41d30p+6, true, 0x1.2bb3333333340p+6, true, -0x1.84ccccccccc00p+2, 4, {1, 2, 4, 8}, {0x1.6f7283c712733p+8, 0x1.71b4d65bb79c0p+8, 0x1.567fe1eb6854ep+8, 0x1.60e2c5d79692ep+8}, 4, {1, 2, 4, 8}, {0x1.2d7240f6a40f5p+8, 0x1.11aaf9f1af9f4p+8, 0x1.0c0f9f1af9f1ap+8, 0x1.08f28307282fep+8}},
    {2, 4, Fidelity::kHiFi4, 0x1.08142439bee6fp+9, 0x1.5a56e4a6c7f60p+10, false, 0.0, false, 0.0, true, 0x1.33ccccccccd00p+5, 4, {1, 2, 4, 8}, {0x1.641b1bf222af0p+8, 0x1.7d72d37ea4e0ep+8, 0x1.65e24297a944ep+8, 0x1.aee787301874ep+8}, 4, {1, 2, 4, 8}, {0x1.036472bb4de54p+8, 0x1.0ad8f7149ce42p+8, 0x1.1e2532fa6e14ep+8, 0x1.1377445faa11ep+8}},
    {3, 2, Fidelity::kLoFi, 0x1.9bf7bdef7bdefp+6, 0x1.c1bef7bdef7c4p+9, true, 0x1.636f8af8af890p+6, true, 0x1.9fccccccccca0p+5, false, 0.0, 4, {1, 2, 4, 8}, {0x1.30c258f3ec5e2p+8, 0x1.35c469780d66ap+8, 0x1.2adbbdb382aa7p+8, 0x1.27f0cc90d3984p+8}, 4, {1, 2, 4, 8}, {0x1.745ecbb2ecbb2p+8, 0x1.7a0da903da904p+8, 0x1.8411ca0c1ca0bp+8, 0x1.559d3f4fd3f4cp+8}},
    {3, 2, Fidelity::kHiFi2, 0x1.9ad8abc48abc5p+7, 0x1.f48a2af122afbp+9, true, 0x1.56e6666666640p+5, false, 0.0, false, 0.0, 4, {1, 2, 4, 8}, {0x1.30ed8c271f918p+8, 0x1.2aa13644da335p+8, 0x1.3a622419e9108p+8, 0x1.2eb0cc90d397ep+8}, 4, {1, 2, 4, 8}, {0x1.0529fee61fee8p+8, 0x1.ba3b5207b520ap+7, 0x1.b3f060e5060e8p+7, 0x1.ac14183941844p+7}},
    {3, 2, Fidelity::kHiFi4, 0x1.8c910f275d2a6p+8, 0x1.2abead6a9d474p+10, false, 0.0, false, 0.0, false, 0.0, 4, {1, 2, 4, 8}, {0x1.44626c7ba15ecp+8, 0x1.37e490877767ep+8, 0x1.2655a56bf045ap+8, 0x1.25a49c01aecfep+8}, 4, {1, 2, 4, 8}, {0x1.83c424dba9de0p+7, 0x1.94a2068cef894p+7, 0x1.aa3763891479cp+7, 0x1.87c21d815e59cp+7}},
    {4, 1, Fidelity::kLoFi, 0x1.1ddf964bf964dp+6, 0x1.8a698c6318c69p+9, false, 0.0, true, 0x1.e5ffffffffff0p+5, true, 0x1.bacccccccccc0p+4, 4, {1, 2, 4, 8}, {0x1.93fdc10122deap+7, 0x1.b741290e595f8p+7, 0x1.8dbb2c5bf9942p+7, 0x1.97d5995da0650p+7}, 4, {1, 2, 4, 8}, {0x1.2da1a2689a267p+8, 0x1.223cf008cf008p+8, 0x1.ee6716927168ep+7, 0x1.c7e7e9fa7e9f8p+7}},
    {4, 1, Fidelity::kHiFi2, 0x1.08843b7543b76p+7, 0x1.c369dbaa1dbb2p+9, true, 0x1.5bea0ea0ea100p+5, true, 0x1.f0cccccccccc0p+4, true, 0x1.03333333333c0p+4, 4, {1, 2, 4, 8}, {0x1.9aa950b1dbd9ep+7, 0x1.bf31e20964ef8p+7, 0x1.b9f637ebaa4d4p+7, 0x1.fdcbb07d01d64p+7}, 4, {1, 2, 4, 8}, {0x1.f3e86e1b86e1ap+7, 0x1.03b0ff730ff70p+7, 0x1.203555555554cp+7, 0x1.40f1344d13444p+7}},
    {4, 1, Fidelity::kHiFi4, 0x1.09068bb39a7f5p+8, 0x1.f9b394c20a396p+9, false, 0.0, false, 0.0, true, 0x1.1e33333333320p+6, 4, {1, 2, 4, 8}, {0x1.b4ba0e59842c2p+7, 0x1.ec935d58b5940p+7, 0x1.a1f92e8a4b96cp+7, 0x1.ac119dba4468cp+7}, 4, {1, 2, 4, 8}, {0x1.ad98bdecc4ce4p+6, 0x1.f21828b7f46b0p+6, 0x1.ca3d64b4ba098p+6, 0x1.de6e4314abad8p+6}},
    {4, 2, Fidelity::kLoFi, 0x1.0d9c97f2c97f5p+7, 0x1.e6c6592fe5933p+9, true, 0x1.bbf5075075060p+5, true, -0x1.4400000000000p+3, false, 0.0, 4, {1, 2, 4, 8}, {0x1.7c3b7ed2a2238p+8, 0x1.81d3993fa3ca9p+8, 0x1.88f7014cda4b0p+8, 0x1.948a9e341418ep+8}, 4, {1, 2, 4, 8}, {0x1.9504d59bcd59cp+8, 0x1.ad2cf008cf005p+8, 0x1.96f38b4938b48p+8, 0x1.970d8e96d8e92p+8}},
    {4, 2, Fidelity::kHiFi2, 0x1.13cf6171f6175p+8, 0x1.162ca5294a52dp+10, true, 0x1.a595f15f15f10p+6, true, 0x1.2d0ccccccccd0p+7, true, 0x1.8f99999999a00p+5, 4, {1, 2, 4, 8}, {0x1.722d9c8d43ff1p+8, 0x1.79de3b1b4de7cp+8, 0x1.6012ab6a94ec6p+8, 0x1.5a61f26f895bep+8}, 4, {1, 2, 4, 8}, {0x1.178a2689a2687p+8, 0x1.187ac517ac514p+8, 0x1.11959bcd59bc2p+8, 0x1.fac35f3e35f1cp+7}},
    {4, 2, Fidelity::kHiFi4, 0x1.0760bf964bf98p+9, 0x1.59dff964bf96ap+10, true, 0x1.8a33333333340p+5, false, 0.0, false, 0.0, 4, {1, 2, 4, 8}, {0x1.75b5186c3bbd2p+8, 0x1.7166cc72d6fdap+8, 0x1.7a57014cda4a6p+8, 0x1.bb776b00e0e4ep+8}, 4, {1, 2, 4, 8}, {0x1.0de808cf008cap+8, 0x1.1ad0233c02336p+8, 0x1.1dad24e2d24d6p+8, 0x1.313d8e96d8e8ep+8}},
    {8, 1, Fidelity::kLoFi, 0x1.105f964bf964cp+7, 0x1.e51318c6318cdp+9, true, 0x1.88141d41d41c0p+6, false, 0.0, false, 0.0, 4, {1, 2, 4, 8}, {0x1.5666cc72d6fdcp+8, 0x1.872a34800d7eap+8, 0x1.7eb76b00e0e66p+8, 0x1.686b719c21502p+8}, 4, {1, 2, 4, 8}, {0x1.6a20233c0233ap+8, 0x1.3449f1af9f1aep+8, 0x1.27ca5b63a5b66p+8, 0x1.2f31953219536p+8}},
    {8, 1, Fidelity::kHiFi2, 0x1.0ec5e2455e247p+8, 0x1.1660abc48abc9p+10, true, 0x1.28cea0ea0e9f0p+6, false, 0.0, false, 0.0, 4, {1, 2, 4, 8}, {0x1.7b43e886a8bebp+8, 0x1.75c46ca7b1008p+8, 0x1.6ed241b68e51ep+8, 0x1.727ab8a115bfep+8}, 4, {1, 2, 4, 8}, {0x1.15d3a5b63a5b3p+8, 0x1.1bcdc370dc36cp+8, 0x1.3088654c86542p+8, 0x1.2860dc370dc26p+8}},
    {8, 1, Fidelity::kHiFi4, 0x1.070cabc48abc6p+9, 0x1.598389157891ap+10, true, 0x1.2a283a83a8380p+6, false, 0.0, true, 0x1.2e66666666c00p+2, 4, {1, 2, 4, 8}, {0x1.70f6d9a957d0ep+8, 0x1.70b0b553758bap+8, 0x1.7b9e06414a996p+8, 0x1.af2bdb5027eaep+8}, 4, {1, 2, 4, 8}, {0x1.1f19ca0c1ca0ep+8, 0x1.0cf072830727ep+8, 0x1.198dc370dc366p+8, 0x1.1978654c8654ep+8}},
};
// clang-format on

// Blackhole p150 (bh-30 p150b SoC descriptor: Tensix columns 7 and 10 harvested; the 11x10 worker grid takes the
// first 11 of the 12 remaining columns, the last one goes to dispatch).
constexpr DeviceParams kBlackholeP150{
    .name = "blackhole_p150",
    .worker_grid_x = 11,
    .worker_grid_y = 10,
    .noc_size_x = 17,
    .noc_size_y = 12,
    .worker_cols = {1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15},
    .worker_rows = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    .num_channels = 8,
    .channel_endpoints =
        {{{{{0, 0}, {0, 1}, {0, 11}}},
          {{{0, 2}, {0, 10}, {0, 3}}},
          {{{0, 9}, {0, 4}, {0, 8}}},
          {{{0, 5}, {0, 7}, {0, 6}}},
          {{{9, 0}, {9, 1}, {9, 11}}},
          {{{9, 2}, {9, 10}, {9, 3}}},
          {{{9, 9}, {9, 4}, {9, 8}}},
          {{{9, 5}, {9, 7}, {9, 6}}}}},
    .num_banks = 8,
    .banks = {{{0, 2, 1}, {1, 0, 1}, {2, 0, 1}, {3, 0, 1}, {4, 2, 1}, {5, 2, 1}, {6, 2, 1}, {7, 2, 1}}},
    .clock_hz = 1.35e9,
    .noc_bytes_per_cycle = 64.0,
    .dram_GBs = 512.0,
    .c =
        {
            .dram_eff = 0x1.8c08a26120fe0p-1,    // 0.7735033744397661
            .noc_eff = 0x1.5ce43f6a4284dp-1,     // 0.6814288918697088
            .l1_frac = 0x1.53c194af54125p+0,     // 1.3271725585983145
            .lat_dram = 0x1.02793dccf56bcp+9,    // 516.9471985052437
            .lat_l1 = 0x1.90275d12aa3a9p+8,      // 400.1537639299009
            .launch_us = 0x1.d3da07a9a61c7p-1,   // 0.9137728113737972
            .tile_rt = 0x1.4e261753628a0p+9,     // 668.2975868445537
            .link_eff = 0x1.d527fc5233340p-1,    // 0.9163206911413013
            .bank_frac = 0x1.8d53f8e6dd5e9p-2,   // 0.38801564129911853
            .burst_KB = 0x1.1631d40f0463fp+9,    // 556.3892840167463
            .rl_init = 0x1.301a48bf23a1ep+7,     // 152.05133626279797
            .u2d = 0x1.715c18fe4fac3p+6,         // 92.33993909226687
            .lat_mcast = 0x1.27e9e2bcef3f5p+8,   // 295.91361599770545
            .ack_rx = 0x1.165b21d136d52p+3,      // 8.698624523755516
            .pack_Bpc = 0x1.33ea8247e27e5p+5,    // 38.489506303405314
            .call = 0x1.9d1d9d9198080p+1,        // 3.2274662934215144
            .lat_write = 0x1.bc9bb3e23a7fap+7,   // 222.30410677875608
            .sfpu_tile = 0x1.214ad4e5a2fd6p+10,  // 1157.1692442027756
            .issue = 0x1.7e2554603e4ddp+5,       // 47.76822734060418
            .mc_eff0 = 0x1.0000000000000p+0,     // 1.0
            .mc_eff_rx = 0x1.82e52088a1421p-11,  // 0.0007379437020779181
            .noc_eff_cc = 0x1.d28eed4aa981fp-2,  // 0.45562334793283016
            .reuse_link = 0x1.c59245017eeedp-1,  // 0.885881573135945
            .dram_w_eff = 0x1.58d00eab89facp-1,  // 0.6734623512647793
            .wf_reuse = 0x1.3b4f2ffac628ep+0,    // 1.231677054142128
            .pack_Bpc32 = 0x1.8f78118a85449p+5,  // 49.933627206985484
            .wf_in0 = 0x1.376f2051a617dp-1,      // 0.6082697009724566
            .wf_in1 = 0x1.2e0ed54d008d0p-1,      // 0.5899569183823363
            .lat_shard = 0x1.b09fc08e5df80p+8,   // 432.62403192323836
            .pad_elem = 0x1.7d7152053d893p-1,    // 0.7450051909825269
        },
    .l1 =
        {
            .overhead_per_subblock = kL1OverheadPerSubblock,
            .pack_p0 = kL1PackP0,
            .pack_p1 = kL1PackP1,
            .block_overhead = 50.0,
            .shapes = kBlackholeL1Shapes,
        },
};

// Blackhole Galaxy: all 12 unharvested Tensix columns are workers (on p150 the 12th goes to dispatch). Which columns
// each Galaxy chip harvested isn't known here, so the p150 pattern is assumed; configs up to 11 columns are placed
// exactly as on p150.
constexpr DeviceParams make_blackhole_galaxy() {
    DeviceParams d = kBlackholeP150;
    d.name = "blackhole_galaxy";
    d.worker_grid_x = 12;
    d.worker_cols[11] = 16;
    return d;
}

constexpr DeviceParams kBlackholeGalaxy = make_blackhole_galaxy();

}  // namespace

std::uint32_t fidelity_phases(Fidelity f) {
    switch (f) {
        case Fidelity::kLoFi: return 1;
        case Fidelity::kHiFi2: return 2;
        case Fidelity::kHiFi3: return 3;
        case Fidelity::kHiFi4: return 4;
    }
    TT_THROW("Unreachable");
}

const DeviceParams& device_params(Arch arch) {
    switch (arch) {
        case Arch::kBlackholeP150: return kBlackholeP150;
        case Arch::kBlackholeGalaxy: return kBlackholeGalaxy;
    }
    TT_THROW("matmul model: no device parameters for arch {}", static_cast<int>(arch));
}

std::optional<Arch> blackhole_arch_for_grid(std::uint32_t grid_x, std::uint32_t grid_y) {
    if (grid_y > kBlackholeP150.worker_grid_y) {
        return std::nullopt;
    }
    if (grid_x <= kBlackholeP150.worker_grid_x) {
        return Arch::kBlackholeP150;
    }
    if (grid_x <= kBlackholeGalaxy.worker_grid_x) {
        return Arch::kBlackholeGalaxy;
    }
    return std::nullopt;
}

}  // namespace ttnn::operations::matmul::model
