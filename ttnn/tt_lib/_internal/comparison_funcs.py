# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch


def get_atol_rtol_pcc(golden, calculated):
    if golden.is_complex() and calculated.is_complex():
        golden = torch.view_as_real(golden.clone())
        calculated = torch.view_as_real(calculated.clone())

    if not (golden.is_floating_point() or calculated.is_floating_point()):
        golden = golden.to(torch.float)
        calculated = calculated.to(torch.float)

    # Calculate atol and rtol
    cal_atol = torch.max(torch.abs(golden - calculated)).item()
    cal_rtol = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()

    # Calculate PCC
    def get_pcc(golden, calculated):
        import ttnn

        return ttnn.pearson_correlation_coefficient(golden, calculated)

    cal_pcc = get_pcc(golden, calculated)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
        f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}",
    )


def comp_equal(golden, calculated):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    while len(golden.shape) < len(calculated.shape):
        golden = torch.unsqueeze(golden, 0)

    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    equal = torch.equal(golden, calculated)

    if not equal:
        output_str += ", Equal check failed"

    return equal, output_str


def comp_shape(golden, calculated):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)
    output_str = "compare shape"
    equal = golden.shape == calculated.shape
    return equal, output_str


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = torch.allclose(golden, calculated, rtol, atol, True)
    if not passing:
        output_str += ", Allclose check failed"
    return passing, output_str


def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)
    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = cal_pcc >= pcc
    if not passing:
        output_str += ", PCC check failed"
    return passing, output_str


def comp_and_get_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)
    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = cal_pcc >= pcc
    if not passing:
        output_str += f", PCC check failed (target: {pcc})"
    return passing, output_str, cal_pcc


def comp_pcc_list(golden, calculated, pcc=0.99):
    total_str = ""
    min_pcc = 1

    for i in range(len(golden)):
        if golden[i].dtype != calculated[i].dtype:
            calculated[i] = calculated[i].type(golden[i].dtype)
        _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden[i], calculated[i])

        total_str = f"{total_str}Tensor {i}: {output_str} "

        if cal_pcc < min_pcc:
            min_pcc = cal_pcc

    passing = min_pcc >= pcc
    if not passing:
        total_str += ", PCC check failed"
    return passing, total_str


def comp_equal_list(golden, calculated):
    total_str = ""
    min_pcc = 1

    equal = []

    for i in range(len(golden)):
        if golden[i].dtype != calculated[i].dtype:
            calculated[i] = calculated[i].type(golden[i].dtype)
        _, _, _, output_str = get_atol_rtol_pcc(golden[i], calculated[i])

        equal.append(torch.equal(golden[i], calculated[i]))

        total_str = f"{total_str}Tensor {i}: {output_str}"

    passing = all(equal)
    if not passing:
        total_str += ", PCC check failed"
    return passing, total_str


def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = True
    allclose_passing = torch.allclose(golden, calculated, rtol, atol, True)
    passing &= allclose_passing
    if not allclose_passing:
        output_str += ", Allclose check failed"
    pcc_passing = cal_pcc >= pcc
    passing &= pcc_passing
    if not pcc_passing:
        output_str += ", PCC check failed"
    return passing, output_str


def comp_using_plot(tname, input, golden, calculated):
    import matplotlib.pyplot as plt

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
        input = input.type(torch.float32)
    shape = "x".join(list(map(str, list(input.size()))))
    plot_name = "plot_" + tname + "_" + shape + ".png"
    input = input.flatten()
    golden = golden.flatten()
    calculated = calculated.flatten()
    plt.plot(input, golden, "+r", label="CPU (golden)")
    plt.plot(input, calculated, "-b", label="On device (calculated)")
    plt.legend(loc="upper center")
    plt.savefig(plot_name)
    plt.close()


def comp_topk_simmilarity(golden, calculated):
    golden_values, golden_indices = golden[0], golden[1]
    calculated_values, calculated_gather_values = calculated[0], calculated[1]

    values_passing, output_str = comp_pcc(golden_values, calculated_values)

    cosine = torch.nn.CosineSimilarity(dim=-1)
    ttnn_torch_cosine = torch.mean(cosine(golden_values, calculated_gather_values))

    indices_passing = ttnn_torch_cosine > 0.99

    output_str += f", Cosine simmilarity: {ttnn_torch_cosine.item()}"

    if not indices_passing:
        output_str += ", Cosine simmilarity check failed"

    return values_passing and indices_passing, output_str
