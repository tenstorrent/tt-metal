# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn


def _golden_function(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.real(input_tensor)

    pyt_y.backward(gradient=grad_tensor)

    grad_res_real = torch.real(input_tensor.grad)
    grad_res_imag = torch.imag(input_tensor.grad)

    golden_tensor = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    return golden_tensor


ttnn.attach_golden_function(ttnn.real_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.imag(input_tensor)

    pyt_y.backward(gradient=grad_tensor)

    grad_res_real = torch.real(input_tensor.grad)
    grad_res_imag = torch.imag(input_tensor.grad)

    golden_tensor = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    return golden_tensor


ttnn.attach_golden_function(ttnn.imag_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.angle(input_tensor)

    pyt_y.backward(gradient=grad_tensor)

    grad_res_real = torch.real(input_tensor.grad)
    grad_res_imag = torch.imag(input_tensor.grad)

    golden_tensor = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    return golden_tensor


ttnn.attach_golden_function(ttnn.angle_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.conj(input_tensor)

    pyt_y.backward(gradient=grad_tensor)

    grad_res_real = torch.real(input_tensor.grad)
    grad_res_imag = torch.imag(input_tensor.grad)

    golden_tensor = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    return golden_tensor


ttnn.attach_golden_function(ttnn.conj_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.polar(input_tensor.real, input_tensor.imag)

    pyt_y.backward(gradient=grad_tensor)

    grad_res_real = torch.real(input_tensor.grad)
    grad_res_imag = torch.imag(input_tensor.grad)

    golden_tensor = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    return golden_tensor


ttnn.attach_golden_function(ttnn.polar_bw, golden_function=_golden_function)

__all__ = []
