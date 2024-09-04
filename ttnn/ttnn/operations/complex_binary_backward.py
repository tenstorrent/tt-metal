# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def _golden_function_complex_add(grad_tensor, input_tensor_a, input_tensor_b, alpha, *args, **kwargs):
    input_tensor_a.retain_grad()

    pyt_y = torch.add(input_tensor_a, input_tensor_b, alpha=alpha)

    pyt_y.backward(gradient=grad_tensor)

    grad_in_real = torch.real(input_tensor_a.grad)
    grad_in_imag = torch.imag(input_tensor_a.grad)
    grad_other_real = torch.real(input_tensor_b.grad)
    grad_other_imag = torch.imag(input_tensor_b.grad)

    golden_tensor = [
        torch.cat((grad_in_real, grad_in_imag), dim=-1),
        torch.cat((grad_other_real, grad_other_imag), dim=-1),
    ]

    return golden_tensor


def _golden_function_complex_sub(grad_tensor, input_tensor_a, input_tensor_b, alpha, *args, **kwargs):
    input_tensor_a.retain_grad()

    pyt_y = torch.sub(input_tensor_a, input_tensor_b, alpha=alpha)

    pyt_y.backward(gradient=grad_tensor)

    grad_in_real = torch.real(input_tensor_a.grad)
    grad_in_imag = torch.imag(input_tensor_a.grad)
    grad_other_real = torch.real(input_tensor_b.grad)
    grad_other_imag = torch.imag(input_tensor_b.grad)

    golden_tensor = [
        torch.cat((grad_in_real, grad_in_imag), dim=-1),
        torch.cat((grad_other_real, grad_other_imag), dim=-1),
    ]

    return golden_tensor


def _golden_function_complex_mul(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    input_tensor_a.retain_grad()

    pyt_y = torch.mul(input_tensor_a, input_tensor_b)

    pyt_y.backward(gradient=grad_tensor)

    grad_in_real = torch.real(input_tensor_a.grad)
    grad_in_imag = torch.imag(input_tensor_a.grad)
    grad_other_real = torch.real(input_tensor_b.grad)
    grad_other_imag = torch.imag(input_tensor_b.grad)

    golden_tensor = [
        torch.cat((grad_in_real, grad_in_imag), dim=-1),
        torch.cat((grad_other_real, grad_other_imag), dim=-1),
    ]

    return golden_tensor


def _golden_function_complex_div(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    input_tensor_a.retain_grad()

    pyt_y = torch.div(input_tensor_a, input_tensor_b)

    pyt_y.backward(gradient=grad_tensor)

    grad_in_real = torch.real(input_tensor_a.grad)
    grad_in_imag = torch.imag(input_tensor_a.grad)
    grad_other_real = torch.real(input_tensor_b.grad)
    grad_other_imag = torch.imag(input_tensor_b.grad)

    golden_tensor = [
        torch.cat((grad_in_real, grad_in_imag), dim=-1),
        torch.cat((grad_other_real, grad_other_imag), dim=-1),
    ]

    return golden_tensor


__all__ = []
