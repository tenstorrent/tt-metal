# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import io
import pathlib
import shutil
from typing import Optional, Union

from loguru import logger
import numpy as np
import torch

import ttnn

TILE_HEIGHT = 32
TILE_WIDTH = 32


def pad_tensor(tensor, height_multiple=TILE_HEIGHT, width_multiple=TILE_WIDTH):
    if len(tensor.shape) > 1:
        *original_batch_sizes, height, width = tensor.shape
        if height % height_multiple == 0 and width % width_multiple == 0:
            return tensor

        padded_height = int(np.ceil(height / height_multiple)) * height_multiple
        padded_width = int(np.ceil(width / width_multiple)) * width_multiple
        tensor = ttnn.core._reshape_to_4D(tensor)
        *batch_sizes, _, _ = tensor.shape
        tensor = ttnn.Tensor(tensor.value.pad(batch_sizes + [padded_height, padded_width], [0, 0, 0, 0], 0.0))
        tensor = ttnn.reshape(
            tensor,
            ttnn.Shape(original_batch_sizes + [height, width], original_batch_sizes + [padded_height, padded_width]),
        )
    else:
        (width,) = tensor.shape
        if width % width_multiple == 0:
            return tensor

        padded_width = int(np.ceil(width / width_multiple)) * width_multiple
        tensor = ttnn.core._reshape_to_4D(tensor)
        *batch_sizes, height, _ = tensor.shape
        tensor = ttnn.Tensor(tensor.value.pad(batch_sizes + [height, padded_width], [0, 0, 0, 0], 0.0))
        tensor = ttnn.reshape(tensor, ttnn.Shape([width], [padded_width]))
    return tensor


def preprocess_linear_weight(weight, *, dtype):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype)
    weight = pad_tensor(weight)
    weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
    return weight


def preprocess_linear_bias(bias, *, dtype):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype)
    bias = pad_tensor(bias)
    bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
    return bias


def preprocess_layernorm_parameter(parameter, *, dtype):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    parameter = pad_tensor(parameter)
    parameter = ttnn.to_layout(parameter, ttnn.TILE_LAYOUT)
    return parameter


def preprocess_embedding_weight(weight, *, dtype):
    weight = ttnn.from_torch(weight, dtype=dtype)
    return weight


class ParameterList(list):
    def __repr__(self):
        file = io.StringIO()
        repr_parameters(file, self)
        return file.getvalue()


class ParameterDict(dict):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        file = io.StringIO()
        repr_parameters(file, self)
        return file.getvalue()


def make_parameter_dict(dictionary: Union[dict, ParameterDict]) -> ParameterDict:
    if isinstance(dictionary, ParameterDict):
        return dictionary
    preprocessed_dictionary = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            value = make_parameter_dict(value)
        preprocessed_dictionary[key] = value
    return ParameterDict(preprocessed_dictionary)


def repr_parameters(file, parameters, indentation=""):
    next_indentation = indentation + "  "
    if isinstance(parameters, ParameterDict):
        if not parameters:
            file.write("{}")
            return

        file.write("{\n")
        for index, (key, value) in enumerate(parameters.items()):
            file.write(next_indentation)
            file.write(f"{key}: ")
            repr_parameters(file, value, next_indentation)
            file.write(",\n" if index < len(parameters) - 1 else "\n")
        file.write(indentation)
        file.write("}")
    elif isinstance(parameters, ParameterList):
        if not parameters:
            file.write("[]")
            return

        file.write("[\n")
        for index, element in enumerate(parameters):
            file.write(next_indentation)
            repr_parameters(file, element, next_indentation)
            file.write(",\n" if index < len(parameters) - 1 else "\n")
        file.write(indentation)
        file.write("]")
    else:
        file.write(repr(parameters.shape))


def default_preprocessor(model, name) -> ParameterDict:
    parameters = {}
    if isinstance(model, torch.nn.Linear):
        parameters[f"weight"] = preprocess_linear_weight(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            parameters[f"bias"] = preprocess_linear_bias(model.bias, dtype=ttnn.bfloat16)
    elif isinstance(model, torch.nn.LayerNorm):
        parameters[f"weight"] = preprocess_layernorm_parameter(model.weight, dtype=ttnn.bfloat16)
        parameters[f"bias"] = preprocess_layernorm_parameter(model.bias, dtype=ttnn.bfloat16)
    elif isinstance(model, torch.nn.Embedding):
        parameters[f"weight"] = preprocess_embedding_weight(model.weight, dtype=ttnn.bfloat16)
    return make_parameter_dict(parameters)


def _preprocess_model_parameters(
    model,
    *,
    convert_to_ttnn,
    custom_preprocessor=None,
    name,
) -> ParameterDict:
    if isinstance(model, torch.nn.modules.container.ModuleList):
        return ParameterList(
            [
                _preprocess_model_parameters(
                    child,
                    convert_to_ttnn=convert_to_ttnn,
                    custom_preprocessor=custom_preprocessor,
                    name=f"{name}.{index}" if name else f"{index}",
                )
                for index, child in enumerate(model.children())
            ]
        )

    if custom_preprocessor is not None:
        custom_preprocessor_parameters = custom_preprocessor(model, name)
        if custom_preprocessor_parameters:
            return make_parameter_dict(custom_preprocessor_parameters)

    if convert_to_ttnn(model, name):
        default_preprocessor_parameters = default_preprocessor(model, name)
        if default_preprocessor_parameters:
            return make_parameter_dict(default_preprocessor_parameters)

    named_children = list(model.named_children())
    if not named_children:
        if isinstance(model, torch.nn.Linear):
            parameters = {"weight": model.weight.T.contiguous()}
            if model.bias is not None:
                parameters["bias"] = model.bias
            return make_parameter_dict(parameters)
        elif isinstance(model, torch.nn.Conv2d):
            raise RuntimeError("Transpose conv weights?")
        return make_parameter_dict(dict(model.named_parameters()))

    parameters = {}
    for child_name, child in named_children:
        parameters[child_name] = _preprocess_model_parameters(
            child,
            convert_to_ttnn=convert_to_ttnn,
            custom_preprocessor=custom_preprocessor,
            name=f"{name}.{child_name}" if name else child_name,
        )

    parameters = make_parameter_dict(parameters)

    return parameters


def _load_parameters(model_cache_path: pathlib.Path) -> ParameterDict:
    output = {}
    for path in model_cache_path.glob("*"):
        if path.name == "version.txt":
            continue

        extension = path.suffix
        name = path.stem

        if path.is_dir():
            parameters = _load_parameters(path)
            if all(str(key).isdigit() for key in parameters):
                parameters = {int(key): value for key, value in parameters.items()}
                parameters = ParameterList([parameters[index] for index in sorted(parameters.keys())])
            output[name] = parameters
        elif extension == ".bin":
            output[name] = ttnn.load_tensor(path)
        elif extension == ".pt":
            output[name] = torch.load(path)
        else:
            raise RuntimeError("Unrecognized file extension!")
    return ParameterDict(output)


def _dump_parameters(model_cache_path: pathlib.Path, parameters: ParameterDict) -> None:
    model_cache_path.mkdir(parents=True)
    for name, value in parameters.items():
        if isinstance(value, ParameterDict):
            _dump_parameters(model_cache_path / name, value)
        elif isinstance(value, ParameterList):
            for index, element in enumerate(value):
                _dump_parameters(model_cache_path / name / str(index), element)
        elif isinstance(value, ttnn.Tensor):
            file_path = str(model_cache_path / name)
            file_name = file_path + ".bin"
            ttnn.dump_tensor(file_name, value)
        elif isinstance(value, (torch.Tensor, torch.nn.Parameter)):
            file_path = str(model_cache_path / name)
            file_name = file_path + ".pt"
            torch.save(value, file_name)
        else:
            raise RuntimeError(f"Unsupported type: {type(value)}")


def move_to_device(parameters, device):
    for name, value in list(parameters.items()):
        if isinstance(value, ParameterDict):
            parameters[name] = move_to_device(value, device)
        elif isinstance(value, ParameterList):
            for index, element in enumerate(value):
                parameters[name][index] = move_to_device(element, device)
        elif isinstance(value, ttnn.Tensor):
            parameters[name] = ttnn.to_device(value, device)
        else:
            parameters[name] = value
    return parameters


def git_hash():
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        raise RuntimeError("Couldn't get git hash!") from e


def preprocess_model_parameters(
    model_name=None,
    version=None,
    *,
    initialize_model,
    convert_to_ttnn=None,
    custom_preprocessor=None,
    device: Optional[ttnn.Device] = None,
    prefix: Optional[str] = None,
) -> ParameterDict:
    if convert_to_ttnn is None:

        def convert_to_ttnn(model, full_name):
            return True

    if model_name is None:
        model = initialize_model()
        parameters = _preprocess_model_parameters(
            model,
            convert_to_ttnn=convert_to_ttnn,
            custom_preprocessor=custom_preprocessor,
            name=prefix if prefix is not None else "",
        )

    else:
        model_cache_path = ttnn.MODEL_CACHE_PATH / model_name.replace("/", "_")
        version_file_path = model_cache_path / "version.txt"

        if version is None:
            version = git_hash()

        cache_exists = model_cache_path.exists()
        if cache_exists:
            if version_file_path.exists():
                with open(version_file_path) as f:
                    cached_version = f.readline()
            else:
                cached_version = None

            version_matches = version == cached_version
        else:
            version_matches = False

        if cache_exists and version_matches:
            logger.info(f'Loading model weights from cache: {model_cache_path}  (version "{version}")')
            parameters = _load_parameters(model_cache_path)
            logger.info(f'Loaded model weights from cache: {model_cache_path}  (version "{version}")')
        else:
            if initialize_model is None:
                raise RuntimeError(f'Cached weights for the model {model_name} (version "{version}") don\'t exist')

            logger.info(f'Saving model weights to cache: {model_cache_path} (version "{version}")')

            model = initialize_model()
            parameters = _preprocess_model_parameters(
                model,
                convert_to_ttnn=convert_to_ttnn,
                custom_preprocessor=custom_preprocessor,
                name=prefix if prefix is not None else "",
            )

            # TODO: use temporary directory
            if model_cache_path.exists():
                shutil.rmtree(model_cache_path)

            _dump_parameters(model_cache_path, parameters)

            with open(version_file_path, "w") as f:
                f.write(version)

            logger.info(f'Saved model weights to cache: {model_cache_path} (version "{version}")')

    if device is not None:
        logger.info(f"Moving model weights to device")
        parameters = move_to_device(parameters, device)
        logger.info(f"Moved model weights to device")

    return parameters
