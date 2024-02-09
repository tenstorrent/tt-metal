# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import io
import pathlib
import shutil
import pickle
from typing import Optional, Union, Callable

from loguru import logger
import torch
import torchtrail

import ttnn
from ttnn.tracer import trace


def preprocess_linear_weight(weight, *, dtype):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return weight


def preprocess_linear_bias(bias, *, dtype):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return bias


def preprocess_layernorm_parameter(parameter, *, dtype):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
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
        if isinstance(value, dict) and not isinstance(value, Conv2dConfig):
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
    elif isinstance(parameters, (ttnn.Tensor, torch.Tensor)):
        file.write(repr(parameters.shape))
    elif isinstance(parameters, Conv2dConfig):
        if not parameters:
            file.write("{}")
            return

        file.write("{\n")
        for index, (key, value) in enumerate(parameters.items()):
            file.write(next_indentation)
            file.write(f"{key}: {value}")
            file.write(",\n" if index < len(parameters) - 1 else "\n")
        file.write(indentation)
        file.write("}")
    else:
        file.write(repr(parameters))


def default_preprocessor(model, name, module_config) -> ParameterDict:
    parameters = {}
    if isinstance(model, torch.nn.Linear):
        parameters[f"weight"] = preprocess_linear_weight(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            parameters[f"bias"] = preprocess_linear_bias(model.bias, dtype=ttnn.bfloat16)
    elif isinstance(model, torch.nn.Conv2d):
        if module_config is None:
            raise RuntimeError(f"torch.nn.Conv2d modules need run_model to be provided to preprocess_model_parameters")

        weight = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        bias = None
        if model.bias is not None:
            bias = ttnn.from_torch(torch.reshape(model.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

        conv = ttnn.Conv2D(
            **module_config.config,
            weight=weight,
            bias=bias,
            reader_patterns_cache=None,
            conv_blocking_and_parallelization_config_override=None,
            move_weights_to_device=False,
            device=None,
        )
        parameters["weight"] = ttnn.Tensor(conv.conv.weight)
        if bias is not None:
            parameters["bias"] = ttnn.Tensor(conv.conv.bias)

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
    custom_preprocessor,
    name,
    module_config,
) -> ParameterDict:
    if isinstance(model, torch.nn.modules.container.ModuleList):
        return ParameterList(
            [
                _preprocess_model_parameters(
                    child,
                    convert_to_ttnn=convert_to_ttnn,
                    custom_preprocessor=custom_preprocessor,
                    name=f"{name}.{index}" if name else f"{index}",
                    module_config=module_config.get(child_name, index) if module_config is not None else None,
                )
                for index, child in enumerate(model.children())
            ]
        )

    if custom_preprocessor is not None:
        custom_preprocessor_parameters = custom_preprocessor(model, name)
        if custom_preprocessor_parameters:
            return make_parameter_dict(custom_preprocessor_parameters)

    named_children = list(model.named_children())
    named_parameters = list((name, parameter) for name, parameter in model.named_parameters() if "." not in name)

    if convert_to_ttnn(model, name):
        default_preprocessor_parameters = default_preprocessor(model, name, module_config)
        if default_preprocessor_parameters:
            if len(default_preprocessor_parameters) != len(named_children) + len(named_parameters):
                raise RuntimeError(
                    f"Not all children or parameters were converted using default_preprocessor_parameters!"
                )
            return make_parameter_dict(default_preprocessor_parameters)

    if not named_children:
        if isinstance(model, torch.nn.Linear):
            parameters = {"weight": model.weight.T.contiguous()}
            if model.bias is not None:
                parameters["bias"] = model.bias
            return make_parameter_dict(parameters)
        return make_parameter_dict(dict(model.named_parameters()))

    parameters = {}
    for child_name, child in named_children:
        child_parameters = _preprocess_model_parameters(
            child,
            convert_to_ttnn=convert_to_ttnn,
            custom_preprocessor=custom_preprocessor,
            name=f"{name}.{child_name}" if name else child_name,
            module_config=module_config.get(child_name, None) if module_config is not None else None,
        )
        if child_parameters:
            parameters[child_name] = child_parameters

    for parameter_name, parameter in named_parameters:
        dtype = {
            torch.int16: ttnn.uint16,
            torch.int32: ttnn.uint32,
            torch.int64: ttnn.uint32,
            torch.bfloat16: ttnn.bfloat16,
            torch.float32: ttnn.bfloat16,
            torch.float64: ttnn.bfloat16,
        }[parameter.dtype]
        parameters[parameter_name] = ttnn.from_torch(parameter, dtype=dtype)

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
        elif extension == ".conv2d_pickle":
            with open(path, "rb") as file:
                output[name] = Conv2dConfig(pickle.load(file))
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
        elif isinstance(value, Conv2dConfig):
            file_path = str(model_cache_path / name)
            file_name = file_path + ".conv2d_pickle"
            with open(file_name, "wb") as file:
                pickle.dump(dict(value), file, protocol=pickle.HIGHEST_PROTOCOL)
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


class Conv2dConfig(dict):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def populate_module_configs(*, model, run_model):
    if run_model is None:
        return None

    with trace():
        output = run_model(model)

    if shutil.which("dot") is not None:
        file_name = ttnn.TMP_DIR / "model_graph.svg"
        logger.info(f"Dumping graph of the model to {file_name}")
        torchtrail.visualize(output, file_name=file_name)

    if isinstance(output, torch.Tensor):
        graph = output.graph
    elif isinstance(output, tuple):
        graph = torchtrail.multidigraph.compose_all(*(value.graph for value in output))
    else:
        raise RuntimeError(f"Unsupported output type: {type(output)}")

    def _populate_module_config(graph):
        module_config = {}
        for node in graph:
            attributes = graph.nodes[node]
            operation = attributes["operation"]
            if isinstance(operation, torchtrail.tracer.TorchModule):
                *_, module_name = operation.module.torchtrail_name.split(".")
                if isinstance(operation.module, torch.nn.Conv2d):
                    (input_node, _, edge_data), *_ = graph.in_edges(node, data=True)
                    input_shape = graph.nodes[input_node]["shapes"][edge_data["source_output_index"]]

                    activation = None
                    successors = list(graph.successors(node))
                    if len(successors) == 1:
                        output_node = successors[0]
                        successor_operation = graph.nodes[output_node]["operation"]
                        if isinstance(successor_operation, torchtrail.tracer.TorchModule):
                            if isinstance(successor_operation.module, torch.nn.ReLU):
                                activation = "relu"
                        elif isinstance(successor_operation, torchtrail.tracer.TorchFunction):
                            if successor_operation.function == torch.nn.functional.relu:
                                activation = "relu"

                    module_config[module_name] = ParameterDict(
                        {
                            "config": Conv2dConfig(
                                in_channels=operation.module.in_channels,
                                out_channels=operation.module.out_channels,
                                kernel_size=operation.module.kernel_size,
                                stride=operation.module.stride,
                                padding=operation.module.padding,
                                dilation=operation.module.dilation,
                                groups=operation.module.groups,
                                padding_mode=operation.module.padding_mode,
                                batch_size=input_shape[0],
                                input_height=input_shape[-2],
                                input_width=input_shape[-1],
                                math_fidelity=ttnn.MathFidelity.HiFi4,
                                dtype=ttnn.bfloat16,
                                weights_dtype=ttnn.bfloat16,
                                use_1d_systolic_array=True,
                                activation=activation,
                            )
                        }
                    )
                else:
                    submodule_config = _populate_module_config(operation.graph)
                    if submodule_config:
                        module_config[module_name] = submodule_config
        return ParameterDict(module_config)

    module_config = _populate_module_config(graph)
    return module_config[""]


def merge_dicts(dict_a: dict, dict_b: dict, path=[]):
    for key in dict_b:
        if key in dict_a:
            if isinstance(dict_a[key], dict) and isinstance(dict_b[key], dict):
                merge_dicts(dict_a[key], dict_b[key], path + [str(key)])
            elif dict_a[key] != dict_b[key]:
                raise Exception("Conflict at " + ".".join(path + [str(key)]))
        else:
            dict_a[key] = dict_b[key]
    return dict_a


def _initialize_model_and_preprocess_parameters(
    *, initialize_model, run_model, convert_to_ttnn, custom_preprocessor, prefix
):
    model = initialize_model()
    module_config = populate_module_configs(model=model, run_model=run_model)
    parameters = _preprocess_model_parameters(
        model,
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=custom_preprocessor,
        name=prefix if prefix is not None else "",
        module_config=module_config if module_config is not None else None,
    )
    if module_config is not None:
        parameters = merge_dicts(parameters, module_config)
    return make_parameter_dict(parameters)


def preprocess_model(
    *,
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    initialize_model: Optional[Callable[[], torch.nn.Module]] = None,
    convert_to_ttnn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
    custom_preprocessor: Optional[Callable[[torch.nn.Module, str], Union[dict, ParameterDict]]] = None,
    device: Optional[ttnn.Device] = None,
    prefix: str = "",
    run_model: Optional[Callable],
    reader_patterns_cache: Optional[dict],
) -> ParameterDict:
    """

    preprocess_model(initialize_model: Optional[Callable[[], torch.nn.Module]]=None, *, model_name: Optional[str]=None, version: Optional[str]=None, convert_to_ttnn: Optional[Callable[[torch.nn.Module, str], bool]]=None, custom_preprocessor: Optional[Callable[[torch.nn.Module, str], Union[dict, ParameterDict]]]=None, device: Optional[ttnn.Device] = None, prefix: Optional[str] = None, run_model: Optional[Callable], reader_patterns_cache: Optional[Dict]) -> ParameterDict

    Preprocess modules and parameters of a given model.

    Args:
        * :attr:`model_name`: Name of the model to be used by the cache. If not provided, the cache will be disabled.
        * :attr:`version`: Version of the model to be used by the cache. If not provided, the current git hash will be used. If the version doesn't match the cached version, the cache will be invalidated.
        * :attr:`initialize_model`: Function for initializing the model. It's not required if the model has already been cached and the cache is valid.
        * :attr:`convert_to_ttnn`: Function for determining whether to convert the parameters of a given module to ttnn.Tensor. If not provided, all modules will be converted.
        * :attr:`custom_preprocessor`: Function for preprocessing the parameters of a given module using user-specified logic. If not provided, the default preprocessor will be used.
        * :attr:`device`: Device on which to put ttnn.Tensor parameters
        * :attr:`prefix`: Prefix string to attach to the names of the modules/parameters. It's useful for making the names of submodules appear in the same way as in the original model.
        * :attr:`run_model`: Function for running the model. It's required for populating module_config. If run_model is provided, the graph of the model will be dumped to /tmp/ttnn/model_graph.svg
        * :attr:`reader_patterns_cache`: Cache for reader patterns. It's useful for avoiding recomputation of reader patterns when the same model is used multiple times.
    """

    if convert_to_ttnn is None:

        def convert_to_ttnn(model, full_name):
            return True

    if model_name is None:
        model = _initialize_model_and_preprocess_parameters(
            initialize_model=initialize_model,
            run_model=run_model,
            convert_to_ttnn=convert_to_ttnn,
            custom_preprocessor=custom_preprocessor,
            prefix=prefix,
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
            model = _load_parameters(model_cache_path)
            logger.info(f'Loaded model weights from cache: {model_cache_path}  (version "{version}")')
        else:
            if initialize_model is None:
                raise RuntimeError(f'Cached weights for the model {model_name} (version "{version}") don\'t exist')

            logger.info(f'Saving model weights to cache: {model_cache_path} (version "{version}")')

            model = _initialize_model_and_preprocess_parameters(
                initialize_model=initialize_model,
                run_model=run_model,
                convert_to_ttnn=convert_to_ttnn,
                custom_preprocessor=custom_preprocessor,
                prefix=prefix,
            )

            # TODO: use temporary directory
            if model_cache_path.exists():
                shutil.rmtree(model_cache_path)

            _dump_parameters(model_cache_path, model)

            with open(version_file_path, "w") as f:
                f.write(version)

            logger.info(f'Saved model weights to cache: {model_cache_path} (version "{version}")')

    if device is not None:
        logger.info(f"Moving model weights to device")
        model = move_to_device(model, device)
        logger.info(f"Moved model weights to device")

    def _convert_configs_to_modules(model):
        if isinstance(model, ParameterDict):
            if "config" in model:
                if isinstance(model.config, Conv2dConfig):
                    if isinstance(model.weight, torch.Tensor):
                        return model
                    return ttnn.Conv2D(
                        **model.config,
                        weight=model.weight,
                        bias=model.bias if "bias" in model else None,
                        reader_patterns_cache=reader_patterns_cache,
                        using_parameters_cache=True,
                        device=device,
                    )
                else:
                    raise RuntimeError(f"Unsupported config type: {type(model.config)}")
            else:
                return make_parameter_dict({name: _convert_configs_to_modules(value) for name, value in model.items()})
        elif isinstance(model, ParameterList):
            return ParameterList([_convert_configs_to_modules(value) for value in model])
        else:
            return model

    model = _convert_configs_to_modules(model)

    return model


def preprocess_model_parameters(
    *,
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    initialize_model: Optional[Callable[[], torch.nn.Module]] = None,
    convert_to_ttnn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
    custom_preprocessor: Optional[Callable[[torch.nn.Module, str], Union[dict, ParameterDict]]] = None,
    device: Optional[ttnn.Device] = None,
    prefix: str = "",
) -> ParameterDict:
    """

    preprocess_model_parameters(initialize_model: Optional[Callable[[], torch.nn.Module]]=None, *, model_name: Optional[str]=None, version: Optional[str]=None, convert_to_ttnn: Optional[Callable[[torch.nn.Module, str], bool]]=None, custom_preprocessor: Optional[Callable[[torch.nn.Module, str], Union[dict, ParameterDict]]]=None, device: Optional[ttnn.Device] = None, prefix: Optional[str] = None) -> ParameterDict

    Preprocess parameters of a given model.

    Args:
        * :attr:`model_name`: Name of the model to be used by the cache. If not provided, the cache will be disabled.
        * :attr:`version`: Version of the model to be used by the cache. If not provided, the current git hash will be used. If the version doesn't match the cached version, the cache will be invalidated.
        * :attr:`initialize_model`: Function for initializing the model. It's not required if the model has already been cached and the cache is valid.
        * :attr:`convert_to_ttnn`: Function for determining whether to convert the parameters of a given module to ttnn.Tensor. If not provided, all modules will be converted.
        * :attr:`custom_preprocessor`: Function for preprocessing the parameters of a given module using user-specified logic. If not provided, the default preprocessor will be used.
        * :attr:`device`: Device on which to put ttnn.Tensor parameters
        * :attr:`prefix`: Prefix string to attach to the names of the modules/parameters. It's useful for making the names of submodules appear in the same way as in the original model.
    """

    return preprocess_model(
        model_name=model_name,
        version=version,
        initialize_model=initialize_model,
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=custom_preprocessor,
        device=device,
        prefix=prefix,
        run_model=None,
        reader_patterns_cache=None,
    )


__all__ = []
