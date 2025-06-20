import sys
from dataclasses import dataclass
from glob import glob
from itertools import chain
from typing import Any, Callable, TypeVar

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import no_init_weights

def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_model_uninitialized(model_path: str):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    torch.set_default_dtype(current_dtype)

    model.eval()
    return model

def load_model_weights(model_path: str) -> dict[str, torch.Tensor]:
    safetensors_filepaths = sorted(glob(f"{model_path}/*.safetensors"))
    weights_dict = {}
    for safetensor_filepath in tqdm(safetensors_filepaths, total=len(safetensors_filepaths), desc="Loading weights"):
        weights_dict.update(load_file(safetensor_filepath))
    print("Loaded all weights" + " " * 50)
    return weights_dict


def apply_with_names(model_name: str, module: torch.nn.Module, func: Callable[[str, torch.Tensor], Any]):
    for tensor_name, tensor in chain(module.named_parameters(), module.named_buffers()):
        func(f"{model_name}{tensor_name}", tensor)


def load_weight_from_weights_dict(weights_dict: dict[str, torch.Tensor]) -> Callable[[str, torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def load_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
        print(f"Loading weight: {name}" + " " * 100, end="\r")
        if name not in weights_dict:
            return tensor
        loaded_weight = weights_dict[name].to(tensor.device).bfloat16()
        assert (
            loaded_weight.shape == tensor.shape
        ), f"Shape mismatch for {name}: {loaded_weight.shape} vs {tensor.shape}"
        tensor.copy_(loaded_weight)
        return tensor

    return load_weight


def unload_weight_from_weights_dict(
    weights_dict: dict[str, torch.Tensor],
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def unload_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name not in weights_dict:
            return tensor
        tensor.copy_(torch.empty_like(tensor))
        return tensor

    return unload_weight


def add_dynamic_weight_loading_hooks(
    module: torch.nn.Module,
    weights_dict: dict[str, torch.Tensor],
    lazy_modules: list[str] = ["DeepseekV3MLP"],
    model_name: str = "",
):
    is_lazy = any(module.__class__.__name__ == lazy_module for lazy_module in lazy_modules)
    if not is_lazy and next(module.children(), None) is not None:
        for child_name, child in module.named_children():
            add_dynamic_weight_loading_hooks(child, weights_dict, model_name=f"{model_name}{child_name}.")
        return
    elif not is_lazy:
        apply_with_names(model_name, module, load_weight_from_weights_dict(weights_dict))
        return
    module.register_forward_pre_hook(
        lambda module, args, kwargs: apply_with_names(model_name, module, load_weight_from_weights_dict(weights_dict)),
        with_kwargs=True,
    )
    module.register_forward_hook(
        lambda module, args, kwargs, output: apply_with_names(
            model_name, module, unload_weight_from_weights_dict(weights_dict)
        ),
        with_kwargs=True,
    )


# This saves the I/O of selected modules to a torch saved dict from the module path to the tuple of (input args, input kwargs, output).
def log_io_to_torch_hooks(
    filepath: str,
) -> tuple[
    Callable[[torch.nn.Module], Any],
    Callable[[torch.nn.Module, str, Any, dict[str, Any], Any], Any],
    Callable[[torch.nn.Module], Any],
]:
    per_model_log: dict[torch.nn.Module, dict[str, tuple[tuple, dict[str, Any], Any]]] = {}

    def module_pre_hook(model: torch.nn.Module):
        per_model_log[model] = {}

    def submodule_hook(model: torch.nn.Module, name: str, args: tuple, kwargs: dict[str, Any], output: Any):
        assert name not in per_model_log[model], (
            name,
            per_model_log[model],
            "Logging modules that are invoked multiple times per model run is not supported at the moment",
        )
        per_model_log[model][name] = (args, kwargs, output)

    def module_hook(model: torch.nn.Module):
        if model not in per_model_log:
            raise RuntimeError(f"module_pre_hook was not called before module_hook for model={model}")
        torch.save(per_model_log[model], filepath)

    return module_pre_hook, submodule_hook, module_hook


def add_model_io_logging_hooks(
    model: torch.nn.Module,
    model_pre_hook: Callable[[torch.nn.Module], Any],
    submodule_hook: Callable[[torch.nn.Module, str, Any, dict[str, Any], Any], Any],
    model_hook: Callable[[torch.nn.Module], Any],
    logged_modules: list[str] = [
        "DeepseekV3MLP",
        "DeepseekV3MoE",
        "DeepseekV3Attention",
        "DeepseekV3DecoderLayer",
        "DeepseekV3Model",
        "DeepseekV3RMSNorm",
    ],
):
    for name, submodule in model.named_modules():
        if any(submodule.__class__.__name__ == logged_module for logged_module in logged_modules):
            assert name != "lm_head"
            submodule.register_forward_hook(
                lambda _, args, kwargs, output, model=model, name=name: submodule_hook(
                    model, name, args, kwargs, output
                ),
                with_kwargs=True,
            )
    model.register_forward_pre_hook(lambda _, args, model=model: model_pre_hook(model))
    model.register_forward_hook(lambda _, args, output, model=model: model_hook(model))


def main(local_model_path: str, prompt: str, model_io_log_filepath: str | None = None):
    # Load the tokenizer
    print("Loading tokenizer")
    tokenizer = load_tokenizer(local_model_path)
    print("Tokenizer loaded successfully")

    # Load the model with uninitialized weights
    print("Loading uninitialized model")
    model = load_model_uninitialized(local_model_path)
    print("Model loaded successfully")

    # Load the model weights
    print("Loading model weights")
    weights_dict = load_model_weights(local_model_path)
    add_dynamic_weight_loading_hooks(model, weights_dict)
    print("Model weights loaded successfully")

    # Set up logging hooks
    if model_io_log_filepath:
        print("Setting up model I/O logging hooks")
        module_pre_hook, submodule_hook, module_hook = log_io_to_torch_hooks(model_io_log_filepath)
        add_model_io_logging_hooks(model, module_pre_hook, submodule_hook, module_hook)

    # Run the model
    model_inputs = tokenizer(prompt, return_tensors="pt")
    print("Running the model")
    with torch.no_grad():
        _ = model(model_inputs.input_ids)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise ValueError(
            "Usage: python deepseek-reference-outputs-gen.py <local_model_path> <prompt> [model_io_log_filepath]\n\
                         The log is a torch-saved dict from model paths to the tuple of (input args, input kwargs, output)."
        )
