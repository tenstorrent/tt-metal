import torch
import functools
import gzip
import json
import numpy as np


def compress_tensor(t, sample_limit=1000):
    t = t.detach().cpu().flatten().to(torch.float32)
    n = t.numel()
    num_quantiles = round(max(7, min(40, n ** (1 / 4))))  # at least 7 quantiles, at most 40 or n**(1/4)
    if n == 0:
        return {
            "type": "dynamic_quantiles_stats",
            "norm": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "quantiles": [0.0] * num_quantiles,
        }

    # If too large, sample subset
    max_samples = sample_limit
    u = t
    if n > max_samples:
        idx = torch.randperm(n)[:max_samples]
        u = t[idx]

    norm = u.norm().item()
    mean = u.mean().item()
    std = u.std().item()

    # Normalize to unit vector to preserve cosine similarity
    u = u / norm

    # Compute quantiles
    if n < num_quantiles:
        # interpolate between max and min if not enough elements
        q = [u.min().item() + (u.max().item() - u.min().item()) * i / (num_quantiles - 1) for i in range(num_quantiles)]
    else:
        q = torch.quantile(u, torch.linspace(0, 1, num_quantiles)).tolist()

    return {"type": "dynamic_quantiles_stats", "norm": norm, "mean": mean, "std": std, "quantiles": q}


def decompress_tensor(summary, shape, dtype):
    assert summary["type"] == "dynamic_quantiles_stats", f"Unsupported summary type: {summary['type']}"
    q = summary["quantiles"]
    num_quantiles = len(q)
    norm = summary["norm"]
    mean = summary["mean"]
    std = summary["std"]

    if len(shape) == 0 or norm == 0 or (len(shape) == 1 and shape[0] == 0):
        return torch.zeros(shape, dtype=dtype)

    # Linearly interpolate 7 quantiles to reconstruct direction
    t_lin = torch.linspace(0, num_quantiles - 1, steps=np.prod(shape)).to(torch.float32)
    q_vals = torch.tensor(q)
    idx_lower = t_lin.floor().long().clamp(0, num_quantiles - 1)
    idx_upper = t_lin.ceil().long().clamp(0, num_quantiles - 1)
    alpha = t_lin - idx_lower
    approx_unit = (1 - alpha) * q_vals[idx_lower] + alpha * q_vals[idx_upper]

    # Rescale to original norm
    approx = approx_unit * norm

    # Optionally adjust mean and std for better approximation
    approx_mean = approx.mean()
    approx_std = approx.std(unbiased=False)

    if approx_std > 0:
        approx = (approx - approx_mean) / approx_std * std + mean
    else:
        approx.fill_(mean)
    approx = approx.clamp(min=-1e6, max=1e6)
    return approx.reshape(shape).to(dtype)


def _tensor_info(obj):
    if isinstance(obj, torch.Tensor) and not obj.__class__.__name__ == "Trackable_Tensor" and obj.device.type != "meta":
        assert not torch.isnan(
            obj
        ).any(), f"NaN detected in tensor with shape {obj.shape} and dtype {obj.dtype}, please make sure the model params are valid. Please fill `download_original_model_state_dict` to load original model state_dict."
        compressed_summary = compress_tensor(obj)
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "summary": compressed_summary,
        }
    elif isinstance(obj, StateDictConstant):
        return {
            "shape": list(obj.value.shape),
            "dtype": str(obj.value.dtype),
            "summary": {"type": "state_dict_constant", "name": obj.name},
        }
    elif isinstance(obj, (list, tuple)):
        return [_tensor_info(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _tensor_info(v) for k, v in obj.items()}
    return str(type(obj))


def flatten_state_dict_constants(args):
    if isinstance(args, (list, tuple)):
        new_args = []
        for arg in args:
            if isinstance(arg, StateDictConstant):
                new_args.append(arg.value)
            else:
                new_args.append(flatten_state_dict_constants(arg))
        return type(args)(new_args)
    elif isinstance(args, dict):
        new_args = {}
        for k, v in args.items():
            if isinstance(v, StateDictConstant):
                new_args[k] = v.value
            else:
                new_args[k] = flatten_state_dict_constants(v)
        return new_args
    return args


def track_input_output(_tensor_io_log):
    def _track_input_output(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            input_info = _tensor_info(args)
            args = flatten_state_dict_constants(args)
            output = fn(*args, **kwargs)
            output_info = _tensor_info(output)
            _tensor_io_log.append(
                {
                    "function": fn.__name__,
                    "inputs": input_info,
                    "outputs": output_info,
                }
            )
            return output

        return wrapper

    return _track_input_output


def get_tensors_from_input_spec(input_specs, state_dict=None):
    if (
        isinstance(input_specs, (tuple, list))
        and len(input_specs) != 4
        and not (len(input_specs) == 4 and isinstance(input_specs[2], str))
    ):
        return [get_tensors_from_input_spec(spec, state_dict) for spec in input_specs]
    assert (
        isinstance(input_specs, (list, tuple)) and len(input_specs) == 4 and isinstance(input_specs[2], str)
    ), "input_specs must be a list of 4 elements where the 3rd element contains the spec type."
    shape, dtype_str, summary_type, summary = input_specs
    dtype = getattr(torch, dtype_str.split(".")[1])
    tensor = None
    if summary_type == "state_dict_constant":
        assert state_dict is not None, "state_dict must be provided for state_dict_constant"
        tensor = state_dict[summary]
        tensor = tensor.to(dtype)
    elif summary_type == "dynamic_quantiles_stats":
        assert len(summary) == 2, "Expected 2 values in summary"
        summary = {
            "quantiles": summary[0],
            "mean": summary[1][0],
            "std": summary[1][1],
            "norm": summary[1][2],
            "type": summary_type,
        }
        tensor = decompress_tensor(summary, shape, dtype)
    else:
        raise ValueError(f"Unknown summary type: {summary_type}")
    return tensor


class StateDictConstant:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class LazyParams:
    def __init__(self, meta_path="const_meta.json", data_path=None, fake=True, empty=False):
        self.meta_path = meta_path
        self.data_path = data_path
        self.data = None
        self.fake = fake and data_path is None
        self.empty = empty
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
        if not self.fake:
            assert self.data_path is not None, "data_path must be provided when fake=False"
            with open(self.data_path, "rb") as f:
                self.data = torch.load(f, map_location="cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, const_name):
        const_meta = self.meta[const_name]
        shape = const_meta["shape"]
        dtype = getattr(torch, const_meta["dtype"].split(".")[1])
        if len(shape) == 0:
            shape = (1,)
        if self.empty:
            return torch.empty(*shape, device="meta", dtype=dtype)
        decompressed_tensor = const_meta["summary"]
        if const_meta["summary"]["type"] == "dynamic_quantiles_stats" and self.data is None:
            assert (
                const_meta["summary"]["type"] == "dynamic_quantiles_stats"
            ), f"Expected dynamic_quantiles_stats tensor type, got {const_meta['summary']['type']}"
            if const_meta["is_state_dict"]:
                print(
                    f"Warning: Generating {const_name} state_dict parameter from statistics, this may lead to unexpected behavior."
                )
            return decompress_tensor(decompressed_tensor, shape, dtype)
        elif const_meta["summary"]["type"] == "state_dict_constant":
            return StateDictConstant(const_name, self.data[const_name])
        elif self.data is not None and const_name in self.data:
            return self.data[const_name]
        raise RuntimeError(
            f"Constant {const_name} not found in data file. Please make sure you populate data_path correctly, or download_original_model_state_dict is implemented."
        )

    def to_dict(self):
        if self.fake:
            return {k: self[k] for k in self.meta}
        return self.data

    def from_dict(self, state_dict):
        self.data = state_dict
        self.fake = False
