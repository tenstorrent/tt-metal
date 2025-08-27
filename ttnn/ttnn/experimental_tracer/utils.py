import torch
import functools
import gzip
import json


def _tensor_info(obj):
    if isinstance(obj, torch.Tensor) and not obj.__class__.__name__ == "Trackable_Tensor":
        return {"shape": list(obj.shape), "dtype": str(obj.dtype), "min_max": [obj.min().item(), obj.max().item()]}
    elif isinstance(obj, (list, tuple)):
        return [_tensor_info(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _tensor_info(v) for k, v in obj.items()}
    return str(type(obj))


def track_input_output(_tensor_io_log):
    def _track_input_output(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            input_info = _tensor_info(args)
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


def get_tensors_from_input_spec(input_specs):
    tensors = []
    for shape, dtype_str, min_max in input_specs:
        dtype = getattr(torch, dtype_str.split(".")[1])
        if torch.is_floating_point(torch.empty((), dtype=dtype)):
            # torch.randn does not support min/max, so use uniform_ after creation
            t = torch.randn(*shape, dtype=dtype)
            t = t * (min_max[1] - min_max[0]) + min_max[0]
        else:
            # torch.randint's high is exclusive, so add 1 if min==max
            low, high = min_max
            if high == low:
                high = low + 1
            t = torch.randint(low, high, shape, dtype=dtype)
        tensors.append(t)
    return tensors


class LazyParams:
    def __init__(self, meta_path="const_meta.json", data_path="graph.pth.gz", fake=False):
        self.meta_path = meta_path
        self.data_path = data_path
        self.data = None
        self.fake = fake
        if not fake:
            self.data = torch.load(gzip.open(self.data_path, "rb"))
        else:
            # load const meta json and fake the data
            with open(self.meta_path, "r") as f:
                self.meta = json.load(f)

    def __getitem__(self, const_name):
        if self.fake:
            const_meta = self.meta[const_name]
            shape = const_meta["shape"]
            dtype = getattr(torch, const_meta["dtype"].split(".")[1])
            min_max = const_meta["min_max"]
            if torch.is_floating_point(torch.empty((), dtype=dtype)):
                # torch.randn does not support min/max, so use uniform_ after creation
                t = torch.randn(*shape, dtype=dtype)
                t = (t - t.min()) / (t.max() - t.min())
                t = t * (min_max[1] - min_max[0]) + min_max[0]
                return t
            low, high = min_max
            if high == low:
                high = low + 1
            t = torch.randint(low, high, shape, dtype=dtype)
            return t
        return self.data[const_name]
