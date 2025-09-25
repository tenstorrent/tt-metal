import torch
import functools
import gzip
import json


def _tensor_info(obj):
    if isinstance(obj, torch.Tensor) and not obj.__class__.__name__ == "Trackable_Tensor" and obj.device.type != "meta":
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "min_max": (obj.min().item(), obj.max().item()) if obj.numel() > 0 else (0, 0),
        }
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


def get_tensors_from_shape_dtype_minmax(shape, dtype, min_max):
    if len(shape) == 1 and shape[0] == 0:
        return torch.tensor([], dtype=dtype)
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
    return t


def get_tensors_from_input_spec(input_specs):
    tensors = []
    for shape, dtype_str, min_max in input_specs:
        dtype = getattr(torch, dtype_str.split(".")[1])
        tensors.append(get_tensors_from_shape_dtype_minmax(shape, dtype, min_max))
    return tensors


class LazyParams:
    def __init__(self, meta_path="const_meta.json", data_path=None, fake=True, empty=False):
        self.meta_path = meta_path
        self.data_path = data_path
        self.data = None
        self.fake = fake
        self.empty = empty
        if not fake:
            assert self.data_path is not None, "data_path must be provided when fake=False"
            with open(self.data_path, "rb") as f:
                self.data = torch.load(f)
        else:
            # load const meta json and fake the data
            with open(self.meta_path, "r") as f:
                self.meta = json.load(f)

    def __getitem__(self, const_name):
        if self.fake:
            const_meta = self.meta[const_name]
            shape = const_meta["shape"]
            if len(shape) == 0:
                shape = (1,)
            dtype = getattr(torch, const_meta["dtype"].split(".")[1])
            min_max = const_meta["min_max"]
            if self.empty:
                return torch.empty(*shape, device="meta", dtype=dtype)
            return get_tensors_from_shape_dtype_minmax(shape, dtype, min_max)
        return self.data[const_name]

    def to_dict(self):
        if self.fake:
            return {k: self[k] for k in self.meta}
        return self.data

    def from_dict(self, state_dict):
        self.data = state_dict
        self.fake = False
