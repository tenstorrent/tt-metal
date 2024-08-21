# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Union


class DotAccessDict(dict):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__


def make_dot_access_dict(dictionary: Union[dict, DotAccessDict], *, ignore_types=None) -> DotAccessDict:
    if isinstance(dictionary, DotAccessDict):
        return dictionary
    elif ignore_types is not None and isinstance(dictionary, ignore_types):
        return dictionary
    preprocessed_dictionary = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            value = make_dot_access_dict(value, ignore_types=ignore_types)
        preprocessed_dictionary[key] = value
    return DotAccessDict(preprocessed_dictionary)
