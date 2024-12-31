import numpy as np
import torch

class MultiTensorSystem():
    def __init__(self, n_examples, n_colors, n_x, n_y, task):
        self.n_examples = n_examples
        self.n_colors = n_colors
        self.n_directions = 4
        self.n_x = n_x
        self.n_y = n_y
        self.task = task
        self.dim_lengths = [self.n_examples, self.n_colors, self.n_directions, self.n_x, self.n_y]

    def dims_valid(self, dims):
        if (dims[3] or dims[4]) and not dims[0]:  # positions always pertain to examples
            return False
        if sum(dims[1:]) == 0:  # examples aren't special
            return False
        return True

    def shape(self, dims, extra_dim=None):
        shape = []
        for dim, length in enumerate(self.dim_lengths):
            if dims[dim]:
                shape.append(length)
        if extra_dim is not None:
            shape.append(extra_dim)
        return shape

    def __iter__(self):
        for i in range(32):
            dims = [(i//2**j) % 2 for j in range(5)]
            if self.dims_valid(dims):
                yield dims
    
    def make_multitensor(self, default=None, index=-1):
        if index == -1:
            return MultiTensor(self.make_multitensor(default=default, index=0), self)
        elif index == 5:
            return default                #[]AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        else:
            return [self.make_multitensor(default=default, index=index+1) for i in range(2)]


class MultiTensor():

    def __init__(self, data, multitensor_system):
        self.data = data
        self.multitensor_system = multitensor_system

    def __getitem__(self, dims):
        return self.data[dims[0]][dims[1]][dims[2]][dims[3]][dims[4]]

    def __setitem__(self, dims, value):
        self.data[dims[0]][dims[1]][dims[2]][dims[3]][dims[4]] = value


def multify(fn):
    def multi_fn(*args, **kwargs):
        multi_mode = False
        multitensor_system = None
        for arg in args:
            if isinstance(arg, MultiTensor):
                multi_mode = True
                multitensor_system = arg.multitensor_system
        for kwarg in kwargs.values():
            if isinstance(kwarg, MultiTensor):
                multi_mode = True
                multitensor_system = kwarg.multitensor_system
        if multi_mode:
            multitensor = multitensor_system.make_multitensor()
            for dims in multitensor_system:
                new_args = []
                for arg in args:
                    if isinstance(arg, MultiTensor):
                        new_args.append(arg[dims])
                    else:
                        new_args.append(arg)
                new_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, MultiTensor):
                        new_kwargs[key] = value[dims]
                    else:
                        new_kwargs[key] = value
                multitensor[dims] = fn(dims, *new_args, **new_kwargs)
            return multitensor
        else:
            return fn(None, *args, **kwargs)
    return multi_fn


def multify(fn):
    def multi_fn(*args, **kwargs):
        multi_mode = False
        multitensor_system = None
        for arg in args:
            if isinstance(arg, MultiTensor):
                multi_mode = True
                multitensor_system = arg.multitensor_system
        for kwarg in kwargs.values():
            if isinstance(kwarg, MultiTensor):
                multi_mode = True
                multitensor_system = kwarg.multitensor_system
        if multi_mode:
            return use_multitensor_system_to_multify(multitensor_system, fn)(*args, **kwargs)
        else:
            return fn(None, *args, **kwargs)
    return multi_fn

def use_multitensor_system_to_multify(multitensor_system, fn):
    def multi_fn(*args, **kwargs):
        multitensor = multitensor_system.make_multitensor()
        for dims in multitensor_system:
            new_args = []
            for arg in args:
                if isinstance(arg, MultiTensor):
                    new_args.append(arg[dims])
                else:
                    new_args.append(arg)
            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, MultiTensor):
                    new_kwargs[key] = value[dims]
                else:
                    new_kwargs[key] = value
            multitensor[dims] = fn(dims, *new_args, **new_kwargs)
        return multitensor
    return multi_fn


