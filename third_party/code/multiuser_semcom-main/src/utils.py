"""
    filled with random functions that i kinda need everywhere...
"""

import random
import traceback
import sys
import numpy as np
import colorama
import torch
from typing import *
import itertools
import platform
import psutil
import subprocess
from PIL import Image, ImageFont, ImageDraw
import contextlib
import joblib
from pathlib import Path
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset, Subset
import inspect
import functools
import types
import datetime
from sklearn.metrics import classification_report, accuracy_score

try:
    import torch.multiprocessing as mp
except:
    import multiprocessing as mp
from queue import Empty

def fixSeed(seed):
    """
        No use. Just to see what is there to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def toColor(text: str, color: str, other: str='') -> str:
    """
        Make a colored (ANSI) string.
        
        Args:
            text: your stuff. Can be anything, will be str()-ed
            color: the color of your text, must be colorama supported. 
                   e.g. 'yellow', 'cyan'
            other: other attribute that you wanna add to the string
                   e.g. colorama.Style.BRIGHT
        
        Returns:
            An ANSI-colored string.
    """
    return f'{getattr(colorama.Fore, color.upper())}{other}{text}{colorama.Style.RESET_ALL}'

def tensor_real2complex(input: torch.Tensor, method: Literal['view', 'concat']) -> torch.Tensor:
    """
        Args:
            input: real tensor with shape (*, N), where N is even. only supports floating precision tensors
            mode: the method of which the tensors are converted.
                  For example, given tensor [a, b, c, d]:
                  - view: [a + bi, c + di] (i.e., torch.view_as_complex())
                  - concat: [a + ci, b + di] (i.e., input = [Re(return), Im(return)])

        Returns:
            complex tensor with shape (*, N/2)
            will return a copy! (i don't really have a way to do inplace in concat mode anyway) 
            # NOTE: this is wrong, view_as_complex_copy() cannot do auto differentiation so i change it to view_as_complex
    """
    if method == 'view':
        size = (*input.size()[:-1], input.size()[-1] // 2, 2)
        input = input.reshape(size).contiguous()
        ret = torch.view_as_complex(input)
    else:   # concat
        half = input.shape[-1] // 2
        real_part = input[..., :half]
        imag_part = input[..., half:]
        ret = torch.complex(real_part, imag_part)
    return ret

def tensor_complex2real(input: torch.Tensor, method: Literal['view', 'concat']) -> torch.Tensor:
    """
        Args:
            input: complex tensor with shape (*, N)
            mode: the method of which the tensors are converted.
                  For example, given tensor [a + bi, c + di]:
                  - view: [a, b, c, d] (i.e., torch.view_as_complex())
                  - concat: [a, c, b, d] (i.e., return = [Re(input), Im(input)])

        Returns:
            real tensor with shape (*, N*2)
            will return a copy!
    """ 
    real_part = input.real      # size (*, N)
    imag_part = input.imag
    
    if method == 'view':
        # (*, N, 2) -> (*, N*2)
        return torch.stack([real_part, imag_part], dim=-1).view(*input.shape[:-1], -1)
    elif method == 'concat':
        return torch.cat([real_part, imag_part], dim=-1)

def to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    elif isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data
    
class LimitedDataLoader:
    """
    Basically iterator take n but works on dataloader.
    """
    def __init__(self, dataloader, limit):
        self.dataloader = dataloader
        self.limit = limit
    
    def __iter__(self):
        return itertools.islice(self.dataloader, self.limit)
    
    def __len__(self):
        return min(self.limit, len(self.dataloader))

class InfiniteDataLoader:
    """
    Dataloader that will loop infinitely.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)  # Reset the data loader
            data = next(self.data_iter)
        return data
    
def pad_tensor_batch(inputs, pad_value=0):
    """
        Pad a list of variable length tensors with padding_value.

        inputs can be list of sequences with size (* , L, *), where L is length of the sequence and * is any number of dimensions (including 0).
        
        Args:
            inputs: A list of batches of tensors with shape (* , L, *)
    """
    n_vecs = [x.shape[1] for x in inputs]
    n_max = max(n_vecs)
    
    symbol_length = inputs[0].shape[-1]
    s_pads = []
    for i, x in enumerate(inputs):
        batch_size = x.shape[0]
        s_pad = x.new_full((batch_size, n_max, symbol_length), pad_value)
        L = x.shape[1]
        s_pad[:,:L] = x
        s_pads.append(s_pad)
        
    return s_pads    


def str_type_indent(obj, iter_limit_items: int = 10, dict_limit_items: int = 1000000, array_limit_items: int = -1,
                    explicit_type: bool=False, indent="    ") -> str:
    """
        ref. str_type

        NOTE: The indent code (str_indent) wouldn't be really fast since it reconstructs the string every time
        it is indented. For smaller object it's fine but for big and deep object it would
        cause some problems indenting a big chunk of text
    """
    def str_indent(s, indent: str):
        """
            to indent a whole multiline block
            s: the multiline block
        """
        return '\n'.join([f'{indent}{line}' for line in s.split('\n')])

    # ---

    def str_str(obj: str):
        return f'"{obj}"'
    
    def str_direct(obj):
        "for simple things that doesn't need the type to show, e.g., python's int, float, None"
        return f'{obj}'

    def str_object(obj: object):
        "default case, for anything that is not listed below, including np.float32 or something like that"
        return f'{str(obj.__class__)[8:-2]}({obj})'
    
    def str_iterable_inner(obj: Iterable, limit=iter_limit_items, is_obj_str_list: bool = False):
        """
            used when you wanna iterate something, it deals with the indentation and limit

            f"List[len=4]({str_iterable_inner([1, 2, 3, [4, 5]])})"
            -> List[len=4](
                   1, 
                   2, 
                   3, 
                   List[len=2](4, 5)
               )
            
            Args:
                limit: the maximum number of items to show, the rest would be shown as "... (%d more)"
                is_obj_str_list: whether the object is a list of object strings (objects that is already "dfs()"ed)
                                if so, won't dfs again
                                used when you have special stringify function for the iterated object,
                                e.g., dict
        """
        if is_obj_str_list:
            s_ls = obj
        else:
            # stringify the iterated object
            # the if-else is to prevent dfs()ing the object over the limit count
            # since we won't see them in the result anyway
            s_ls = [dfs(o) if _ < limit else None for _, o in enumerate(obj)]
            
        if len(s_ls) == 0:
            # just don't do intent at all
            s = f""
        else:
            if len(s_ls) > limit:
                # limit the items
                s_ls = s_ls[:limit] + [f'...({len(s_ls) - limit} more)']
            
            if all([type_map.get(o.__class__, None) == str_direct for _, o in zip(range(limit), obj)]):
                # if all items are direct, don't do indentation
                s = ", ".join(s_ls)
            else:
                # do indentation
                inner = ", \n".join(s_ls)
                s = f"\n{str_indent(inner, indent)}\n"
        return s
    
    def str_iterable(obj: Iterable):
        "for default iterables that i don't really know the type of"
        s_ls = [dfs(o) for o in obj]
        s = f"{str(obj.__class__)[8:-2]}[len={len(s_ls)}]({str_iterable_inner(obj)})"
        return s
    
    def str_list(obj: list):
        return f'List[len={len(obj)}]({str_iterable_inner(obj)})'
    
    def str_tuple(obj: tuple):
        return f'Tuple[len={len(obj)}]({str_iterable_inner(obj)})'
    
    def str_tensor(obj: torch.Tensor):
        if obj.nelement() <= array_limit_items:
            return f'torch.Tensor[size={list(obj.size())}, dtype={obj.dtype}, dev={obj.device}](\n{str_indent(str(obj), indent)}\n)'
        return f'torch.Tensor[size={list(obj.size())}, dtype={obj.dtype}, dev={obj.device}]()'
    
    def str_np_ndarray(obj: np.ndarray):
        if obj.size <= array_limit_items:
            return f'np.ndarray[shape={list(obj.shape)}, dtype={obj.dtype}](\n{str_indent(str(obj), indent)}\n)'
        return f'np.ndarray[shape={list(obj.shape)}, dtype={obj.dtype}]()'
    
    def str_set(obj: set):
        return f'Set[len={len(obj)}]({str_iterable_inner(obj)})'
    
    def str_dict(obj: set):
        s_ls = [f'{dfs(key)}: {dfs(value)}' for key, value in obj.items()]
        s = f'Dict[len={len(s_ls)}]({str_iterable_inner(s_ls, dict_limit_items, True)})'
        return s
    
    def str_dataloader(obj: DataLoader):
        # `shuffle` is not a variable, need to check the sampler
        shuffle = isinstance(obj.sampler, torch.utils.data.RandomSampler) 
        
        # dataloader may be multiuser dataloader, in this case we can get the batch_size and n_user 
        # by accessing closure of collate_fn
        collate_fn = obj.collate_fn
        if collate_fn.__closure__:
            collate_function, per_user_batch_size, n_user = [cell.cell_contents for cell in collate_fn.__closure__]
            prop_dict = {
                'per_user_batch_size': per_user_batch_size,
                'n_user': n_user,
                'pin_memory': obj.pin_memory,
                'num_workers': obj.num_workers,
                'batch_size': obj.batch_size,
                'dataset': obj.dataset,
            }   
            s_ls = [f'{key}: {dfs(value)}' for key, value in prop_dict.items()]
            return f'MultiuserDataLoader({str_iterable_inner(s_ls, 100000, True)})'
        else:
            prop_dict = {
                'pin_memory': obj.pin_memory,
                'num_workers': obj.num_workers,
                'batch_size': obj.batch_size,
                'dataset': obj.dataset,
            }   
            s_ls = [f'{key}: {dfs(value)}' for key, value in prop_dict.items()]
            return f'DataLoader({str_iterable_inner(s_ls, 100000, True)})'
    
    def str_dataset(obj: Dataset):
        if isinstance(obj, Subset):
            prop_dict = {
                'indices': obj.indices,
                'dataset': obj.dataset,
            }   
            s_ls = [f'{key}: {dfs(value)}' for key, value in prop_dict.items()]
            return f'SubsetDataset({str_iterable_inner(s_ls, 100000, True)})'
        return f'Dataset(\n{str_indent(str(obj), indent)}\n)'

    def str_limited_dataloader(obj: LimitedDataLoader):
        return f'LimitedDataLoader[limit={obj.limit}](\n{str_indent(dfs(obj.dataloader), indent)}\n)'

    
    if explicit_type:
        # make simple types also show the type
        str_direct = str_object

    type_map = {
        list: str_list,
        tuple: str_tuple,
        torch.Tensor: str_tensor,
        np.ndarray: str_np_ndarray,
        set: str_set,
        dict: str_dict,
        str: str_str,
        int: str_direct,
        float: str_direct,
        None.__class__: str_direct,
        # torch.utils.data.DataLoader: str_dataloader,
        torch.utils.data.Dataset: str_dataset,
        LimitedDataLoader: str_limited_dataloader,
        torch.optim.lr_scheduler.LRScheduler: get_scheduler_str,
        Iterable: str_iterable,
    }

    def dfs(obj: object):
        for tp, str_fn in type_map.items():
            if isinstance(obj, tp):
                return str_fn(obj)
        return str_object(obj)

    return dfs(obj)

def str_type(obj, iter_limit_items: int = 10, dict_limit_items: int = 1000000, array_limit_items: int = -1,
             explicit_type: bool=False, indent: int | str | None = None):
    """
        Actually dump everything about... a thing.
        can handle list and tensors and all kinds of stuff.
        useful when you don't know what a thing is and don't wanna just print()
        and see a bunch of tensor values, such as the output of dataloader...

        e.g., 
            >>> str_type({1: 2, 3: [4, 5], 6: torch.randn(1, 2, 3), 7: "hello", 8: np.array([2, 3, 4]), 9: ["aaa", "bbb", ""]}, indent=4)
            Dict[len=6](
                1: 2, 
                3: List[len=2](4, 5), 
                6: torch.Tensor[size=[1, 2, 3], dtype=torch.float32, dev=cpu](), 
                7: "hello", 
                8: np.ndarray[shape=[3], dtype=int64](), 
                9: List[len=3](
                    "aaa", 
                    "bbb", 
                    ""
                )
            )
            >>> str_type([torch.randn(1, 2, 3), 17239813], explicit_type=True, indent=4)
            List[len=2](
                Tensor[size=[1, 2, 3], dtype=torch.float32](), 
                int(17239813)
            )
            >>> str_type([torch.randn(1, 2, 3), torch.randn(3, 5, 100), 17239813], explicit_type=True)
            List[len=2](torch.Tensor[size=[1, 2, 3], dtype=torch.float32, dev=cpu](), int(17239813))

        Args:
            obj: the object to be dumped
            iter_limit_items: the maximum number of items to show in an iterable (other than dict)
                              if the size of the iter is larger than this, will only print up to this amount of items
                              and skip the rest by adding '... (%d more)' at the end
            dict_limit_items: the maximum number of items to show in a dict
            array_limit_items: the maximum number of items to show in an array-like object (i.e., tensor, np.ndarray)
                               if the size of the array is larger than this, no content would be printed
            explicit_type: whether to show the type of simple objects (e.g., int, float, None)
            indent: the indentation of the string. 
                    if int, it will be the number of spaces to indent
                    if str, it will be the string to indent
                    if None, the string will be returned without newlines
    """
    if indent is None:
        return str_type_indent(obj, iter_limit_items, dict_limit_items, array_limit_items, explicit_type, '').replace('\n', '')
    elif isinstance(indent, int):
        return str_type_indent(obj, iter_limit_items, dict_limit_items, array_limit_items, explicit_type, ' ' * indent)
    else:
        return str_type_indent(obj, iter_limit_items, dict_limit_items, array_limit_items, explicit_type, str(indent))

class LimitedDataLoader:
    """
        Basically iterator take n but works on dataloader.
    """
    def __init__(self, dataloader, limit):
        self.dataloader = dataloader
        self.limit = limit
    
    def __iter__(self):
        return itertools.islice(self.dataloader, self.limit)
    
    def __len__(self):
        return min(self.limit, len(self.dataloader))

class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)  # Reset the data loader
            data = next(self.data_iter)
        return data

def get_gpu_info() -> str:
    """
        dump gpu info of a machine
    """
    import torch
    if not torch.cuda.is_available():
        return "No GPU available"
    
    gpu_str_ls = []
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        memory_gb = f"{torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f}"
        gpu_str_ls.append(f"GPU {i}: {torch.cuda.get_device_name(i)} ({memory_gb} GB)")
        
    return '\n'.join(gpu_str_ls)

def get_system_info() -> str:
    # OS and kernel information
    system = platform.system()
    distro = platform.freedesktop_os_release()['PRETTY_NAME'] if system == "Linux" else platform.platform()
    kernel = platform.release()
    
    # CPU information
    try:
        from cpuinfo import get_cpu_info
        cpu_info = get_cpu_info()
        cpu_info = f"{cpu_info['brand_raw']} ({cpu_info['arch']})"
    except ImportError:
        cpu_info = platform.processor()
    
    # Memory information
    mem = psutil.virtual_memory()
    total_mem = f"{mem.total / (1024 ** 3):.2f} GB"
    used_mem = f"{mem.used / (1024 ** 3):.2f} GB"
    
    # GPU information
    gpu_info = get_gpu_info()
    
    # Formatting the system information string
    system_info = (
        f"OS: {distro}\n"
        f"Kernel: {kernel}\n"
        f"CPU: {cpu_info}\n"
        f"Memory: {used_mem} / {total_mem}\n"
        f"{gpu_info}\n"
    )
    
    return system_info

def get_class_str(obj, **kwargs) -> str:
    """
        Use to make object string
        e.g., for a RayleighFadingMultiD2DChannel, its __str__ can output:
        src.channel.RayleighFadingMultiD2DChannel(snr_db=[1, 1, 1], channel_gain_var=[[1, 1, 1], [1, 1, 1], [1, 1, 1]], divide_gain=True)
    """
    return f"{str(obj.__class__)[8:-2]}({', '.join(f'{key}={value}' for key, value in kwargs.items())})"

import torch.optim.lr_scheduler as lr_scheduler
def get_scheduler_str(scheduler: lr_scheduler.LRScheduler) -> str:
    """
        Scheduler don't have a __str__ implementation so I asked chatgpt to make one
        I'm guessing it is doing a bad job so there may be exceptions...

        No LambdaLR, MultiplicativeLR because it's not meaningful to print lambda functions
    """
    if isinstance(scheduler, lr_scheduler.StepLR):
        return f'StepLR(optimizer, step_size={scheduler.step_size}, gamma={scheduler.gamma})'
    
    elif isinstance(scheduler, lr_scheduler.MultiStepLR):
        return f'MultiStepLR(optimizer, milestones={scheduler.milestones}, gamma={scheduler.gamma})'
    
    elif isinstance(scheduler, lr_scheduler.ConstantLR):
        return f'ConstantLR(optimizer, factor={scheduler.factor}, total_iters={scheduler.total_iters})'
    
    elif isinstance(scheduler, lr_scheduler.LinearLR):
        return f'LinearLR(optimizer, start_factor={scheduler.start_factor}, end_factor={scheduler.end_factor}, total_iters={scheduler.total_iters})'
    
    elif isinstance(scheduler, lr_scheduler.ExponentialLR):
        return f'ExponentialLR(optimizer, gamma={scheduler.gamma})'
    
    elif isinstance(scheduler, lr_scheduler.PolynomialLR):
        return f'PolynomialLR(optimizer, power={scheduler.power}, total_iters={scheduler.total_iters})'
    
    elif isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
        return f'CosineAnnealingLR(optimizer, T_max={scheduler.T_max}, eta_min={scheduler.eta_min})'
    
    elif isinstance(scheduler, lr_scheduler.ChainedScheduler):
        return f'ChainedScheduler(schedulers={[get_scheduler_str(s) for s in scheduler._schedulers]})'
    
    elif isinstance(scheduler, lr_scheduler.SequentialLR):
        return f'SequentialLR(optimizer, schedulers={[get_scheduler_str(s) for s in scheduler._schedulers]}, milestones={scheduler._milestones})'
    
    elif isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        return (f'ReduceLROnPlateau(optimizer, mode={scheduler.mode}, factor={scheduler.factor}, '
                f'patience={scheduler.patience}, threshold={scheduler.threshold}, threshold_mode={scheduler.threshold_mode})')
    
    elif isinstance(scheduler, lr_scheduler.CyclicLR):
        return (f'CyclicLR(optimizer, base_lr={scheduler.base_lr}, max_lr={scheduler.max_lr}, '
                f'total_size={scheduler.total_size}, step_ratio={scheduler.step_ratio})')
    
    elif isinstance(scheduler, lr_scheduler.OneCycleLR):
        return (f'OneCycleLR(optimizer, _schedule_phases={scheduler._schedule_phases}, total_steps={scheduler.total_steps}, '
                f'epochs={scheduler.epochs})')
    
    elif isinstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
        return (f'CosineAnnealingWarmRestarts(optimizer, T_0={scheduler.T_0}, T_mult={scheduler.T_mult}, '
                f'eta_min={scheduler.eta_min})')
    return str(scheduler)

def get_callable_str(fn):
    """
    Get a string representation of a callable object
    Can support:
    ```
        def foo(): pass
        class A:
            def method(self): pass
        import functools, joblib

        # lambda, (built-in) function, (main module) function, (other module) function
        # (main modules won't show module name (__main__))
        print(get_callable_str(lambda x: x))             # "<lambda>"
        print(get_callable_str(print))                   # "<built-in> print"
        print(get_callable_str(foo))                     # "foo"

        # method, class, partial
        print(get_callable_str(A().method))              # "A.method"
        print(get_callable_str(A))                       # "A"
        print(get_callable_str(functools.partial(foo)))  # "partial(foo)"

        # (other module) class / function
        print(get_callable_str(joblib.Parallel))         # "joblib.parallel.Parallel"
        print(get_callable_str(joblib.delayed))          # "joblib.parallel.delayed"
    ```
    """
    if isinstance(fn, functools.partial):
        return f"partial({get_callable_str(fn.func)})"

    if inspect.isfunction(fn):
        if fn.__name__ == "<lambda>":
            return "<lambda>"
        if fn.__module__ == '__main__':
            return f"{fn.__qualname__}"
        return f"{fn.__module__}.{fn.__qualname__}"

    if inspect.ismethod(fn):
        self_obj = fn.__self__
        if inspect.isclass(self_obj):  # This is a classmethod
            cls_name = self_obj.__name__
        else:                          # This is an instance method
            cls_name = type(self_obj).__name__
        return f"{cls_name}.{fn.__name__}"

    if inspect.isclass(fn):
        if fn.__module__ == '__main__':
            return f"{fn.__name__}"
        return f"{fn.__module__}.{fn.__name__}"

    if isinstance(fn, types.BuiltinFunctionType):
        return f"<built-in> {fn.__name__}"

    if callable(fn):
        return f"{type(fn).__name__}.__call__"

    return str(fn)

def get_git_commit_hash() -> str:
    """
    Get the current Git commit hash of the repository.
    
    Returns:
        str: The Git commit hash or 'unknown' if the commit hash can't be retrieved.
    """
    try:
        # Run git command to get the short commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError:
        return 'unknown'

def color_from_text(s: str) -> Tuple[int, int, int, int]:
    """
    Convert a color name to its corresponding RGBA tuple.

    Args:
        s (str): The name of the color. Supported colors are:
                 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.

    Returns:
        Tuple[int, int, int, int]: The RGBA tuple corresponding to the color name.

    Raises:
        ValueError: If the color name is not supported.
    """
    color_map = {
        'black': (0, 0, 0, 255),
        'red': (255, 0, 0, 255),
        'green': (0, 255, 0, 255),
        'yellow': (255, 255, 0, 255),
        'blue': (0, 0, 255, 255),
        'magenta': (255, 0, 255, 255),
        'cyan': (0, 255, 255, 255),
        'white': (255, 255, 255, 255),
    }
    color = s.strip().lower()
    if color not in color_map:
        raise ValueError(f"Invalid color: {color}. Supported colors are: {', '.join(color_map.keys())}")
    return color_map[color]

def text_to_image(
    text: str,
    font_size: int,
    text_color: Tuple[int, int, int, int] | str,
    bg_color: Tuple[int, int, int, int] | str,
    font_filepath: str = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
):
    """
        Converts a given text to an image with specified font and colors.
        Args:
            text (str): The text to be converted into an image.
            font_size (int): The size of the font.
            text_color (Tuple[int, int, int, int] | str): The color of the text in RGBA format.
                                                    (the value range is [0, 255])
                                                    can support ANSI supported color string
            bg_color (Tuple[int, int, int, int] | str): The background color of the image in RGBA format.
                                                        can support ANSI supported color string
            font_filepath (str): The file path to the font to be used.
                                 defaults to some ubuntu font
        Returns:
            Image: An image object with the rendered text.
    """

    if isinstance(text_color, str):
        text_color = color_from_text(text_color)
    if isinstance(bg_color, str):
        bg_color = color_from_text(bg_color)

    font = ImageFont.truetype(font_filepath, size=font_size)

    # get image size
    image = np.full(shape=(1, 1, 3), fill_value=0, dtype=np.uint8)
    image = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(image)
    size = draw.multiline_textbbox((0, 0), text=text, font=font, align='left', spacing=0)  # returns: (180, 837) which is correct"

    # actually draw the thing
    image = Image.new("RGBA", (size[2], size[3]), color=bg_color)
    draw = ImageDraw.Draw(image)
    draw.multiline_text((0, 0), text, fill=text_color, font=font, anchor=None, spacing=0, align="left", direction=None, features=None, language=None) 

    return image


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
        https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697

        Context manager to patch joblib to report into tqdm progress bar given as argument

        ```
        from math import sqrt
        from joblib import Parallel, delayed

        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))
        ```
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

@contextlib.contextmanager
def elapsed_timer():
    """
        with elapsed_timer() as elapsed:
            do_something()
            print(f"{elapsed()} seconds has passed")
            do_something_else()

        print(f"The context uses {elapsed()} seconds")
        
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

@contextlib.contextmanager
def override_stdout(file):
    """
        this is when you wanna print something to a file but don't wanna
        write something  like `print("something", file=f)` everywhere

        with open("log.txt", "w") as f:
            with override_stdout(f):
                print("hello")  # will print to f
    """
    import sys
    current_out = sys.stdout
    try:
        sys.stdout = file
        yield
    finally:
        sys.stdout = current_out

def dp_delayed(func):
    """
        basically joblib.delayed, you can use either for DeviceParallel
    """
    def wrapper(*args, **kwargs):
        return (func, args, kwargs)
    return wrapper

def calc_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, to_print=True):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    
    Args:
        y_pred: Prediction output by the model. A tensor with shape (N, )
        y_true: Ground truth.  A tensor with shape (N, )
    """
    def multiclass_acc(preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
    
    test_preds = y_pred.squeeze().detach().cpu().numpy()
    test_truth = y_true.squeeze().detach().cpu().numpy()

    # non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    # print(non_zeros)
    
    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    # corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    
    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    # pos - neg
    binary_truth = (test_truth > 0)
    binary_preds = (test_preds > 0)

    if to_print:
        # print("mae: ", mae)
        # print("corr: ", corr)
        # print("mult_acc: ", mult_a7)
        # print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
        
        # non-neg - neg
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)

        # if to_print:
        #     print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
        
        return accuracy_score(binary_truth, binary_preds)
    
def compute_acc_AVE(labels, x_labels, nb_batch=10):
    """
        From https://github.com/YapengTian/AVE-ECCV18/blob/master/supervised_main.py: compute_acc
    """
    x_labels = x_labels.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # N = int(nb_batch * 10)
    # print(str_type(x_labels)) # (batch_size, 10, 29)
    pres = x_labels.reshape(-1, x_labels.shape[-1])
    truths = labels.reshape(-1, labels.shape[-1])
    
    N = pres.shape[0]
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for j in range(N): 
        pre_labels[c] = np.argmax(pres[j, :])
        real_labels[c] = np.argmax(truths[j, :])
        c += 1
    # target_names = []
    # for i in range(29):
    #     target_names.append("class" + str(i))

    return accuracy_score(real_labels, pre_labels)

class DeviceParallel:
    """
    A class to parallelize function calls across multiple GPUs using multiprocessing.
    It's like joblib.Parallel where the tasks are given to idle processes, but the difference is that in this class,
    each process have its own unique resource (i.e., GPU) to use.

    For a toy example, let's say you wanna calculate something heavy for 100 pairs of (x, y):
        ```
        def f(x, y):
            # do something heavy with cuda
            return result
        result = [f(x, y) for x, y in zip(x_ls, y_ls)]
        ```
    But you wanna use parallelization and you have 4 GPUs available, you can instead do:
        ```
        def f(x, y, device=None):    # put device=None at last, it will be automatically assigned
            # do something heavy with device
            return result
        
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        result = DeviceParallel(devices)(delayed(f)(x, y) for x, y in zip(x_ls, y_ls))
        ```
    
    This will use 4 processes, each process will be assigned to a GPU: ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], 
    Let's say one process gets the GPU 'cuda:2', then the function f() run by this process will call f() with device='cuda:2' as the last argument.   

    During the parallelization, you can also use the activity callback to monitor the progress of the tasks.
        ```
        ls = [dp_delayed(f)(x, y) for x, y in zip(x_ls, y_ls)]
        aw = MonitorFileActivityWatcher(Path('./mon.txt'), show_queue=True, name=f'Test Parallel')
        DeviceParallel(devices=['cuda:0', 'cuda:1'], activity_callback=aw)(ls)
        # use `watch -n 0.1 cat ./mon.txt` to see the progress of the tasks
        ```
    The activity callback will be called with the status and the arguments of the task when the task is started and finished.
    For example, MonitorFileActivityWatcher() is an activity callback that dumps the current status of the parallelization to a file.
    This is useful if you need some kind of logging / monitoring / progress bar of the tasks.

    ref. DeviceParallel.Status for the possible status that can be given to an activity callback.
    ref. MonitorFileActivityWatcher for an example of an activity callback.

    NOTE: 
        1. If you wanna allow multiple processes to use the same device, you can do something like this:
                devices = ['cuda:0', 'cuda:0', 'cuda:1', 'cuda:1']  # 2 cuda:0 and 2 cuda:1 in devices
        2. tqdm_joblib() won't work on this, please implement an activity callback class if you want any form of progress bar
           e.g., MonitorFileActivityWatcher acts as a fancy progress bar
        3. I suggest that you don't pass something too "exotic" into this function that might cause problems such as memory leak,
           namely, torch cuda tensors, as they require special treatments in multiprocessing
           If you insists on doing that, please use the relevant multiprocessing implementation (e.g., 'torch') and read its related documentation
           for some niche details about how to handle them properly
           If you are not sure, pass in some normal objects (e.g., list, etc) and instantiate these exotic objects in the task functions
        4. The activity_callback will be replicated once (i.e., main process and activity watcher process will have one copy in their individual process)
           but only the activity watcher process will actually use / call its activity_callback. Just FYI.
        5. If you want this to act as a normal joblib.Parallel, you can set the device to DeviceParallel.NoDevice().
           The process getting NoDevice() will not inject the device kwarg into the function call.
           (just in case you wanna use the activity_callback thing)
    """

    class NoDevice:
        # a special device that means no device (i.e., don't even inject the device kwarg into the function call)
        def __str__(self):
            return '<NoDevice>'

    class Status:
        """
        Status for the activity callback
        When something happens, we will call the activity_callback with: 
            `activity_callback(status, *args)`
        The status and args is defined below
        """

        """
        Used when a parallel is started.
        Args:
            devices (list[Any]): The devices of this parallel call.
            tasks (list[Any]): The tasks of this parallel call.
        Note:
            The below device_idx and task_idx are the index of these two lists
        """
        START_PARALLEL = 'START_PARALLEL'

        """
        Used when a parallel is ended
        Args: None
        """
        END_PARALLEL = 'END_PARALLEL'
        
        """
        Used when a task is started
        Args:
            device_idx (int): The index of the device that the task is running on
            device (Any): The device that the task is running on (e.g., 'cuda:0', 'cuda:1')
            task_idx (int): The index of the task in the task list
            task (Any): The task itself, in the form of dp_delayed(func)(args, kwargs)
        Note:
            device = devices[device_idx], same for task
            do note that we can have multiple same devices / tasks in one parallel call, so the best way to identify which
            device / task is used is to use the device_idx and task_idx.
        """
        START_TASK = 'START_TASK'   
        
        """
        Used when a task is ended successfully
        Args: same as START_TASK
        """
        END_TASK = 'END_TASK'

        """
        Used when a task is ended with an exception
        Note that when a task is ended with an exception::
        - if exception_blocking is True, the worker process will call this status and stop running tasks
        - if exception_blocking is False, the worker process will continue to run other tasks and return the exception object as the result of that task
        Args:
            device_idx (int): The index of the device that the task is running on
            device (Any): The device that the task is running on (e.g., 'cuda:0', 'cuda:1')
            task_idx (int): The index of the task in the task list
            task (Any): The task itself, in the form of dp_delayed(func)(args, kwargs)
            exception (Any): The exception object raised
            tb_str (str): The traceback string of the exception, if available
        """
        END_TASK_EXCEPTION = 'END_TASK_EXCEPTION'

        """
        Used when some time (usually 1 second) has passed.
        This is used to update the progress bar or to see if the _activity_watcher process is still alive
        Args: None
        """
        HEARTBEAT = 'HEARTBEAT'

    def __init__(self, devices: list[Any], activity_callback: Callable = None, exception_blocking: bool = True, 
                 device_arg_name: str = 'device', mp_implementation: Literal['multiprocessing', 'torch', 'multiprocess'] = 'torch'):
        """
        Args:
            devices: A list of devices to use for parallelization. Each device will be assigned to a process.
                    The device can be anything, (e.g., ['cuda:0', 'cuda:1']), and it will be given to the task
                    as a keyword argument, e.g., `device=...`
            activity_callback: A callback function to be called when something happens.
                               When the activity in DeviceParallel.Status happens, a special process will call:
                                    `activity_callback(status, *args)`.
            exception_blocking: If True, the worker process will stop running tasks if any task raises an exception.
                                    (this also means that if any process raises an exception, the whole parallel call cannot return normally,
                                    please Ctrl+C to stop the program in that case)
                                If False, the parallel call will continue to run other tasks even if some tasks raise exceptions
                                    and return the exception object as the result of that task.
            device_arg_name: The name of the keyword argument that will be used to pass the device to the task function.
                             Defaults to 'device', so the task function in the worker holding 'cuda:1' would be called like `func(..., device='cuda:1')`.
                             If you change it to e.g., 'gpu' then it will be called like `func(..., gpu='cuda:1')`.
                             Note that this is given by the worker process, do not pass this argument in dp_delayed
            mp_implementation: what multiprocessing implementation to use.
                               - 'multiprocessing': the built-in `multiprocessing` module, nothing too special, use pickle for serialization
                               - 'torch': torch.multiprocessing, which is the default in PyTorch and is the best choice if you are passing tensors
                                          as arguments to the task functions
                                          (if you're only instantiating the tensors in the task function, use any mp_implementation you like)
                               - 'multiprocess': the `multiprocess` module, which is a fork of the multiprocessing, use dill instead of pickle
                                                 for all your weird serialization needs (e.g., lambda functions, nested functions, etc.)
                                                 EXCEPT torch tensors, please use 'torch' instead, you can only pick one :)
        """
        if mp_implementation == 'multiprocessing':
            import multiprocessing
            self.mp = multiprocessing
        elif mp_implementation == 'torch':
            import torch.multiprocessing as multiprocessing
            self.mp = multiprocessing
        elif mp_implementation == 'multiprocess':
            import multiprocess
            self.mp = multiprocess
        else:
            raise ValueError(f"Invalid mp_implementation: {mp_implementation}. Supported values are 'multiprocessing', 'torch', 'multiprocess'.")

        self.devices = devices
        self.task_queue = self.mp.Queue()
        self.result_queue = self.mp.Queue()
        self.activity_queue = self.mp.Queue()
        self.activity_callback = activity_callback
        self.exception_blocking = exception_blocking
        self.devices_arg_name = device_arg_name


        
        self.processes = []
        
    def _worker(self, device_idx: int, device: Any, task_queue: mp.Queue, result_queue: mp.Queue, activity_queue: mp.Queue):
        """
        The main worker function.
        Each worker holds a device and will run tasks on that device whenever it gets some.
        """
        no_device = isinstance(device, self.NoDevice)  # if device is NoDevice, don't even inject it into the function call
        while True:
            try:
                task_idx, task = task_queue.get(timeout=1)
            except :
                continue
            if task is None:
                break
            # get task
            activity_queue.put((self.Status.START_TASK, (device_idx, device, task_idx, task)))
            
            func, args, kwargs = task
            tb_str = None
            has_exception = False
            try:
                if no_device:
                    result = func(*args, **kwargs)
                else:
                    kwargs[self.devices_arg_name] = device  # inject the device into the kwargs
                    result = func(*args, **kwargs)
            except Exception as e:
                result = e
                tb_str = traceback.format_exc()
                has_exception = True
            
            if self.exception_blocking and has_exception:
                activity_queue.put((self.Status.END_TASK_EXCEPTION, (device_idx, device, task_idx, task, result, tb_str)))
                raise result

            result_queue.put(result)

            if has_exception:
                # print traceback
                print(f'Exception on worker {device_idx} ({device}): {result}', file=sys.stderr)
                print(tb_str, file=sys.stderr)
                activity_queue.put((self.Status.END_TASK_EXCEPTION, (device_idx, device, task_idx, task, result, tb_str)))
            else:
                activity_queue.put((self.Status.END_TASK, (device_idx, device, task_idx, task)))
    
    def _activity_watcher(self, activity_queue: mp.Queue, activity_callback: Callable):
        """
        The main activity watcher function.
        This process is responsible for interfacing with the activity_callback function.

        NOTE:
        The activity_callback will be replicated once for this process
        """
        if activity_callback is None:
            activity_callback = lambda status, *args: None  # if no callback is given, ignore all activity

        while True:
            try:
                status, args = activity_queue.get(timeout=1)
            except Empty:   # for our mp_implementations they all use queue.Empty, so i guess this is fine...? 
                activity_callback(self.Status.HEARTBEAT)
                continue

            if status == self.Status.END_PARALLEL:
                activity_callback(status, *args)
                break

            activity_callback(status, *args)

    def __call__(self, tasks: Iterable[Any]) -> list[Any]:
        # if tasks is not a list, convert it to a list so it can be iterated multiple times
        if not isinstance(tasks, list):
            tasks = list(tasks)

        # start activity queue (START_PARALLEL should be the first thing _activity_watcher processes)
        self.activity_process = self.mp.Process(target=self._activity_watcher, args=(self.activity_queue, self.activity_callback))
        self.activity_process.start()
        self.activity_queue.put((self.Status.START_PARALLEL, (self.devices, tasks)))

        # start workers
        self.processes = []
        for device_idx, device in enumerate(self.devices):
            p = self.mp.Process(target=self._worker, args=(device_idx, device, self.task_queue, self.result_queue, self.activity_queue))
            p.start()
            self.processes.append(p)

        # push tasks into the queue
        for task_idx, task in enumerate(tasks):
            self.task_queue.put((task_idx, task))

        # add stop signals
        for _ in self.processes:
            self.task_queue.put((None, None))

        # gather results
        results = []
        for _ in tasks:
            results.append(self.result_queue.get())

        # cleanup
        for p in self.processes:
            p.join()

        # cleanup activity watcher
        self.activity_queue.put((self.Status.END_PARALLEL, tuple()))
        self.activity_process.join()

        return results

class MonitorFileActivityWatcher:
    """
    A class that you can give to DeviceParallel to watch the activity of the processes.
    It will write the progress to a file s.t. you can monitor it

    For example:
        ```
        def tmp_task(i, device=None):
            import time
            print(f'i got {device}')
            time.sleep(i)

        aw = MonitorFileActivityWatcher(Path('./tmp.txt'))
        DeviceParallel(['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], aw)([dp_delayed(tmp_task)(i) for i in range(10)])
        ```
    
    When the main process is running, you can watch the file and see the progress by e.g., splitting a new terminal in tmux and run:
        ```
        watch -n 0.1 ./tmp.txt
        watch -n 0.1 'cut -c -$(tput cols) ./tmp.txt'  # if your terminal is too small, this disables line wrapping 
        ```
    
    NOTE:
    - Sometimes you might see running_count being unnaturally high (it should've always been <= device count)
        when the process will die an "unnatural death" (e.g., exception ignore in atexit, etc),
        not even the try-except block in the worker process will catch it properly, it kinda just hangs?
        or keeps running still? anyways, that will ruin the monitor state and the numbers would look weird
        im not handling that
    """

    class TaskStatus:
        QUEUEING = 'QUEUEING'
        RUNNING = 'RUNNING'
        DONE = 'DONE'
        RAISED_EXCEPTION = 'RAISED_EXCEPTION'
    class DeviceStatus:
        IDLE = -1
        EXCEPTION = -2

    def __init__(self, file_path: Path, show_queue: bool = True, name: str = None):
        """
        Args:
            file_path (Path): The path to the file where the activity will be written.
            show_queue (bool): Whether to show the queueing tasks in the file.
                If False, only the overall status of the tasks will be shown.
            name (str | None): The name of the parallel call, will be shown at the top of the file.
                If None, no name will be shown.
        """
        self.file_path = file_path
        self.show_queue = show_queue
        self.name = name

        self.task: list[Any] = None                              # list of tasks (dp_delayed(func)(args))
        self.task_status: list[str] = None                       # list of TaskStatus corresponding to each task
        self.task_exception: list[tuple[Exception, str]] = None  # list of exceptions raised
        self.devices: list[Any] = None                           # list of devices (e.g., ['cuda:0', 'cuda:1'])
        self.device_status: list[None | int] = None              # list of task index corresponding to each device
        self.is_parallel_ended = False                           # whether the parallel has ended
        self.start_time = None                                   # start time of the parallel
        self.last_task_done_time = None                          # last time a task is done

    def _task_str(self, task, bold_fn_name: bool = True) -> str:
        """
        Convert a task to a string representation.
        e.g., self._task_str(task=(tmp_task, (1, 2), {'a': 'b'}))
              >>> "task(1, 2, a='b')"

        Args:
            task (tuple): A tuple of (function, args, kwargs) representing the task.
                typically from dp_delayed(func)(args, kwargs)
            bold_fn_name (bool): Whether to bold the function name in the string representation.

        Returns:
            str: A string representation of the task.
        """
        fn, args, kwargs = task
        s = get_callable_str(fn)
        if bold_fn_name:
            s = colorama.Style.BRIGHT + s + colorama.Style.NORMAL
        s += '('
        if args:
            s += ', '.join([str_type(arg) for arg in args])
        if kwargs:
            if args:
                s += ', '
            s += ', '.join([f'{k}={str_type(v)}' for k, v in kwargs.items()])
        s += ')'
        return s

    def make_content(self) -> str:
        """
        Make the content of the file to be written.
        The string should contain basically all the status of the tasks and devices.

        TODO:
        - performance improvements (tbh you dont really need to do that unless you have a lot of tasks and show_queue=True):
            - use StringIO or store the strings in a list then join them at the end for performance?
            - cache the task_strs so that we don't have to call _task_str every time
        """

        # header
        content = f'{self.name}\n\n' if self.name is not None else ''

        if self.task is None or self.task_status is None or self.device_status is None:
            content += f'No task running\n'
            return content

        """ task status """
        total_count = len(self.task_status)
        queueing_count = sum(1 for status in self.task_status if status == self.TaskStatus.QUEUEING)
        running_count = sum(1 for status in self.task_status if status == self.TaskStatus.RUNNING)
        done_count = sum(1 for status in self.task_status if status == self.TaskStatus.DONE)
        exception_count = sum(1 for status in self.task_status if status == self.TaskStatus.RAISED_EXCEPTION)

        content += f'Task Status:\n'
        content += f'─────────────────────────────\n'
        content += (f'Total: {total_count} / '
                    f'{toColor(f"Queueing: {queueing_count}", "red")} / '
                    f'{toColor(f"Running: {running_count}", "yellow")} / '
                    f'{toColor(f"Done: {done_count}", "green")} / '
                    f'{toColor(f"Exception: {exception_count}", "cyan")}'
                    '\n\n')
        content += f'Start Time: {self.start_time}\n'
        content += f'Passed time: {datetime.datetime.now() - self.start_time}\n'

        avg_time_per_task = (self.last_task_done_time - self.start_time) / (done_count + exception_count) \
                            if self.last_task_done_time is not None else None
        est_time = avg_time_per_task * (running_count + queueing_count) if avg_time_per_task is not None else None
        est_time_dt = datetime.datetime.now() + est_time if est_time is not None else None

        
        content += f'Average time per task done: {avg_time_per_task}\n'
        content += f'Estimated remaining time: {est_time} ({est_time_dt})\n'
        content += f'─────────────────────────────\n'

        """ special messages """
        if self.is_parallel_ended:
            # ended normally (i.e., got END_PARALLEL status)
            # this happens when (1) no exception raised or (2) you set exception_blocking=False
            content += (f'{toColor("<<<", "yellow", colorama.Style.BRIGHT)} '
                        f'{toColor("PARALLEL ENDED!", "white", colorama.Style.BRIGHT)} '
                        f'{toColor(">>>", "yellow", colorama.Style.BRIGHT)}\n')
            content += f'─────────────────────────────\n'
        elif all([ds == self.DeviceStatus.EXCEPTION for ds in self.device_status]):
            # ended midway, often when some processes still has tasks to run, but all processes raised exception (not always though)
            # this happens when exception_blocking=True and all processes raised exception
            content += (f'{toColor("<<<", "red", colorama.Style.BRIGHT)} '
                        f'{toColor("STOPPED MIDWAY: EXCEPTION BLOCKED ALL WORKER PROCESSES!", "white", colorama.Style.BRIGHT)} '
                        f'{toColor(">>>", "red", colorama.Style.BRIGHT)}\n')
            content += f'─────────────────────────────\n'
        elif queueing_count == 0 and running_count == 0:
            # ended but exception cause worker process to not end properly, 
            # this happens when exception_blocking=True, and most processes are either idle or raised exception
            content += (f'{toColor("<<<", "red", colorama.Style.BRIGHT)} '
                        f'{toColor("FINISHED BUT SOME EXCEPTIONS BLOCKED SOME WORKER PROCESSES!", "white", colorama.Style.BRIGHT)} '
                        f'{toColor(">>>", "red", colorama.Style.BRIGHT)}\n')
            content += f'─────────────────────────────\n'


        """ device status """
        content += f'Running Tasks:\n' 
        for i in range(len(self.devices)):
            content += f'{toColor(f"[Device {self.devices[i]}]", "yellow")} '
            if self.device_status[i] == self.DeviceStatus.IDLE:
                content += f'(Idle)\n'
            elif self.device_status[i] == self.DeviceStatus.EXCEPTION:
                content += f'(Raised exception)\n'
            else:
                content += f'{self._task_str(self.task[self.device_status[i]])}\n'

        """ task queue """
        if self.show_queue:
            content += f'─────────────────────────────\n'

            content += f'Queue:\n'
            # show queueing & running first, and then done
            for i in range(len(self.task_status)):
                if self.task_status[i] == self.TaskStatus.QUEUEING:
                    content += f'{toColor(f"({i}) {self._task_str(self.task[i])}", "red")}\n'
                elif self.task_status[i] == self.TaskStatus.RUNNING:
                    content += f'{toColor(f"({i}) {self._task_str(self.task[i])}", "yellow")}\n'
            for i in range(len(self.task_status)):
                if self.task_status[i] == self.TaskStatus.DONE:
                    content += f'{toColor(f"({i}) {self._task_str(self.task[i])}", "green")}\n'
            for i in range(len(self.task_status)):
                if self.task_status[i] == self.TaskStatus.RAISED_EXCEPTION:
                    content += (f'{toColor(f"({i}) {self._task_str(self.task[i])}", "cyan")}\n'
                               f'   - <{toColor(str(self.task_exception[i][0].__class__)[8:-2], "magenta")}> {self.task_exception[i][0]}\n')

        if self.show_queue and any(ts == self.TaskStatus.RAISED_EXCEPTION for ts in self.task_status):
            content += f'─────────────────────────────\n'
            content += f'Exceptions:\n'
            for i in range(len(self.task_status)):
                if self.task_status[i] == self.TaskStatus.RAISED_EXCEPTION:
                    content += (f'{toColor(f"({i}) {self._task_str(self.task[i])}", "cyan")}\n'
                               f'   - <{toColor(str(self.task_exception[i][0].__class__)[8:-2], "magenta")}> {self.task_exception[i][0]}\n'
                               f'   - Traceback:\n{self.task_exception[i][1]}')

        return content

    def update_file(self, content: str):
        with open(self.file_path, 'w') as f:
            f.write(content)

    def __call__(self, status, *args):
        """
        The call function for DeviceParallel to call

        Args:
            status (DeviceParallel.Status): The status of the activity.
            *args: The arguments corresponding to the status.
        Returns:
            None
        """
        # since there might be multiple same devices / tasks in the list,
        # we mostly use idx to index the device / task 
        if status == DeviceParallel.Status.START_TASK:
            device_idx, device, task_idx, task = args
            self.task_status[task_idx] = self.TaskStatus.RUNNING
            self.device_status[device_idx] = task_idx
            
        elif status == DeviceParallel.Status.END_TASK:
            device_idx, device, task_idx, task = args
            self.task_status[task_idx] = self.TaskStatus.DONE
            self.device_status[device_idx] = self.DeviceStatus.IDLE
            self.last_task_done_time = datetime.datetime.now()

        elif status == DeviceParallel.Status.END_TASK_EXCEPTION:
            device_idx, device, task_idx, task, exception, tb_str = args
            self.task_status[task_idx] = self.TaskStatus.RAISED_EXCEPTION
            self.device_status[device_idx] = self.DeviceStatus.EXCEPTION
            self.task_exception[task_idx] = (exception, tb_str)
            self.last_task_done_time = datetime.datetime.now()

        elif status == DeviceParallel.Status.START_PARALLEL:
            devices, tasks = args
            self.task = tasks
            self.task_status = [self.TaskStatus.QUEUEING] * len(tasks)
            self.task_exception = [None] * len(tasks)
            self.devices = devices
            self.device_status = [self.DeviceStatus.IDLE] * len(devices)
            self.is_parallel_ended = False
            self.start_time = datetime.datetime.now()
            self.last_task_done_time = None

        elif status == DeviceParallel.Status.END_PARALLEL:
            self.is_parallel_ended = True

        # update the file no matter what happened (including HEARTBEAT)
        self.update_file(self.make_content())

class TestClass:
    def __init__(self, a):
        self.a = a
    def start_parallel(self):
        def test(b, device=None):
            import time
            time.sleep(random.random())  # simulate some work
            if random.random() < 0.5:
                raise ValueError(f'Simulated error for b={b} on device={device}')
            print(f'TestClass.test called with b={b}, device={device}, self.a = {self.a}')
            
        ls = [
            dp_delayed(test)(i) for i in range(10)
        ]
        print(str_type(ls))
        aw = MonitorFileActivityWatcher(Path('./mon.txt'), show_queue=True, name='Test Parallel')
        DeviceParallel(['cuda:0', 'cuda:1'], aw, exception_blocking=False, device_arg_name='device', mp_implementation='multiprocess')(ls)

if __name__ == '__main__':
    TestClass(1).start_parallel()
    print(str_type({
        1: 2, 
        3: [4, 5], 
        6: torch.randn(1, 2, 3), 7: "hello", 8: np.array([2, 3, 4]), 9: ["aaa", "bbb", ""]}, print_type='type'))
    print(str_type([1, 2, 3], print_type='type'))