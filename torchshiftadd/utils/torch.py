import os

import torch
from torch.utils import cpp_extension
from . import decorator, comm

class LazyExtensionLoader(object):

    def __init__(self, name, sources, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None,
                 extra_include_paths=None, build_directory=None, verbose=False, **kwargs):
        self.name = name
        self.sources = sources
        self.extra_cflags = extra_cflags
        self.extra_cuda_cflags = extra_cuda_cflags
        self.extra_ldflags = extra_ldflags
        self.extra_include_paths = extra_include_paths
        worker_name = "%s_%d" % (name, comm.get_rank())
        self.build_directory = build_directory or cpp_extension._get_build_directory(worker_name, verbose)
        self.verbose = verbose
        self.kwargs = kwargs

    def __getattr__(self, key):
        return getattr(self.module, key)

    @decorator.cached_property
    def module(self):
        return cpp_extension.load(self.name, self.sources, self.extra_cflags, self.extra_cuda_cflags,
                                  self.extra_ldflags, self.extra_include_paths, self.build_directory,
                                  self.verbose, **self.kwargs)


def load_extension(name, sources, extra_cflags=None, extra_cuda_cflags=None, **kwargs):
    """
    Load a PyTorch C++ extension just-in-time (JIT).
    Automatically decide the compilation flags if not specified.

    This function performs lazy evaluation and is multi-process-safe.

    See `torch.utils.cpp_extension.load`_ for more details.

    .. _torch.utils.cpp_extension.load:
        https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load
    """
    if extra_cflags is None:
        extra_cflags = ["-Ofast"]
        if torch.backends.openmp.is_available():
            extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags.append("-DAT_PARALLEL_NATIVE")
    if extra_cuda_cflags is None:
        if torch.cuda.is_available():
            extra_cuda_cflags = ["-O3"]
            extra_cflags.append("-DCUDA_OP")
        else:
            new_sources = []
            for source in sources:
                if not cpp_extension._is_cuda_file(source):
                    new_sources.append(source)
            sources = new_sources

    return LazyExtensionLoader(name, sources, extra_cflags, extra_cuda_cflags, **kwargs)

