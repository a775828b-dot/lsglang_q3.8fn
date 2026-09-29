"""Opt-in launch dependency workaround for dynamically produced router bias.

The inspected SGLang Triton router reads bias before gdc_wait(). With PDL that
can observe the producer's stale buffer. Disable early launch in this module
only; the router's arithmetic, top-k, dtypes, and other kernels stay intact.
This is an isolated candidate until the end-to-end quality gate passes.
"""
import importlib.abc
import importlib.machinery
import sys

_NAME = 'sglang.kernels.ops.moe.moe_fused_gate'

def patch(module):
    if getattr(module, '_lkqmoe_router_dependency_guard', False):
        return
    if not callable(getattr(module, 'is_arch_support_pdl', None)):
        raise RuntimeError('Unsupported SGLang router dependency interface')
    module._lkqmoe_original_router_pdl = module.is_arch_support_pdl
    module.is_arch_support_pdl = lambda: False
    module._lkqmoe_router_dependency_guard = True

class _Loader(importlib.abc.Loader):
    def __init__(self, delegate): self.delegate = delegate
    def create_module(self, spec):
        fn = getattr(self.delegate, 'create_module', None)
        return fn(spec) if fn else None
    def exec_module(self, module):
        self.delegate.exec_module(module)
        patch(module)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _NAME:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader)
        return spec

def install():
    if _NAME in sys.modules:
        patch(sys.modules[_NAME])
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
