"""Opt-in: keep the checkpoint's quantization for a serialized-quantized MTP draft
(LKQMOE_MTP_QUANT=1).

SGLang's Qwen3.5/Qwen4 MTP builds the draft unquantized whenever the target uses
modelopt_mixed, because such checkpoints usually ship the MTP module in BF16. When
the checkpoint's `quantized_layers` does list `mtp.*` entries (for example NVFP4
W4A16 draft experts), that forces the draft MoE to BF16 shapes and every quantized
expert tensor is skipped at load. With this hook the draft keeps the mixed config:
only the listed `mtp.*` layers are quantized (the mixed config resolves every other
draft layer to unquantized), and the per-expert tensors load through the existing
non-fused expert mapping. Checkpoints without `mtp.*` entries are unaffected.
"""
import importlib.abc
import importlib.machinery
import os
import sys

_MODULES = ('sglang.srt.models.qwen3_5_mtp', 'sglang.srt.models.qwen4_exp_mtp')
_reported = False


def _mtp_layers(quant_config):
    layers = getattr(quant_config, 'quantized_layers', None) or {}
    return sorted(k for k in layers if k.startswith('mtp.'))


def patch(module):
    original = getattr(module, '_mtp_quant_config', None)
    if original is None or getattr(original, '_lkqmoe_mtp_quant', False):
        return
    # qwen4_exp_mtp imports the helper by name; wrap whichever object it holds.
    base = getattr(original, '_lkqmoe_original', original)

    def _mtp_quant_config(quant_config):
        global _reported
        if quant_config is not None and quant_config.get_name() == 'modelopt_mixed':
            layers = _mtp_layers(quant_config)
            if layers:
                if not _reported:
                    print(f'[lkqmoe] MTP draft keeps modelopt_mixed for {layers}', file=sys.stderr, flush=True)
                    _reported = True
                return quant_config
        return base(quant_config)

    _mtp_quant_config._lkqmoe_mtp_quant = True
    _mtp_quant_config._lkqmoe_original = base
    module._mtp_quant_config = _mtp_quant_config


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        fn = getattr(self.delegate, 'create_module', None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    if os.environ.get('LKQMOE_MTP_QUANT') != '1':
        return
    for name in _MODULES:
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
