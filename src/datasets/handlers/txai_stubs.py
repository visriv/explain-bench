# src/datasets/handlers/txai_stubs.py
"""
Make minimal 'txai.*' modules available so torch.load can unpickle Timex++ .pt files
without installing the real txai package. We only need classes with matching
qualified names; their state (.X, .times, .y) will be restored from the pickle.
"""

import sys
import types

# (module_path, {class_name: class_obj})
_STUBS = {
    "txai.synth_data.synth_data_base": ["SynthTrainDataset"],
    "txai.synth_data.simple_spike":    ["SpikeTrainDataset"],
    "txai.datasets.datagen.state_data": ["StateTrainDataset"],
    # add more if you hit other missing globals in your .pt files:
    # "txai.utils.data.datasets": ["DatasetwInds"],
}

def _ensure_module(path: str) -> types.ModuleType:
    if path in sys.modules:
        return sys.modules[path]
    # recursively create packages
    parts = path.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            mod = types.ModuleType(name)
            # mark packages as packages
            if i < len(parts):
                mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
    return sys.modules[path]

def install_txai_stubs():
    for mod_path, class_names in _STUBS.items():
        mod = _ensure_module(mod_path)
        for cname in class_names:
            if not hasattr(mod, cname):
                # simplest possible placeholder
                mod.__dict__[cname] = type(cname, (), {})  # empty class
