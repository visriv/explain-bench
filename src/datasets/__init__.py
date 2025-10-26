# src/datasets/__init__.py
# Import dataset modules so their @Registry decorators run on startup.
from .freqshapes import FreqShape
from .pam import PAM

# from .pam_dataset import PAMDataset  # noqa: F401
# from .freqshape_dataset import FreqShapeDataset  # noqa: F401
