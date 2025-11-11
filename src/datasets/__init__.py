# src/datasets/__init__.py
# Import dataset modules so their @Registry decorators run on startup.
from .freqshapes import FreqShape
from .pam import PAM
from .boiler import Boiler
from .seqcombuv import SeqCombUV
# from .seqcombmv import SeqCombMV