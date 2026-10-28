"""odd_suite — unified ODD perception suite.

General 3D perception (Sparse4D) + ODD attribute classification + ODD judgment.
Heavy dependencies (torch / CUDA ops) are imported lazily; the mapping layer,
pipeline glue and the vendored ``odd_judge`` engine import without them.
"""

__version__ = "0.1.0"
