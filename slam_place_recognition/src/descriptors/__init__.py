"""Descriptor package.

Common interface (used by :mod:`retrieval`)::

    d = Descriptor(cfg...)      # name, dimension
    d.fit([feat_frame_0, ...])  # optional normalisation state
    d.similarity(left_list, right_list) -> (k,) similarities in [0, 1]

``left``/``right`` are lists of the descriptor's own per-frame features, so the
same code path handles JSSS (multi-component dict), ScanContext (2-D matrix)
and the scalar baselines.
"""

from descriptors.baselines import (  # noqa: F401
    GlobalHistogram,
    IntensityStats,
    MeanSpectrumDescriptor,
    PureSpace,
    RadialHistogram,
    SpectralSumDescriptor,
)
from descriptors.jsss import (  # noqa: F401
    COMPONENTS,
    JSSS,
    Normalizer,
    component_slices,
    jsss_feature,
    rms_similarity,
    stack_feature,
)
from descriptors.m2dp import M2DP  # noqa: F401
from descriptors.scan_context import ScanContext  # noqa: F401

__all__ = [
    "JSSS", "ScanContext", "M2DP", "PureSpace", "RadialHistogram",
    "GlobalHistogram", "IntensityStats", "SpectralSumDescriptor",
    "MeanSpectrumDescriptor", "COMPONENTS", "Normalizer", "rms_similarity",
    "component_slices", "jsss_feature", "stack_feature",
]
