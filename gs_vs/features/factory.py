from gs_vs.features.FeatureLuminancePinhole import FeatureLuminancePinhole
from gs_vs.features.FeatureLuminanceUnifiedIP import FeatureLuminanceUnifiedIP
from gs_vs.features.FeatureLuminanceUnifiedCS import FeatureLuminanceUnifiedCS
from gs_vs.features.FeatureLuminanceUnifiedPS import FeatureLuminanceUnifiedPS
from gs_vs.features.FeatureLuminanceEquidistant import FeatureLuminanceEquidistant

FEATURE_REGISTRY = {
    "pinhole": FeatureLuminancePinhole,
    "unified_ip": FeatureLuminanceUnifiedIP,
    "unified_cs": FeatureLuminanceUnifiedCS,
    "unified_ps": FeatureLuminanceUnifiedPS,
    "equidistant": FeatureLuminanceEquidistant,
}

def create_feature(name: str, **kwargs):
    if name not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature type: {name}")
    return FEATURE_REGISTRY[name](**kwargs)
