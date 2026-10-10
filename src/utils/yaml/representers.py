import yaml
import numpy as np
from collections import defaultdict
from utils.struct import Tagged

representers = defaultdict(dict)

# =================================
#         tag representers
# =================================

def tag_representer(dumper: yaml.dumper, obj: Tagged):
    tag, value = obj.tag, obj.obj
    if tag == u'!INSTANCE':
        assert isinstance(value, dict)
        node = dumper.represent_mapping(tag, mapping=value)
    elif tag == u'!MODULE':
        assert isinstance(value, str)
        node = dumper.represent_scalar(tag, value=value)
    elif tag == u'!PARTIAL':
        assert isinstance(value, dict)
        node = dumper.represent_mapping(tag, mapping=value)
    return node

def np_array_representer(dumper: yaml.dumper, data: np.ndarray):
    assert isinstance(data, np.ndarray) and data.ndim <= 1
    return dumper.represent_sequence("!np.ndarray", data.tolist())  # !!seq default yaml sequence
    # for full representation:
    return dumper.represent_mapping(
        "!numpy.ndarray",
        {
            "dtype": str(data.dtype),
            "shape": list(data.shape),
            "data": data.tolist(),
        },
    )

representers['tag'] = {
    Tagged: tag_representer,
    np.ndarray: np_array_representer,
}


# =================================
#        object representers
# =================================

representers['obj'] = {
    # CustomClass: custom_representer,
}
