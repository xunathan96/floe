"""
There are two types of constructors: those that create "Tagged" instances (the original object is wrapped with a tag),
and those that create custom python objects.
"""
import yaml
import numpy as np
from functools import partial
from importlib import import_module
from collections import defaultdict
from utils.struct import Tagged

constructors = defaultdict(dict)

# =================================
#         tag constructors
# =================================

def tag_constructor(loader: yaml.loader, node: yaml.Node, tag: str):
    if isinstance(node, yaml.ScalarNode):
        return Tagged(tag, loader.construct_scalar(node))
    elif isinstance(node, yaml.SequenceNode):
        return Tagged(tag, loader.construct_sequence(node, deep=False))
    elif isinstance(node, yaml.MappingNode):
        return Tagged(tag, loader.construct_mapping(node, deep=False))

constructors['tag'] = {
    u'!INSTANCE': partial(tag_constructor, tag=u'!INSTANCE'),
    u'!MODULE': partial(tag_constructor, tag=u'!MODULE'),
    u'!PARTIAL': partial(tag_constructor, tag=u'!PARTIAL'),
    u'!np.ndarray': partial(tag_constructor, tag=u'!np.ndarray'),
}


# =================================
#        object constructors
# =================================

def instance_constructor(loader: yaml.loader, node: yaml.Node):
    assert isinstance(node, yaml.nodes.MappingNode)
    data = loader.construct_mapping(node, deep=True)
    assert 'module' in data and 'args' in data
    prefix, name = data['module'].rsplit('.', 1)
    module = getattr(import_module(prefix), name)
    return module(**data['args'])

def module_constructor(loader: yaml.loader, node: yaml.Node):
    assert isinstance(node, yaml.nodes.ScalarNode)
    data = loader.construct_scalar(node)
    prefix, name = data.rsplit('.', 1)
    module = getattr(import_module(prefix), name)
    return module

def partial_constructor(loader: yaml.loader, node: yaml.Node):
    assert isinstance(node, yaml.nodes.MappingNode)
    data = loader.construct_mapping(node, deep=True)
    assert 'module' in data and 'args' in data
    prefix, name = data['module'].rsplit('.', 1)
    module = getattr(import_module(prefix), name)
    assert callable(module)
    return partial(module, **data['args'])

def np_array_constructor(loader: yaml.loader, node: yaml.Node):
    assert isinstance(node, yaml.nodes.SequenceNode)
    data = loader.construct_sequence(node, deep=True)
    return np.array(data)

constructors['obj'] = {
    u'!INSTANCE': instance_constructor,
    u'!MODULE': module_constructor,
    u'!PARTIAL': partial_constructor,
    u'!np.ndarray': np_array_constructor,
}

