from typing import Tuple, List, Any, Union, Dict, NewType  # , TypeVar  #, override, get_type_hints
# from typing import Optional as Opt

# type aliases
num = Union[int, float]
numList = List[num]
numTup = Tuple[num]

simpTypes = Union[str, num, bool, None]
simpList = List[simpTypes]
simpTup = Tuple[simpTypes, ...]
simpDict = Dict[str, Union[simpTypes, simpList]]
compList = List[Union[simpTypes, List[Any], Dict[str, Any]]]
compDict = Dict[str, Union[simpTypes, List[Any], Dict[str, Any]]]

pathtype = NewType('pathtype', str)
pathType = Union[pathtype, str]
