from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]
