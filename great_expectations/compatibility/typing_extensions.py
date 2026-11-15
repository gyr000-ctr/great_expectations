from __future__ import annotations

import sys
from typing import Any, Callable, TypeVar

try:
    # default to the typing_extensions version if available as it contains bug fixes & improvements
    from typing_extensions import Annotated
except ImportError:
    from typing import Annotated

try:
    from typing_extensions import override
except ImportError:
    F = TypeVar("F", bound=Callable[..., Any])

    def override(__arg: F, /) -> F:
        return __arg


# ``typing.Self`` was added in Python 3.11. On older interpreters typing_extensions (a declared
# runtime dependency of this project) provides it; the version check keeps static type checkers
# from seeing two conflicting definitions of the name.
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


__all__ = ["Annotated", "Self", "override"]
