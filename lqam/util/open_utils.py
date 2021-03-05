import contextlib
import sys
from typing import Any, IO, Optional


# From https://stackoverflow.com/a/29824059/1165181
@contextlib.contextmanager
def smart_open(path: str, mode: Optional[str] = "r") -> IO[Any]:
    if path == "-":
        file = sys.stdin if mode is None or mode == "" or "r" in mode else sys.stdout
    else:
        file = open(path, mode)
    try:
        yield file
    finally:
        if path != "-":
            file.close()
