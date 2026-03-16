import os
import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def has_c_compiler() -> bool:
    cc = os.environ.get("CC")
    if cc:
        compiler = cc.split()[0]
        return shutil.which(compiler) is not None
    return shutil.which("cc") is not None or shutil.which("gcc") is not None

