from __future__ import annotations

__version__ = "0.1.0"


def main() -> int:
    from .main import main as _main

    return _main()
