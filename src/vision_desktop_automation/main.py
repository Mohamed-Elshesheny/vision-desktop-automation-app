from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from .config import get_settings

from .workflow import run_demo, run_workflow


def _configure_logging(log_file: Path | None = None) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> "
            "<level>{level: <7}</level> "
            "{message}"
        ),
        colorize=True,
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level="DEBUG",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <7} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            encoding="utf-8",
            enqueue=True,
        )
        logger.info(f"Full debug log → {log_file}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vision-desktop-automation")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "demo"),
    )
    parser.add_argument(
        "--tag",
        default="demo",
        help="demo only — saves docs/deliverables/grounding_TAG.png",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    log_location = (
        settings.output_dir
        / "logs"
        / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    _configure_logging(log_file=log_location)

    if args.command == "demo":
        return run_demo(settings, tag=args.tag)
    return run_workflow(settings)


if __name__ == "__main__":
    raise SystemExit(main())
