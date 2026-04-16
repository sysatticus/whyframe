"""Whyframe CLI entry point."""
import sys

from whyframe.cli import main as cli_main
from whyframe.setup import run_setup


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        run_setup()
    else:
        cli_main()


if __name__ == "__main__":
    main()