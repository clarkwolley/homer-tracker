"""
CLI for grading past predictions.

Usage:
    python -m src.predictions.grade 2025-06-15      # Grade a specific date
    python -m src.predictions.grade --lifetime       # Show running stats
"""

import sys
from src.predictions.tracker import run_grading, lifetime_stats


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.predictions.grade 2025-06-15   # Grade a date")
        print("  python -m src.predictions.grade --lifetime    # Running stats")
        return

    arg = sys.argv[1]
    if arg == "--lifetime":
        lifetime_stats()
    else:
        run_grading(arg)


if __name__ == "__main__":
    main()
