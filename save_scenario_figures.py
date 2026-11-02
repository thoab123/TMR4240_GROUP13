"""Run a Part 2 scenario from run_case_part_2.py and save all its figures.

    python save_scenario_figures.py sim1
    python save_scenario_figures.py sim1 sim3

The scenario is run exactly as ``python run_case_part_2.py <name>`` would run
it, but without opening windows.  Every figure it produces is saved to
figures/<name>_<nn>_<title>.pdf/.png next to this script, so the report
figures can be reproduced with one command.
"""
import re
import sys
from pathlib import Path

import matplotlib

# Saves figures without opening windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_case_part_2 import SCENARIOS

OUT_DIR = Path(__file__).resolve().parent / "figures"


def figure_name(fig, k):
    """Short file name from the figure title (suptitle or first axes title)."""
    title = fig._suptitle.get_text() if fig._suptitle is not None else ""
    if not title and fig.axes:
        title = fig.axes[0].get_title()
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]
    return f"{k:02d}_{slug}" if slug else f"{k:02d}"


def main(argv=None):
    names = (sys.argv[1:] if argv is None else argv) or ["sim1"]
    OUT_DIR.mkdir(exist_ok=True)
    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}; options: {', '.join(SCENARIOS)}")
        plt.close("all")
        try:
            SCENARIOS[name]()
        except NotImplementedError as exc:
            # Same behaviour as the runner: keep what was plotted before the stop
            print(f"{name}: stopped — {exc}")
        for k, num in enumerate(plt.get_fignums(), start=1):
            fig = plt.figure(num)
            stem = f"{name}_{figure_name(fig, k)}"
            fig.savefig(OUT_DIR / f"{stem}.pdf")
            fig.savefig(OUT_DIR / f"{stem}.png", dpi=200)
            print(f"saved figures/{stem}.pdf/.png")


if __name__ == "__main__":
    main()
