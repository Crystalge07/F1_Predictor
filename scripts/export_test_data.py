"""Write the dashboard's test data: the export of the simulated test world.

    python scripts/export_test_data.py

Runs the fake world's 2027 rounds 1-3 through the real pipeline (with two
debutants in round 2, as tests/test_export_dashboard.py does) and exports
it into web/src/test/data/. The React tests load these files, so the app is
tested against exactly what the Python export writes; and
tests/test_export_dashboard.py checks they're still what a fresh export
gives, so the two sides can't drift apart. Re-run it after changing the
export, and commit the result.

No network (the world is fake), ~10 seconds.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "src" / "test"
sys.path.insert(0, str(ROOT / "tests"))      # the fake world lives with the tests


def main() -> None:
    from f1predictor.export.dashboard import export
    from f1predictor.predict import stage
    from test_export_dashboard import DEBUTANTS
    from test_simulated_weekend import END, simulate

    stage.model_version = lambda: "abc1234+settings.00000000"    # as the tests do
    conn, _ = simulate(debutants=DEBUTANTS)
    shutil.rmtree(OUT / "data", ignore_errors=True)
    version = export(conn, OUT, now=END)
    print(f"wrote {version['files'] + 1} files to {OUT / 'data'}")


if __name__ == "__main__":
    main()
