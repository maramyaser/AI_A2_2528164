
"""Run the complete reproducible ML pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_SEED = 8164


def run_script(script_path):
    """Execute a pipeline stage and stop if it fails."""
    print(f"\n{'=' * 55}", flush=True)
    print(f"Running: {script_path}", flush=True)
    print(f"{'=' * 55}", flush=True)

    if script_path.startswith("src/"):
        module_name = script_path.replace("/", ".").removesuffix(".py")
        command = [sys.executable, "-m", module_name]
    else:
        command = [sys.executable, str(ROOT / script_path)]

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )



def main():
    parser = argparse.ArgumentParser(
        description="Musanze Harvest Intelligence ML pipeline"
    )

    parser.add_argument(
        "--student-id",
        required=True,
        help="Student registration number or institutional ID",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    if args.seed != EXPECTED_SEED:
        parser.error(
            f"Seed must be {EXPECTED_SEED} for this submission."
        )

    print("Student ID:", args.student_id)
    print("Seed:", args.seed)
    print("Verification code: AI-A2-8164")

    scripts = [
        "data/generate_data.py",
        "src/regression.py",
        "src/classification.py",
        "src/clustering.py",
    ]

    for script in scripts:
        if not (ROOT / script).is_file():
            raise FileNotFoundError(
                f"Required pipeline script not found: {script}"
            )

        run_script(script)

    print("\nPipeline completed successfully!")

    print("\nGenerated artifacts:")

    for folder_name in ["artifacts", "models"]:
        folder = ROOT / folder_name

        if folder.exists():
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    print(file.relative_to(ROOT))


if __name__ == "__main__":
    main()
