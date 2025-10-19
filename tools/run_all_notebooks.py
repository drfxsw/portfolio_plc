import nbformat
from nbclient import NotebookClient
from pathlib import Path
import argparse
import traceback

BASE = Path(__file__).resolve().parents[1]  # d:\workplace\portfolio_plc

PROJECT_NOTEBOOKS = {
    "project_failure": [
        "researching/01_data_exploration.ipynb",
        "researching/02_baseline_lstm.ipynb",
        "researching/03_lstm.ipynb",
        "researching/04_gru.ipynb",
        "researching/05_cnn.ipynb",
        "researching/06_model_comparison.ipynb"
    ],
    "project_defect": [
        "researching/01_data_exploration.ipynb",
        "researching/02_data_preprocessing.ipynb",
        "researching/03_logistic_regression.ipynb",
        "researching/04_random_forest.ipynb",
        "researching/05_xgboost.ipynb",
        "researching/06_model_comparison.ipynb"
    ]
}

def execute(nb_path: Path, out_path: Path, timeout=1200):
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    client.execute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=list(PROJECT_NOTEBOOKS.keys()), required=False)
    parser.add_argument("--timeout", type=int, default=1200, help="notebook execution timeout (seconds)")
    args = parser.parse_args()
    projects = [args.project] if args.project else list(PROJECT_NOTEBOOKS.keys())

    for proj in projects:
        print(f"\n=== Executing notebooks for {proj} ===")
        nb_list = PROJECT_NOTEBOOKS[proj]
        executed_dir = BASE / "executed_notebooks" / proj
        for nb_rel in nb_list:
            nb_path = BASE / proj / nb_rel
            if not nb_path.exists():
                print(f"  SKIP (not found): {nb_path}")
                continue
            out_path = executed_dir / nb_rel
            print(f"  -> {nb_rel} ...", end=" ")
            try:
                execute(nb_path, out_path, timeout=args.timeout)
                print("OK (saved to executed_notebooks)")
            except Exception:
                print("ERROR")
                traceback.print_exc()
                # continue to next notebook
        print(f"Finished project {proj}. Check {BASE / proj / 'models'} for generated results if notebooks save them.")
    print("\nAll done.")

if __name__ == "__main__":
    main()