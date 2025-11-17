import pandas as pd
import sys
from pathlib import Path
import re


def main(tested_model_name: str, task: str):
    # List all files in results directory
    results = {
        f.stem.split(":")[-1]: pd.read_csv(f)
        for f in Path("results").iterdir()
        if f.is_file()
        and re.match(
            rf"^{task}:{tested_model_name.replace('/', '__')}:(.+)\.csv$", f.name
        )
    }

    if not results:
        print("No results found for the given model and task.")
        return

    # Combine results into a single DataFrame
    combined_df = results[list(results.keys())[0]][
        ["id", "question", "answer", "prediction"]
    ].copy()

    for judge_model_name, df in results.items():
        combined_df[judge_model_name] = df["evaluation"]

    # Majority vote
    combined_df["majority_vote"] = combined_df.drop(
        columns=["id", "answer", "prediction"]
    ).mode(axis=1)[0]

    # Save combined results
    combined_df.to_csv(
        f"results/{task}:{tested_model_name.replace('/', '__')}.csv",
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generation.py <tested_model_name> <task>")
        sys.exit(1)

    tested_model_name, task = sys.argv[1], sys.argv[2]

    main(tested_model_name, task)
