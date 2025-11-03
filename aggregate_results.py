from pathlib import Path
import pandas as pd
import re


def load_dataset(task: str):
    # Take the file that containe the task name
    dataset_folder = Path("dataset_json")

    for file in dataset_folder.iterdir():
        if f"dataset_{task}" in file.name:
            df = pd.read_json(file)

            return df


def load_model_eval(model_name: str, task_name: str):
    eval_folder = Path("results")
    return pd.read_csv(eval_folder / f"{task_name}:{model_name.replace('/', '__')}.csv")


def get_results_files(task: str) -> dict[str, Path]:
    # Match ^charts:[^:]+\.py$
    results_folder = Path("results")
    files = {}
    for file in results_folder.iterdir():
        # The group is the model name
        match = re.match(rf"^{task}:([^:]+)\.csv$", file.name)

        if match:
            model_name = match.group(1).replace("__", "/")
            files[model_name] = file

    return files


def aggregate_results(
    task: str,
    dataset_df: pd.DataFrame,
    results_files: dict[str, Path],
    group_by: list[str],
) -> None:
    for group in group_by:
        aggregated_results = None

        for model_name, file_path in results_files.items():
            model_eval_df = pd.read_csv(file_path)

            print(task, model_name, group)
            # Merge dataset_df and model_eval_df on a common column, e.g., 'id'
            merged_df = pd.merge(dataset_df, model_eval_df, on="id").rename(
                columns={"majority_vote": "accuracy"}
            )[[group, "accuracy"]]

            # Group by specified columns and calculate accuracy
            grouped = merged_df.groupby([group]).mean().reset_index()

            grouped["model"] = model_name

            pivoted = grouped.pivot(index=group, columns="model", values="accuracy")

            # Add mean accuracy row
            row = pd.DataFrame(
                {
                    group: ["Average"],
                    model_name: [grouped["accuracy"].mean()],
                }
            )

            pivoted = pd.concat([pivoted, row.set_index(group)], axis=0)

            if aggregated_results is None:
                aggregated_results = pivoted
            else:
                aggregated_results = pd.concat(
                    [aggregated_results, pivoted[model_name]], axis=1
                )

        aggregated_results.to_csv(
            Path("aggregated_results") / task / f"{task}_{group}.csv"
        )


def main():
    tasks = {
        "NER": ["Type", "Named Entities", "Context size"],
        "charts": ["Type", "Question_type", "Domain", "Input_Context_Size"],
        "tables": ["Type", "Question_type", "Domain", "Input_Context_Size"],
        "tables_yn_tf": ["Type", "Question_type", "Domain", "Input_Context_Size"],
        "special_cases": ["Type", "Question_type", "Domain", "Input_Context_Size"],
        "calculs_conversation": [
            "Type",
            "Question_type",
            "Domain",
            "Input_Context_Size",
        ],
    }

    for task, group_by in tasks.items():
        dataset_df = load_dataset(task)
        results_files = get_results_files(task)

        aggregate_results(task, dataset_df, results_files, group_by=group_by)


if __name__ == "__main__":
    main()
