from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
from utils.evaluation import evaluate_answers
from utils.image import get_asset


def load_tables_yn_tf_dataset(dataset_path: Path):
    # Load json dataset
    for filesname in dataset_path.iterdir():
        if filesname.suffix == ".json" and "dataset_tables_yn_tf" in filesname.stem:
            with open(filesname, "r") as f:
                ds = pd.read_json(f).rename(
                    columns={
                        "Answer": "answer",
                        "Question": "question",
                        "Element_filename(input)": "file",
                    }
                )[["id", "file", "question", "answer"]]
            break

    return ds


def generate_tables_yn_tf(llm: LLM, dataset_path: Path) -> list[str]:
    ds = load_tables_yn_tf_dataset(dataset_path)

    prompts = ds.apply(
        lambda row: [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": get_asset(
                                Path("raw_documents/dataset_tables") / row["file"]
                            ),
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Question: {row['question']}\nAnswer the question concisely based on the image provided.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Answer:",
            },
        ],
        axis=1,
    ).tolist()

    outputs = llm.chat(
        prompts,
        sampling_params=SamplingParams(temperature=0, max_tokens=64, n=1),
        add_generation_prompt=False,
        continue_final_message=True,
    )

    return [output.outputs[0].text for output in outputs]


def evaluate_tables_yn_tf(
    judge_llm: LLM, dataset_path: Path, predictions: list[str]
) -> pd.DataFrame:
    ds = load_tables_yn_tf_dataset(dataset_path)

    ds["prediction"] = predictions

    ds["evaluation"] = evaluate_answers(
        judge_llm,
        ds["question"].tolist(),
        predictions,
        ds["answer"].tolist(),
    )

    return ds
