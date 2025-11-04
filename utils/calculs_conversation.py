import PIL
from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
from utils.evaluation import evaluate_answers
from utils.image import get_asset


def load_calcul_conversation_dataset(dataset_path: Path):
    # Load json dataset
    for filesname in dataset_path.iterdir():
        if (
            filesname.suffix == ".json"
            and "dataset_calculs_conversation" in filesname.stem
        ):
            with open(filesname, "r") as f:
                df = pd.read_json(f).rename(
                    columns={
                        "Answer": "answer",
                        "Question": "question",
                        "Element_filename(input)": "file",
                        "Element_id": "conversation_id",
                    }
                )[["id", "conversation_id", "file", "question"]]
            break

    df_grouped = df.groupby("conversation_id")["question"].apply(list).reset_index()

    df = df_grouped.merge(df[["conversation_id", "file"]], on="conversation_id")

    return df


def generate_calcul_conversation(llm: LLM, dataset_path: Path) -> list[str]:
    ds = load_calcul_conversation_dataset(dataset_path)
    outputs = []

    for index, conversation in ds.iterrows():
        conv = []
        outputs_conv = []

        for i, question in enumerate(conversation["question"]):
            if i == 0:
                conv.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": get_asset(
                                        Path(
                                            "raw_documents/dataset_calculs_conversation"
                                        )
                                        / conversation["file"]
                                    )
                                },
                            },
                            {
                                "type": "text",
                                "text": f"Answer all questions concisely based on the image provided. Don't include any explanations. {question}",
                            },
                        ],
                    }
                )
            else:
                conv.append({"role": "user", "content": question})

            outputs = llm.chat(
                conv
                + [
                    {
                        "role": "assistant",
                        "content": "Answer:",
                    },
                ],
                sampling_params=SamplingParams(temperature=0, max_tokens=64, n=1),
                add_generation_prompt=False,
                continue_final_message=True,
            )

            outputs_conv.append(outputs[0].outputs[0].text.strip())
            conv.append(
                {
                    "role": "assistant",
                    "content": f"Answer: {outputs[0].outputs[0].text.strip()}",
                }
            )

        outputs += outputs_conv

    return outputs


def evaluate_calcul_conversation(
    judge_llm: LLM, dataset_path: Path, predictions: list[str]
) -> pd.DataFrame:
    ds = load_calcul_conversation_dataset(dataset_path)

    ds["prediction"] = predictions

    ds["evaluation"] = evaluate_answers(
        judge_llm,
        ds["question"].tolist(),
        predictions,
        ds["answer"].tolist(),
    )

    return ds[
        [
            "id",
            "conversation_id",
            "file",
            "question",
            "answer",
            "prediction",
            "evaluation",
        ]
    ]
