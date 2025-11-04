import PIL
from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
from utils.evaluation import evaluate_answers
from utils.image import get_asset


def load_calcul_conversation_gold_dataset(dataset_path: Path):
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
                )[["id", "conversation_id", "file", "question", "answer"]]
            break

    df["previous_questions"] = [[] for _ in range(len(df))]
    df["previous_answers"] = [[] for _ in range(len(df))]

    for conv_id, group in df.groupby("conversation_id", sort=False):
        prev_qs, prev_as = [], []
        for idx in group.index:
            # Assign accumulated context
            df.at[idx, "previous_questions"] = prev_qs.copy()
            df.at[idx, "previous_answers"] = prev_as.copy()

            # Update context history for next step
            prev_qs.append(group.at[idx, "question"])
            prev_as.append(group.at[idx, "answer"])

    return df


def build_conversational_prompt(
    previous_questions: list[str],
    previous_answers: list[str],
    question: str,
    image: str,
) -> list[dict]:
    prompt = []

    for i in range(len(previous_questions)):
        if i == 0:
            prompt.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": f"Answer all questions concisely based on the image provided. Don't include any explanations. {previous_questions[i]}"},
                    ],
                }
            )
        else:
            prompt.append(
                {
                    "role": "user",
                    "content": previous_questions[i]
                }
            )

        prompt.append(
            {
                "role": "assistant",
                "content": f"Answer: {previous_answers[i]}",
            }
        )

    prompt += [
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "assistant",
            "content": "Answer:",
        },
    ]

    return prompt


def generate_calcul_conversation_gold(llm: LLM, dataset_path: Path) -> list[str]:
    ds = load_calcul_conversation_gold_dataset(dataset_path)

    prompts = ds.apply(
        lambda row: build_conversational_prompt(
            row["previous_questions"],
            row["previous_answers"],
            row["question"],
            get_asset(Path("raw_documents/dataset_calculs_conversation") / row["file"]),
        ),
        axis=1,
    ).tolist()

    outputs = llm.chat(
        prompts,
        sampling_params=SamplingParams(temperature=0, max_tokens=64, n=1),
        add_generation_prompt=False,
        continue_final_message=True,
    )

    return [output.outputs[0].text.strip() for output in outputs]


def evaluate_calcul_conversation_gold(
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
