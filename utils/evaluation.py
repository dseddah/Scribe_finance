from vllm import LLM, SamplingParams


def evaluate_answers(
    llm: LLM, questions: list[str], predictions: list[str], answers: list[str]
):
    prompts = [
        [
            {
                "role": "user",
                "content": f"Tu vas recevoir une question, une réponse de référence, et une réponse à évaluer.\nTa tâche consiste à déterminer si la prédiction est correcte ou incorrecte.\n\nRègles d’évaluation :\n1. Une prédiction est correcte si elle répond avec précision à la question en se basant sur la réponse de référence.\n2. Si la réponse concerne une valeur numérique ou financière, considère comme équivalentes les expressions représentant la même valeur. Fait attention à l'égalité entre le nombre representé avec une virgule et un point. (exemples : 20% = 0.2; 1 000 000 = 1 million; 2 200 000 = 2,2M; 12,3 = 12.3).\n3. Si la question est une question à choix multiples, une reponse est correcte si elle correspond exactement à l'une des options correctes fournies dans la réponse de référence. Elle peut etre nommée par son libellé ou par sa lettre/numero (exemple : 'A' / '1') ou par le texte associé au libellé.\n4. Si la prédiction est une paraphrase exacte de la réponse de référence, elle est correcte.\n5. Ne prends pas en compte la formulation ou le style, uniquement la justesse factuelle ou numérique.\n6. Réponds uniquement par « Correct » si la prédiction est exacte, ou « Incorrect » sinon. N’ajoute aucune explication ni commentaire supplémentaire.\n\nExemples :\n\nQuestion : Quel est le taux d’intérêt annuel ?\nRéponse de référence : 0\nRéponse à évaluer : 0.00%\nEvaluation attendue : Correct\n\nQuestion : Quelle est la valeur actuelle nette du projet ?\nRéponse de référence : 1 400 000 €\nRéponse à évaluer : 1 M€\nEvaluation attendue : Incorrect\n\nQuestion : Si j’investis 25 000 $ et que le scénario favorable se réalise au bout d’un an, combien vais-je obtenir environ à la fin de l’année ?\nA. 29 500 $\nB. 38 375 $\nC. 39 725 $\nD. 40 000 $\nRéponse de référence : C\nRéponse à évaluer : C. 39 725 $\nEvaluation attendue : Correct\n\nQuestion : Si je place 25 000 € sur la part A, combien me coûteraient les frais d’entrée maximum ?\nA. 500 €\nB. 2 %\nC. 625 €\nD. 700 €\nRéponse de référence : A\nRéponse à évaluer : 500 €\nEvaluation attendue : Correct\n\nQuestion : Si j’investis 10 000 USD et que le scénario favorable se réalise au bout d’un an, combien j’obtiens ?\nA. 13 540 USD\nB. 10 630 USD\nC. 16 410 USD\nD. 17 530 USD\nRéponse de référence : C\nRéponse à évaluer : A. 13 540 USD\nEvaluation attendue : Incorrect\n\nRéponse à évaluer :\n\nQuestion: {questions[i]}\nRéponse de référence: {answers[i]}\nRéponse à évaluer: {predictions[i]}",
            },
            {
                "role": "assistant",
                "content": "Evaluation:",
            },
        ]
        for i in range(len(predictions))
    ]

    outputs = llm.chat(
        prompts,
        sampling_params=SamplingParams(temperature=0, max_tokens=8, n=1),
        add_generation_prompt=False,
        continue_final_message=True,
    )

    return [
        True if output.outputs[0].text.strip() == "Correct" else False
        for output in outputs
    ]
