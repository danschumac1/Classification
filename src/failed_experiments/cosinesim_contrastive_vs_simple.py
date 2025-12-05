"""
python ./src/blah.py
"""

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from utils.loaders import load_train_test
from utils.file_io import load_jsonl, load_json
from utils.image_prompter import ImagePrompter, Prompt


def get_embeddings_batch(
    texts: List[str],
    client: OpenAI,
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """
    Embed a list of texts with OpenAI and return an array of
    shape (N, D) with row-wise L2 normalization.
    """
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    resp = client.embeddings.create(
        model=model,
        input=texts,
    )
    embs = np.array([d.embedding for d in resp.data], dtype=np.float32)

    # normalize each row to unit length
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms
    return embs


if __name__ == "__main__":
    # ------------------------------------------------------------
    # CONFIG (later: make these command-line args)
    # ------------------------------------------------------------
    dataset = "har"  # ctu, emg, har
    data_path = f"./data/samples/{dataset}"
    simple_features_path = f"./data/features/{dataset}/simple.jsonl"
    pair_wise_contrastive_path = (
        f"./data/features/{dataset}/10-imgs_5-rounds_contrastive.json"
    )
    contrastive_summaries_path = (
        f"./data/features/{dataset}/contrastive_generation/final_summaries.json"
    )
    mappings_path = f"./data/samples/{dataset}/label_maps.json"
    temperature = 0.7
    embedding_model = "text-embedding-3-small"

    # ------------------------------------------------------------
    # OPENAI CLIENT
    # ------------------------------------------------------------
    load_dotenv("./resources/.env")
    os.getenv("OPENAI_API_KEY")

    client = OpenAI()

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    train, test = load_train_test(
        data_path,
        n_shots=0,
    )

    simple_features = load_jsonl(simple_features_path)
    pair_wise_contrastive = load_json(pair_wise_contrastive_path)
    contrastive_summaries = load_json(contrastive_summaries_path)
    mappings = load_json(mappings_path)

    # ------------------------------------------------------------
    # SETUP IMAGE PROMPTER (for future use)
    # ------------------------------------------------------------
    prompter = ImagePrompter()
    prompter.model_name = "gpt-4o-mini"
    prompter.system_prompt = "..."  # TODO: define system prompt when needed

    # ------------------------------------------------------------
    # EXPERIMENT #1
    # Take the simple feature summaries and calculate cosine similarity
    # with contrastive summaries (via embeddings) to predict class.
    # ------------------------------------------------------------

    # 1) Prepare contrastive summary texts
    class_names = list(contrastive_summaries.keys())
    summary_texts = [contrastive_summaries[cname] for cname in class_names]

    # 2) Embed contrastive summaries once
    summary_embs = get_embeddings_batch(
        summary_texts,
        client=client,
        model=embedding_model,
    )
    if summary_embs.shape[0] == 0:
        raise ValueError("No contrastive summaries found to embed.")

    # 3) Embed all simple feature model outputs
    row_summaries = [row["model_output"] for row in simple_features]
    row_summaries_embs = get_embeddings_batch(
        row_summaries,
        client=client,
        model=embedding_model,
    )
    if row_summaries_embs.shape[0] != len(simple_features):
        raise ValueError(
            "Number of embedded simple features does not match number of rows."
        )

    # 4) Compute cosine similarity matrix
    #    (embeddings are already normalized, so dot product == cosine similarity)
    sim_matrix = row_summaries_embs @ summary_embs.T  # shape: (N_examples, N_classes)

    # 5) Evaluate predictions
    correct = 0
    total = len(simple_features)

    per_class_counts = {cname: 0 for cname in class_names}
    per_class_correct = {cname: 0 for cname in class_names}

    for i, row in enumerate(simple_features):
        gt_id = row["gt"]
        gt_class_name = mappings["id_to_name"][str(gt_id)]

        # predicted class index from similarity matrix
        pred_idx = int(np.argmax(sim_matrix[i]))
        pred_class_name = class_names[pred_idx]

        per_class_counts[gt_class_name] = per_class_counts.get(gt_class_name, 0) + 1
        if gt_class_name == pred_class_name:
            correct += 1
            per_class_correct[gt_class_name] = (
                per_class_correct.get(gt_class_name, 0) + 1
            )

    overall_acc = correct / total if total > 0 else 0.0

    print("===================================================")
    print("EXPERIMENT #1: Simple feature vs. contrastive summaries")
    print("---------------------------------------------------")
    print(f"Dataset: {dataset}")
    print(f"Total examples: {total}")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print("---------------------------------------------------")
    print("Per-class accuracy:")
    for cname in class_names:
        n = per_class_counts.get(cname, 0)
        if n == 0:
            acc = 0.0
        else:
            acc = per_class_correct.get(cname, 0) / n
        print(f"  {cname:20s}: {acc:.4f} ({per_class_correct.get(cname, 0)}/{n})")
    print("===================================================")


# EXAMPLE USAGE OF PROMPTER WITH CUSTOM PROMPTS
# (kept here for future experimentation)
#
# prompter.system_prompt = (
#     "You are a knowledgeable, reliable assistant who excels at answering "
#     "tricky questions, combining context with general world knowledge. "
#     "When helpful, you clarify ambiguities, correct misconceptions, and "
#     "provide additional insights beyond the given context."
# )
#
# example1 = Prompt(
#     user={
#         "question": "Tell me more (beyond the context) about Michael Jordan",
#         "context": (
#             "Michael I. Jordan is the Pehong Chen Distinguished Professor in the "
#             "Department of Electrical Engineering and Computer Science and the "
#             "Department of Statistics."
#         ),
#     },
#     assistant={
#         "answer": (
#             "Michael I. Jordan is a Senior Researcher at Inria, Paris, and the "
#             "Pehong Chen Distinguished Professor Emeritus at the University of California, "
#             "Berkeley. He earned his Masters in Mathematics from Arizona State University "
#             "and his PhD in Cognitive Science from UC San Diego in 1985. He later served "
#             "as a professor at MIT from 1988 to 1998. His research spans computational, "
#             "statistical, cognitive, biological, and social sciences. He is a member of the "
#             "National Academy of Sciences, the National Academy of Engineering, the American "
#             "Academy of Arts and Sciences, and a Foreign Member of the Royal Society. He is "
#             "also a Fellow of the American Association for the Advancement of Science. "
#             "His awards include the BBVA Foundation Frontiers of Knowledge Award (2025), "
#             "the WLA Prize (2022), the Ulf Grenander Prize (2021), the IEEE John von Neumann "
#             "Medal (2020), the IJCAI Research Excellence Award (2016), the Rumelhart Prize "
#             "(2015), and the ACM/AAAI Allen Newell Award (2009). He has delivered numerous "
#             "prestigious lectures, including the IMS Grace Wahba Lecture (2022), the IMS "
#             "Neyman Lecture (2011), and an IMS Medallion Lecture (2004). He was a Plenary "
#             "Lecturer at the International Congress of Mathematicians in 2018. In 2016, an "
#             "article in Science ranked him the 'most influential computer scientist' based on "
#             "Semantic Scholar metrics."
#         )
#     }
# )
#
# query = Prompt(
#     user={
#         "question": "Tell me more (beyond the context) about Will Smith",
#         "context": (
#             "Anthropologist Will Smith works across the Indo-Pacific on climate change "
#             "adaptation, rural livelihoods and environmental governance"
#         ),
#     }
# )
#
# messages = prompter.format_prompt([example1], query)
# result = prompter.get_completion(messages, temperature=temperature)
# print(result)
