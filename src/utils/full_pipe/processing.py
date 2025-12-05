from dataclasses import dataclass
import numpy as np
from openai import OpenAI
from utils.build_questions import LEGEND_MAPPINGS, TITLE_MAPPINGS, X_MAPPINGS, Y_MAPPINGS
from utils.image_prompter import Prompt
from utils.loaders import Split
from utils.visualization import plot_time_series

@dataclass
class PackagedTSData:
    # test
    test_idx:int
    test_summary: str
    test_label: int
    test_X: np.ndarray
    test_classname: str
    # train
    train_idxs:list[int]
    train_summarys:list[ str]
    train_labels:list[ int]
    train_Xs:list[ np.ndarray]
    train_classnames:list[ str]

@dataclass
class PromptRow:
    query: Prompt
    examples: list[Prompt]


def build_nn_packagedtsdata(
        train_embeddings:np.ndarray, 
        train_summaries: list[str],
        test_embeddings:np.ndarray, 
        test_summaries: list[str],
        train:Split, 
        test:Split, 
        mappings:dict
    ) -> list[PackagedTSData]:

    nns = []
    for i, test_emb in enumerate(test_embeddings):
        sims = train_embeddings @ test_emb
        nn_idxs = np.argsort(sims)[::-1][:10]
        nn_labels = [train[idx].y for idx in nn_idxs]
        nn_class_names = [mappings["id_to_name"][code] for code in nn_labels]
        nn_summaries = [train_summaries[idx] for idx in nn_idxs]
        nn_Xs = [train[idx].X for idx in nn_idxs]

        nns.append(PackagedTSData(
            test_idx=i,
            test_summary=test_summaries[i],
            test_label=test[i].y,
            test_X=test[i].X,
            test_classname=mappings["id_to_name"][test[i].y],
            train_idxs=list(nn_idxs),
            train_summarys=nn_summaries,
            train_labels=nn_labels,
            train_Xs=nn_Xs,
            train_classnames=nn_class_names,
        ))

    return nns

def build_summary_prompts(train:Split, test:Split, question: str, dataset:str
                          ) -> tuple[list[PromptRow], list[PromptRow]]:
    prompt_row_splits = []
    for split, split_name in zip([train,test],["train","test"]):
        prompt_rows = []
        for i in range(0, len(split)):
            query_img_path = plot_time_series(
                    split[i].X,
                    method="line",
                    title=TITLE_MAPPINGS[dataset.upper()],
                    xlabs=X_MAPPINGS[dataset.upper()],
                    ylabs=Y_MAPPINGS[dataset.upper()],
                    legends=LEGEND_MAPPINGS[dataset.upper()],
                    save_path=f"./data/images/{dataset}/{split_name}/{i}.png",
                    recreate=True,
                    )
            query = Prompt(
                user={"question": question},
                # user={"question": question}, # this way if you want to generate summary and answer
                img_path=query_img_path,
                img_detail="auto"
            )
            prompt = PromptRow(query, [])
            prompt_rows.append(prompt)

        prompt_row_splits.append(prompt_rows)

    return prompt_row_splits[0], prompt_row_splits[1]


def build_prompts(question:str, nn_packages:list[PackagedTSData], dataset:str) -> list[PromptRow]:
    prompts: list[PromptRow] = []
    for package in nn_packages:
        # extract from dictionary
        query_img_path = plot_time_series(
            package.test_X,
            method="line",
            title=TITLE_MAPPINGS[dataset.upper()],
            xlabs=X_MAPPINGS[dataset.upper()],
            ylabs=Y_MAPPINGS[dataset.upper()],
            legends=LEGEND_MAPPINGS[dataset.upper()],
            save_path=f"./data/images/{dataset}/test/{package.test_idx}.png",
            recreate=True,
        )

        query = Prompt(
            user={"question": question, "summary": package.test_summary},
            # user={"question": question}, # this way if you want to generate summary and answer
            img_path=query_img_path,
            img_detail="auto"
        )

        # BUILD EXAMPLES
        # reset row_examples
        examples = []
        # extract from dictionary
        for nn_idx, nn_summary, nn_X, nn_class_name in zip(
            package.nn_idxs, package.nn_summarys, package.nn_Xs, package.nn_class_names):

            ex_img_path = plot_time_series(
                nn_X,
                method="line",
                title=TITLE_MAPPINGS[dataset.upper()],
                xlabs=X_MAPPINGS[dataset.upper()],
                ylabs=Y_MAPPINGS[dataset.upper()],
                legends=LEGEND_MAPPINGS[dataset.upper()],
                save_path=f"./data/images/{dataset}/train/{nn_idx}.png",
                recreate=True,
            )

            example = Prompt(
                user = {"question": question, "summary": nn_summary},
                assistant={"content": f"The answer is {nn_class_name}"},
                # # this way if you want to generate summary and answer
                # user={"question": question},
                # assistant={"content": nn_summary + f"\n\nThe answer is {nn_class_name}"},
                img_path=ex_img_path,
                img_detail="auto"
            )
            examples.append(example)
        prompts.append(PromptRow(query, examples))