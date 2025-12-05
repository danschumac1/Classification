"""
python ./src/blah.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from utils.build_questions import HELP_STRING, LEGEND_MAPPINGS, TITLE_MAPPINGS, X_MAPPINGS, Y_MAPPINGS
from utils.loaders import load_train_test
from utils.file_io import append_jsonl, load_jsonl, load_json
from utils.image_prompter import ImagePrompter, Prompt
from utils.visualization import plot_time_series



ARGUMENT_STRENGTH_MAP = {
    1: "Express your answer with low confidence. Emphasize uncertainty and provide caveats.",
    2: "Express your answer with moderate uncertainty. Suggest a leaning but acknowledge doubts.",
    3: "Express your answer with moderate confidence. Provide clear reasoning while allowing room for error.",
    4: "Express your answer with high confidence. Argue assertively and present your reasoning as reliable.",
}

if __name__ == "__main__":
    # ------------------------------------------------------------
    # CONFIG (later: make these command-line args)
    # ------------------------------------------------------------
    dataset = "har"     # ctu, emg, har
    top_k = 5           # 3, 5, 10
    temperature = 0.7
    embedding_model = "text-embedding-3-small"
    img_detail="high"
    out_file = f"./data/sample_generations/{dataset}/debator_judge.jsonl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # PATHS
    # input
    data_path = f"./data/samples/{dataset}"
    simple_features_path = f"./data/features/{dataset}/simple.jsonl"
    pair_wise_contrastive_path = (
        f"./data/features/{dataset}/10-imgs_5-rounds_contrastive.json"
    )
    # output
    md_dir = os.path.join(os.path.dirname(out_file), "prompt_examples")
    os.makedirs(md_dir, exist_ok=True)
    md_debator_path = os.path.join(md_dir, "debator.md")
    md_judge_path = os.path.join(md_dir, "judge.md")


    contrastive_summaries_path = (
        f"./data/features/{dataset}/contrastive_generation/final_summaries.json"
    )
    mappings_path = f"./data/samples/{dataset}/label_maps.json"
    top_k_path = f"data/sample_generations/{dataset}/sf_classification/top-{top_k}.jsonl"
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
    mappings["name_to_id"] = {v: k for k, v in mappings["id_to_name"].items()}
    top_k_generations = load_jsonl(top_k_path)

    # ------------------------------------------------------------
    # SETUP IMAGE PROMPTER (for future use)
    # ------------------------------------------------------------
    prompter = ImagePrompter()
    prompter.model_name = "gpt-4o-mini"
    debator_system_prompt = (
        "You are an expert debater and a time series expert. "
        "You will be given:\n"
        "  1. A time series visualization,\n"
        "  2. A class name to argue for,\n"
        "  3. A summary describing the characteristics of that class,\n"
        "  4. Other class names that are in contention,\n"
        "  5. An argument strength (1–4) indicating how confident you should be "
        "     when arguing that the time series belongs to the target class.\n\n"
        "Your task is to construct a persuasive argument that supports the class you are assigned. "
        "Use the provided summary and your time series expertise to highlight distinguishing features. "
        "You may reference competing classes when helpful. "
        "Your tone and level of certainty must match the provided argument strength.\n\n"
        "Your contribution is valuable and will later be used to help "
        f"{HELP_STRING[dataset.upper()]}."
    )
    judge_system_prompt = (
        "You are an expert judge. "
        "You will be given:"
        " 1. a time series visualization, "
        " 2. a list of possible classes that the visualization can belong to, "
        " 3. an argument for each class. "
        "Your task is to evaluate the arguments and determine which one is the most convincing. "
        "Consider the strength of the arguments, the evidence provided, and the overall persuasiveness. "
        "Provide a clear explanation for your decision."
        " Your contribution is valuable and will later be used to help "
        f"{HELP_STRING[dataset.upper()]}."
    )

    # ------------------------------------------------------------
    # EXPERIMENT #2
    # use contrastive summaries to decide class in top-k scenerio
    # ------------------------------------------------------------.

    saved_debator_prompt = False
    saved_judge_prompt = False
    for row in tqdm(top_k_generations, desc="Debator-Judge eval"):
        prompter.system_prompt = debator_system_prompt
        idx = row["idx"]
        gt = row["gt"]
        pred = row["pred"]
        neighbors = row["nn_labels"]
        neighbor_names = [mappings['id_to_name'][str(n)] for n in neighbors]
        name_counts = {name: 0 for name in set(neighbor_names)}
        for name in neighbor_names:
            name_counts[name] += 1

        # --- SKIP UNANIMOUS CASES ---
        if len(name_counts) == 1:
            unanimous_class = next(iter(name_counts.keys()))
            print(
                f"IDX {idx}: unanimous top-{top_k} classification → {unanimous_class}. "
                "Skipping debator + judge."
            )

            # log the unanimous result just like normal
            gt_name = mappings["id_to_name"][str(gt)]
            unanimous_class_id = mappings["name_to_id"][unanimous_class]

            line = {
                "idx": idx,
                "gt": gt,
                "gt_class_name": gt_name,
                "gt_letter": mappings["id_to_letter"][str(gt)],

                "prev_pred": unanimous_class_id,
                "prev_class_name": unanimous_class,
                "prev_letter": mappings["id_to_letter"][str(unanimous_class_id)],
                "prev_correct": int(unanimous_class == gt_name),

                # no judge prediction → keep it the same
                "pred": unanimous_class_id,
                "class_name": unanimous_class,
                "letter": mappings["id_to_letter"][str(unanimous_class_id)],
                "correct": int(unanimous_class == gt_name),

                "changed": 0,
                "change_label": "UNCHANGED",
                "change_direction": "none",

                "judge_output": None,
                "judge_reasoning": None,
                "arguments": {},
                "skipped_unanimous": True,
            }

            append_jsonl(out_file, line)
            continue   # 🔥 IMPORTANT: skip to next row


        arguments = {}
        for name, count in name_counts.items():
            summary = contrastive_summaries[name]
            assert count in ARGUMENT_STRENGTH_MAP, f"Count {count} not in ARGUMENT_STRENGTH_MAP"
            argument_strength_str = ARGUMENT_STRENGTH_MAP[count]

            user_kwargs = {
                    "instruction": (
                        "Argue that the given time series belongs to the ground truth class, "
                        "using the provided summary and your time series expertise."
                    ),
                    "your_class_to_argue": name,
                    "your_class_summary": summary,
                    "other_classes_in_contention": ", ".join(
                        [n for n in name_counts.keys() if n != name]
                    ),
                    "argument_strength": argument_strength_str,
                }
            prompt_kwargs = {"user": user_kwargs}
            img_path =plot_time_series(
                test[idx].X,
                method="line",
                save_path=f"./data/images/{dataset}/test/{idx}.png",
                title=TITLE_MAPPINGS[dataset.upper()],
                xlabs=X_MAPPINGS[dataset.upper()],
                ylabs=Y_MAPPINGS[dataset.upper()],
                legends=LEGEND_MAPPINGS[dataset.upper()],
                recreate=True
            )
            prompt_kwargs["img_path"] = img_path
            prompt_kwargs["img_detail"] = img_detail
            zs_prompt = Prompt(**prompt_kwargs)
            messages = prompter.format_prompt([], zs_prompt)

                
            # Save ONE example prompt in Markdown for inspection
            if not saved_debator_prompt:
                prompter.export_prompt_markdown(
                    examples=[],
                    query=zs_prompt,
                    out_md_path=md_debator_path,
                    save_images=False,
                )
                saved_debator_prompt = True
                print(f"Saved debator prompt for class {name} to {md_debator_path}")

            debator_output = prompter.get_completion(messages, temperature=temperature)
            debator_output = debator_output['content']
            arguments[name] = debator_output
            # print(f"Argument for class {name}:\n{debator_output}\n")
        # Judge step

        prompter.system_prompt = judge_system_prompt
        user_kwargs = {
            "question": (
                "Evaluate the arguments and determine which one is the most convincing. "
                "Consider the strength of the arguments, the evidence provided, and the overall persuasiveness."
                "Format your response as:\n"
                "`<REASONING>reasoning</REASONING>.\nThe most convincing argument is for class <CLASS_NAME>class_name></CLASS_NAME>`"
                "Be sure to delimit your reasoning and final class name decision with the specified tags."
                "For example:\n"
                "`<REASONING>After evaluating the arguments, I found that...</REASONING>\n"
                "The most convincing argument is for class <CLASS_NAME>Walking</CLASS_NAME>`"
            ),
            "arguments": arguments,
        }
        prompt_kwargs = {"user": user_kwargs}
        zs_prompt = Prompt(**prompt_kwargs)
        messages = prompter.format_prompt([], zs_prompt) 
        # Save ONE example prompt in Markdown for inspection
        if not saved_judge_prompt:
            prompter.export_prompt_markdown(
                examples=[],
                query=zs_prompt,
                out_md_path=md_judge_path,
                save_images=False,
            )
            saved_judge_prompt = True
            print(f"Saved judge prompt to {md_judge_path}")
        judge_output = prompter.get_completion(messages, temperature=temperature)
        judge_output = judge_output['content']
        reasoning = judge_output.split("<REASONING>")[1].split("</REASONING>")[0].strip()
        predicted_class = judge_output.split("<CLASS_NAME>")[1].split("</CLASS_NAME>")[0].strip()

        # FINAL EVALUATION LOGIC
        gt_name = mappings["id_to_name"][str(gt)]
        prev_class_id = pred
        prev_class_name = mappings["id_to_name"][str(prev_class_id)]

        correct = predicted_class == gt_name
        previous_correct = gt == prev_class_id

        correct_str = "CORRECT" if correct else "INCORRECT"
        previous_correct_str = "CORRECT" if previous_correct else "INCORRECT"

        changed = correct != previous_correct
        if changed:
            changed_direction = "improved" if correct else "got worse"
            print(
                f"IDX {idx} - {correct_str} | previously {previous_correct_str} "
                f"=> CHANGED ({changed_direction})"
            )
        else:
            print(f"IDX {idx} - {correct_str} | previously {previous_correct_str} => UNCHANGED")

        print(f"Ground truth: {gt_name}, Predicted: {predicted_class}")

        class_id = mappings["name_to_id"][predicted_class]

        line = {
            "idx": idx,
            # ground truth
            "gt": gt,
            "gt_class_name": gt_name,
            "gt_letter": mappings["id_to_letter"][str(gt)],
            # previous (baseline / top-k) prediction
            "prev_pred": prev_class_id,
            "prev_class_name": prev_class_name,
            "prev_letter": mappings["id_to_letter"][str(prev_class_id)],
            "prev_correct": int(previous_correct),
            # new (judge-based) prediction
            "pred": class_id,
            "class_name": predicted_class,
            "letter": mappings["id_to_letter"][str(class_id)],
            "correct": int(correct),
            # change diagnostics
            "changed": int(changed),  # 1 if changed, 0 if unchanged
            "change_label": "CHANGED" if changed else "UNCHANGED",
            "change_direction": (
                "improved" if changed and correct
                else "worse" if changed and not correct
                else "none"
            ),
            # judge + debator info
            "judge_output": judge_output,
            "judge_reasoning": reasoning,  # assuming you parsed this earlier
            "arguments": arguments,
        }

        append_jsonl(out_file, line)


    # print("\n=== TEST 9: Export zero-shot prompt to Markdown ===")
    # zero_shot_query = Prompt(
    #     user={"question": default_question, "context": default_context},
    #     img_path="./demo/images/query.jpg",
    # )
    # zero_shot_md_path = "./prompt_exports/test9_zero_shot.md"
    # zero_shot_md_abs = prompter.export_prompt_markdown(
    #     examples=[],  # no few-shot examples
    #     query=zero_shot_query,
    #     out_md_path=zero_shot_md_path,
    #     save_images=True,
    #     images_dirname="images",
    # )
    # print(f"TEST 9 zero-shot prompt exported to: {zero_shot_md_abs}")   




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
