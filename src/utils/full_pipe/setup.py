import argparse

from dotenv import load_dotenv

from utils.file_io import load_json
from utils.image_prompter import ImagePrompter
from utils.loaders import Split, load_train_test
from utils.loggers import MasterLogger

def parse_pipe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-shot visual + text prompting for time-series classification."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., har, emg, ctu).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Name of the chat/vision model to use.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs/blah.log",
        help="Path to save logs.",
    )
    parser.add_argument(
        "--print_to_console",
        default=1,
        type=int,
        choices=[0, 1],
        help="Whether to print logs to console.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of queries to send in one batch to the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--load_summary",
        type=int,
        choices=[0,1],
        default=0,
        help="Should we load the summaries or (re)generate them?")
    parser.add_argument(
        "--summary_type",
        type=str,
        choices=["visual", "ts", "both"],
        default="visual",
        help="What type of data should be used for summarization?"
    )
    return parser.parse_args()



def setup(
    args: argparse.Namespace,
) -> tuple[ImagePrompter, MasterLogger, Split, Split, dict]:
    """
    Sets up the prompter, logger, and loads train/test data.

    Returns
    -------
    (prompter, logger, train, test, mappings)
    """
    # Ensure env is loaded (ImagePrompter also does this internally, but this is harmless)
    load_dotenv("./resources/.env")

    # Prompter (this will load OPENAI_API_KEY from ./resources/.env)
    prompter = ImagePrompter()
    prompter.model_name = args.model_name

    # Logger
    logger = MasterLogger(
        log_path=args.log_path,
        init=True,
        clear=True,
        print_to_console=bool(args.print_to_console),
    )

    # Data
    train, test = load_train_test(
        f"./data/samples/{args.dataset}",
        n_shots=0,
    )

    # Label mappings (id→name, id→letter, etc.)
    mappings = load_json(f"./data/samples/{args.dataset}/label_maps.json")

    return prompter, logger, train, test, mappings

