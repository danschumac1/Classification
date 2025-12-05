from typing import Any
from utils.file_io import append_jsonl

def load_summary() -> {int, str}:
    pass

def save_results(output_path:str, batch_results:list[dict[str, Any]]):
    for result in batch_results:
        line = {...}
        append_jsonl(output_path, line)