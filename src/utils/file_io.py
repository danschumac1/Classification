import json
import os
from typing import List
# from utils.logging import MasterLogger


def append_jsonl(output_path: str, data: dict):
    """
    Append a dictionary to the specified output JSONL file.
    Creates parent directories if needed.
    """
    output_file = output_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(data, f)
        f.write('\n')
    
    # logger = MasterLogger.get_instance()

def load_jsonl(file_path):
    """Load a JSON Lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json(file_path:str) -> dict | list:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def ensure_header(results_path: str, header_cols: List[str]):
    # create parent dir
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    write_header = (not os.path.exists(results_path)) or (os.path.getsize(results_path) == 0)
    if write_header:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("\t".join(header_cols) + "\n")
            
def append_row(results_path: str, row_vals: List[str]):
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\t".join(map(str, row_vals)) + "\n")

def save_json(results_path: str, data_dict: dict):
    with open(results_path, "w", encoding="utf-8") as fo:
        json.dump(data_dict, fo, indent=2, ensure_ascii=False)
