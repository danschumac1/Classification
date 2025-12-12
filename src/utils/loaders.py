from dataclasses import dataclass
import json
import os
import textwrap
from typing import Dict, Optional, Sequence, Tuple, Union, Any, List
import warnings
import numpy as np


@dataclass
class Split:
    """
    Container for a single dataset split (train or test).

    Attributes
    ----------
    X : np.ndarray
        Time-series data with shape (N, T, D) or similar.
    y : np.ndarray
        Integer labels of shape (N,).
    idx : np.ndarray
        Original row indices (0..N-1) for traceability.
    shot_idxs : Optional[np.ndarray]
        Optional 1D array of indices used for few-shot examples
        (relative to this split's indexing).
    fixed_shot_idxs : Optional[np.ndarray]
        Legacy field for compatibility (not used in new pipeline).
    label_maps : Optional[Dict]
        Metadata mapping label ids to letters/names.
    general_question : Optional[str]
        Natural-language task description for this dataset.
    dataset : Optional[str]
        Name of the dataset (e.g., "har", "emg").
    """

    X: np.ndarray
    y: np.ndarray
    idx: np.ndarray
    shot_idxs: Optional[np.ndarray] = None
    label_maps: Optional[Dict] = None
    general_question: Optional[str] = None
    dataset: Optional[str] = None

    @property
    def unique_classes(self) -> np.ndarray:
        return np.unique(self.y)

    @property
    def n_classes(self) -> int:
        return int(self.unique_classes.size)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def class_dist(self) -> Dict[int, int]:
        labels, counts = np.unique(self.y, return_counts=True)
        return {int(lbl): int(cnt) for lbl, cnt in zip(labels, counts)}

    def __repr__(self) -> str:
        """
        Detailed, structured summary of the Split object with
        pretty label_maps and wrapped general_question.
        """

        def _indent(s: str, n: int = 4) -> str:
            pad = " " * n
            return "\n".join(pad + line if line else pad for line in s.splitlines())

        def _fmt_dict(d: Dict, max_items: int = 10) -> str:
            items = list(d.items())
            if len(items) <= max_items:
                return str(d)
            head = ", ".join(f"{k}: {v}" for k, v in items[:max_items])
            return f"{{{head}, ... ({len(items)} total)}}"

        def _wrap_block(s: Optional[str], width: int = 88, max_lines: int = 12) -> str:
            s = (s or "").strip()
            if not s:
                return "none"
            wrapped = textwrap.fill(s, width=width)
            lines = wrapped.splitlines()
            if len(lines) > max_lines:
                lines = lines[:max_lines] + ["..."]
            return "\n" + _indent("\n".join(lines), 4)

        def _label_maps_table(maps: Optional[Dict], max_rows: int = 20) -> str:
            if not isinstance(maps, dict):
                return "none"

            id2name = maps.get("id_to_name")
            id2letter = maps.get("id_to_letter")
            if not isinstance(id2name, dict) or not isinstance(id2letter, dict):
                pretty = json.dumps(maps, indent=4, sort_keys=True)
                return "\n" + _indent(pretty, 4)

            try:
                id2name_i = {int(k): str(v) for k, v in id2name.items()}
                id2letter_i = {int(k): str(v) for k, v in id2letter.items()}
            except Exception:
                pretty = json.dumps(maps, indent=4, sort_keys=True)
                return "\n" + _indent(pretty, 4)

            rows: List[Tuple[int, str, str]] = []
            for k in sorted(id2name_i.keys()):
                rows.append((k, id2letter_i.get(k, "?"), id2name_i.get(k, str(k))))

            total = len(rows)
            if total == 0:
                return "none"

            if total > max_rows:
                head = rows[:max_rows]
                tail_note = f"... ({total - max_rows} more)"
            else:
                head = rows
                tail_note = ""

            col1, col2, col3 = "id", "letter", "name"
            w1 = max(len(col1), max(len(str(r[0])) for r in head))
            w2 = max(len(col2), max(len(str(r[1])) for r in head))
            w3 = max(len(col3), max(len(str(r[2])) for r in head))

            header = f"{col1:<{w1}} | {col2:<{w2}} | {col3:<{w3}}"
            sep = "-" * len(header)
            body_lines = [f"{r[0]:<{w1}} | {r[1]:<{w2}} | {r[2]:<{w3}}" for r in head]
            if tail_note:
                body_lines.append(tail_note)

            table = "\n".join([header, sep, *body_lines])
            return "\n" + _indent(table, 4)

        num_examples = len(self)
        shape_str = "×".join(map(str, self.X.shape))
        classes = self.unique_classes
        cls_dist = self.class_dist()
        dist_str = _fmt_dict(cls_dist, max_items=10)

        label_maps_block = _label_maps_table(self.label_maps, max_rows=20)
        general_question_block = _wrap_block(self.general_question, width=88, max_lines=12)

        lines: List[str] = []
        lines.append("Split(")
        lines.append(f"  dataset          = {self.dataset!r},")
        lines.append(f"  N                = {num_examples},")
        lines.append(f"  X.shape          = ({shape_str}),")
        lines.append(f"  y.shape          = {tuple(self.y.shape)},")
        lines.append(f"  n_classes        = {self.n_classes},")
        lines.append(f"  classes          = {classes.tolist()},")
        lines.append(f"  class_dist       = {dist_str},")
        lines.append(f"  label_maps       ={label_maps_block},")
        lines.append(f"  general_question ={general_question_block},")
        lines.append(")")
        return "\n".join(lines)

    def _take(self, idxs: np.ndarray) -> "Split":
        """Return a new Split containing rows at idxs (0-based)."""
        idxs = np.asarray(idxs, dtype=int)
        new_X = self.X[idxs]
        new_y = self.y[idxs]
        new_idx = self.idx[idxs] if self.idx is not None else idxs.copy()

        # Remap shot indices if present and stored as a flat ndarray
        new_shot: Optional[np.ndarray] = None
        if isinstance(self.shot_idxs, np.ndarray):
            mask = np.isin(self.shot_idxs, idxs)
            kept = self.shot_idxs[mask]
            if kept.size > 0:
                remap = {int(old): i for i, old in enumerate(idxs.tolist())}
                new_shot = np.array(
                    [remap[int(s)] for s in kept if int(s) in remap],
                    dtype=int,
                )
            else:
                new_shot = np.empty((0,), dtype=int)

        return Split(
            X=new_X,
            y=new_y,
            idx=new_idx,
            shot_idxs=new_shot,
            label_maps=self.label_maps,
            general_question=self.general_question,
            dataset=self.dataset,
        )

    def __getitem__(self, key: Union[int, slice, Sequence[int], np.ndarray]) -> "Split":
        """
        Support int, slice, list/ndarray of indices.
        Always returns a Split (so batch ops are uniform).
        """
        if isinstance(key, slice):
            idxs = np.arange(len(self))[key]
        elif isinstance(key, (list, tuple, np.ndarray)):
            idxs = np.asarray(key, dtype=int)
        else:  # int / np.integer
            idxs = np.asarray([int(key)], dtype=int)
        return self._take(idxs)

# ======================================================================
#  I/O HELPERS FOR NEW FORMAT
# ======================================================================

def _load_npz_split(root: str, split_name: str, mmap: bool = False, normalize: bool = False) -> Split:
    """
    Load a single split ('train' or 'test') from a directory containing
    train.npz / test.npz as created by clean_data.py.

    Parameters
    ----------
    root : str
        Folder path containing {train,test}.npz.
    split_name : str
        Either "train" or "test".
    mmap : bool
        Whether to memory-map the npz arrays (read-only).
    normalize : bool
        If True, z-normalize each time series along the last axis.

    Returns
    -------
    Split
    """
    npz_path = os.path.join(root, f"{split_name}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {split_name}.npz in {root}")

    mmap_mode = "r" if mmap else None
    data = np.load(npz_path, mmap_mode=mmap_mode)

    if split_name == "train":
        X = data["X_train"]
        y = data["y_train"]
    elif split_name == "test":
        X = data["X_test"]
        y = data["y_test"]
    else:
        raise ValueError(f"split_name must be 'train' or 'test', got {split_name!r}")


    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    if y.dtype != np.int64:
        y = y.astype(np.int64, copy=False)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got shape {y.shape} for split={split_name}")


    if normalize:
        # Decide which axis is "time"
        if X.ndim == 2:
            # (N, T)
            time_axis = 1
        elif X.ndim == 3:
            # assume (N, T, D): time is axis=1, channels last
            # if your convention is different, flip this
            time_axis = 1
        else:
            raise ValueError(f"Expected X.ndim in {{2,3}}, got {X.ndim} for split={split_name}")

        # z-normalize along the time axis; protect against divide-by-zero
        mean = np.mean(X, axis=time_axis, keepdims=True)
        std = np.std(X, axis=time_axis, keepdims=True) + 1e-6

        X = (X - mean) / std

        # sanity check: any *entire* series collapsed to zero?
        # collapse across time + channels for the check
        if X.ndim == 2:
            all_zero = np.all(X == 0, axis=1)
        else:  # ndim == 3
            all_zero = np.all(X == 0, axis=(1, 2))

        if np.any(all_zero):
            raise ValueError(
                f"[NORMALIZE] Found all-zero time series in split={split_name} after normalization."
            )

    idx = np.arange(len(X), dtype=np.int64)
    return Split(X=X, y=y, idx=idx)


def _load_artifacts_new(root: str) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Load class_shots.json, label_maps.json, and general_question.txt
    from the given directory, if present.

    Returns
    -------
    class_shots : Optional[Dict[int, np.ndarray]]
        Mapping label -> 1D np.ndarray of indices (relative to TRAIN split).
    label_maps : Optional[Dict[str, Any]]
        Raw JSON dict with label metadata.
    general_question : Optional[str]
        Task description string.
    """
    class_shots: Optional[Dict[int, np.ndarray]] = None
    label_maps: Optional[Dict[str, Any]] = None
    general_question: Optional[str] = None

    # class_shots.json
    class_shots_path = os.path.join(root, "class_shots.json")
    if os.path.isfile(class_shots_path):
        with open(class_shots_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Keys may be strings from JSON; normalize to int -> np.ndarray
        class_shots = {
            int(lbl): np.asarray(indices, dtype=int)
            for lbl, indices in raw.items()
        }
    else:
        warnings.warn(f"[ARTIFACT] Missing class_shots.json at {class_shots_path}; few-shot info will be None.")

    # label_maps.json
    label_maps_path = os.path.join(root, "label_maps.json")
    if os.path.isfile(label_maps_path):
        with open(label_maps_path, "r", encoding="utf-8") as f:
            try:
                label_maps = json.load(f)
            except json.JSONDecodeError as exc:
                warnings.warn(f"[ARTIFACT] Could not parse label_maps.json: {exc}")
                label_maps = None
    else:
        warnings.warn(f"[ARTIFACT] Missing label_maps.json at {label_maps_path}")

    # general_question.txt
    question_path = os.path.join(root, "general_question.txt")
    if os.path.isfile(question_path):
        with open(question_path, "r", encoding="utf-8") as f:
            general_question = f.read().strip()
    else:
        warnings.warn(f"[ARTIFACT] Missing general_question.txt at {question_path}")

    return class_shots, label_maps, general_question


# ======================================================================
#  PUBLIC API
# ======================================================================

def load_train_test(
    input_folder: str,
    n_shots: int,
    mmap: bool = False,
    attach_artifacts: bool = True,
    normalize: bool = False,
) -> Tuple[Split, Split]:
    """
    Load train/test splits from <input_folder> in the NEW format.

    Expects the following files in input_folder:
        - train.npz  (X_train, y_train)
        - test.npz   (X_test,  y_test)
        - class_shots.json       [optional but expected for cleaned data]
        - label_maps.json        [optional]
        - general_question.txt   [optional]

    Parameters
    ----------
    input_folder : str
        Directory containing prepared splits (e.g., data/datasets/har).
    n_shots : int
        Number of shots to load per class. Example:
            - if 0, no shots are loaded (shot_idxs = None)
            - if 2 and there are 3 classes, up to 6 total shot indices
              (2 per class, if available).
    mmap : bool
        If True, use memory mapping for npz arrays (read-only).
    attach_artifacts : bool
        If True, load class_shots, label_maps, and general_question and
        attach them to the returned Split objects.
    normalize : bool
        If True, z-normalize each time series along the last axis.

    Returns
    -------
    train_split : Split
    test_split : Split
    """
    # Load core arrays
    train_split = _load_npz_split(input_folder, "train", mmap=mmap, normalize=normalize)
    test_split = _load_npz_split(input_folder, "test", mmap=mmap, normalize=normalize)

    dataset_name = os.path.basename(os.path.normpath(input_folder))
    train_split.dataset = dataset_name
    test_split.dataset = dataset_name

    # Default: no shots unless we attach artifacts
    train_split.shot_idxs = None
    test_split.shot_idxs = None

    if not attach_artifacts:
        return train_split, test_split

    # Load artifacts (class_shots, label_maps, general_question)
    class_shots, label_maps, general_question = _load_artifacts_new(input_folder)

    # Attach label maps & question to both splits
    train_split.label_maps = label_maps
    test_split.label_maps = label_maps
    train_split.general_question = general_question
    test_split.general_question = general_question

    # ------------------------------------------------------------------
    # Build an INTERLEAVED flat array of few-shot indices for TRAIN
    # ------------------------------------------------------------------
    flat_shots: Optional[np.ndarray] = None

    if n_shots > 0 and class_shots is not None and len(class_shots) > 0:
        # Normalize to per-class lists of candidate indices
        # class_shots: Dict[int, np.ndarray], indices relative to TRAIN
        sorted_labels = sorted(class_shots.keys())

        per_class_lists: List[List[int]] = []
        for lbl in sorted_labels:
            arr = np.asarray(class_shots[lbl], dtype=int)
            if arr.size == 0:
                per_class_lists.append([])
                continue

            # take up to n_shots for this class
            if arr.size > n_shots:
                selected = arr[:n_shots]
            else:
                selected = arr

            per_class_lists.append(selected.tolist())

        # Round-robin interleave: A,B,A,B,... rather than AAA...BBB...
        interleaved: List[int] = []
        # maximum length across classes (n_shots or fewer)
        max_len = max((len(lst) for lst in per_class_lists), default=0)

        for i in range(max_len):
            for class_idx, lst in enumerate(per_class_lists):
                if i < len(lst):
                    interleaved.append(lst[i])

        if len(interleaved) > 0:
            # Make sure we have a 1D ndarray; do NOT sort,
            # to preserve the interleaved order.
            flat_shots = np.asarray(interleaved, dtype=int)
        else:
            flat_shots = None

    train_split.shot_idxs = flat_shots
    test_split.shot_idxs = None  # test does not need few-shot indices

    # Optional sanity checks if we actually had class_shots
    if class_shots is not None and len(class_shots) > 0:
        n_classes_from_shots = len(set(class_shots.keys()))
        assert len(train_split.unique_classes) == n_classes_from_shots, \
            "[LOAD] Mismatch between class_shots and train classes."
        assert len(test_split.unique_classes) == n_classes_from_shots, \
            "[LOAD] Mismatch between class_shots and test classes."

    return train_split, test_split