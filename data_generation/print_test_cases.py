import os
import pickle
from pathlib import Path
from typing import Any, Iterable

from pprint import pformat


def load_pkl(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"PKL file not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def summarize_dict(d: dict, max_items: int) -> str:
    lines = ["{",]
    for idx, (key, val) in enumerate(d.items()):
        if idx >= max_items:
            lines.append(f"  ... (showing first {max_items} keys out of {len(d)})")
            break
        lines.append(f"  {key!r}: {pformat(val, compact=True)}")
    lines.append("}")
    return "\n".join(lines)


def summarize_iterable(items: Iterable, max_items: int) -> str:
    lines = ["["]
    for idx, elem in enumerate(items):
        if idx >= max_items:
            lines.append(f"  ... (showing first {max_items} entries)")
            break
        lines.append(f"  {pformat(elem, compact=True)}")
    lines.append("]")
    return "\n".join(lines)


def create_summary(data: Any, max_items: int) -> str:
    lines = ["========== PKL FILE SUMMARY ==========\n"]
    lines.append(f"Type: {type(data)}")
    if isinstance(data, dict):
        lines.append(f"Number of keys: {len(data)}")
        lines.append(f"Keys: {str(data.keys())}")
        lines.append("\n")
        lines.append(summarize_dict(data, max_items))
    elif isinstance(data, (list, tuple, set)):
        lines.append(f"Number of elements: {len(data)}")
        lines.append(summarize_iterable(data, max_items))
    else:
        lines.append(pformat(data, compact=True))
    lines.append("\n======================================")
    return "\n".join(lines)


def save_summary(text: str, output_path: Path) -> None:
    output_path.write_text(text, encoding="utf-8")
    print(f"Summary saved to: {output_path}")


def main() -> None:
    pkl_path = Path("./test_cases.pkl")
    data = load_pkl(pkl_path)
    summary = create_summary(data, 200)
    output_path = Path(f"{os.path.splitext(os.path.basename(pkl_path))[0]}_print.txt")
    save_summary(summary, output_path)


if __name__ == "__main__":
    main()
