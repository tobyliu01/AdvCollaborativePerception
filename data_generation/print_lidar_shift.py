import pickle
import argparse
import os
import json
import numpy as np

def load_pkl_file(pkl_path):
    """
    Load and return a Python object from a .pkl file.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def summarize_pkl(data):
    """
    Create a structured summary string for printing and saving.
    - Each scenario is printed on a separate line as a JSON-like object.
    """
    lines = []
    lines.append("========== PKL FILE SUMMARY ==========\n")

    # Basic type info
    lines.append(f"Data Type: {type(data)}")

    if isinstance(data, dict):
        lines.append(f"Keys: {list(data.keys())}")
        lines.append("\n")

        # # Print scenario list
        # scenarios = data.get("scenario", None)
        # if isinstance(scenarios, list):
        #     lines.append("Scenarios:")
        #     for sc in scenarios:
        #         try:
        #             lines.append(json.dumps(sc, ensure_ascii=False))
        #             lines.append(f"Total {len(sc.get("frame_ids"))} frames")
        #         except Exception:
        #             lines.append(str(sc))
        # lines.append("\n")

        # # Print multi-frame
        # multi_frames = data.get("multi_frame", None)
        # if isinstance(multi_frames, list):
        #     lines.append("Multi frames:")
        #     index = 0
        #     for mf in multi_frames:
        #         try:
        #             lines.append(f"{index}: {json.dumps(mf, ensure_ascii=False)}")
        #             lines.append(f"Total {len(mf.get("frame_ids"))} frames")
        #             index += 1
        #         except Exception:
        #             lines.append(str(mf))

        # # LiDAR Data (only shapes to avoid flooding output)
        # if "lidar_map" in data and isinstance(data["lidar_map"], np.ndarray):
        #     lines.append(f"LiDAR Map Shape: {data['lidar_map'].shape}")
        # if "lidar_sensor" in data and isinstance(data["lidar_sensor"], np.ndarray):
        #     lines.append(f"LiDAR Sensor Shape: {data['lidar_sensor'].shape}")

        # # BBox Data
        # if "pred_bboxes" in data and isinstance(data["pred_bboxes"], np.ndarray):
        #     lines.append(f"Number of BBoxes: {len(data['pred_bboxes'])}")
        #     if len(data["pred_bboxes"]) > 0:
        #         lines.append(f"First BBox: {data['pred_bboxes'][0].tolist() if hasattr(data['pred_bboxes'][0], 'tolist') else data['pred_bboxes'][0]}")
        # if "pred_scores" in data and data["pred_scores"] is not None:
        #     try:
        #         lines.append(f"Scores Available: Yes ({len(data['pred_scores'])})")
        #     except Exception:
        #         lines.append("Scores Available: Yes")

    elif isinstance(data, list):
        i = 0
        for line in data:
            # lines.append(str(line))
            lines.append(str(i) + ":")
            lines.append("attack_opts:")
            lines.append(str(line.get("attack_opts"))+'\n')
            lines.append("attack_meta:")
            lines.append(str(line.get("attack_meta")))
            lines.append("\n")
            i += 1
    
    else:
        lines.append("The loaded object is not a dictionary. Cannot parse keys.")

    lines.append("\n======================================")
    return "\n".join(lines)

def save_summary(text, output_path):
    """
    Save the summary text to a .txt file.
    """
    with open(output_path, 'w', encoding="utf-8") as f:
        f.write(text)
    print(f"Summary saved to: {output_path}")

def main():
    pkl_path = "./lidar_shift_new.pkl"

    # Load PKL
    data = load_pkl_file(pkl_path)

    # Summarize content
    summary = summarize_pkl(data)

    # Determine output path
    out_path = f"{os.path.splitext(os.path.basename(pkl_path))[0]}_print.txt"

    # Save to text file
    save_summary(summary, out_path)

if __name__ == "__main__":
    main()
