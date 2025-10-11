"""
Run OpenCOOD perception on original LiDAR for a single selected vehicle
and save+visualize the result.

Usage (examples):
    python run_opencood_on_single_vehicle.py --case_id 0 --frame_id 9 --vehicle_id 3
    python run_opencood_on_single_vehicle.py --fusion early --model pointpillar

If vehicle_id is omitted, the script will pick the first vehicle id found
in the selected frame.
"""
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ensure repo imports like evaluate.py/opencood_perception.py work
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)

from mvp.config import data_root, model_root
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.data.util import pcd_sensor_to_map, pcd_map_to_sensor

def draw_bbox_topdown(ax, center, l, w, yaw, edgecolor='r', linewidth=1.5, alpha=0.9):
    """
    Draw oriented rectangle on axes ax (top-down: x->east, y->north).
    center: (x,y)
    l: length (along vehicle heading), w: width (perpendicular)
    yaw: in radians (OpenCOOD uses yaw in radians)
    """
    cx, cy = center
    # corners in vehicle frame (centered)
    # length along x, width along y
    half_l = l / 2.0
    half_w = w / 2.0
    corners = np.array([
        [ half_l,  half_w],
        [ half_l, -half_w],
        [-half_l, -half_w],
        [-half_l,  half_w],
    ])  # (4,2)
    # rotation
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw),  np.cos(yaw)]])
    rotated = corners.dot(R.T)
    rotated[:,0] += cx
    rotated[:,1] += cy
    # close loop
    poly = np.vstack([rotated, rotated[0]])
    ax.plot(poly[:,0], poly[:,1], '-', color=edgecolor, linewidth=linewidth, alpha=alpha)

def topdown_plot(pcd_map, pred_bboxes, pred_scores, save_path=None, title=None):
    """
    pcd_map: (N, >=3) numpy array in map coords
    pred_bboxes: (M,7) format from Opencood after conversion to center format
                 assumed order: [x, y, z, l, w, h, yaw] (this matches evaluate/opencood processing)
    pred_scores: (M,) numpy
    """
    fig, ax = plt.subplots(figsize=(8,8))
    if pcd_map is not None and pcd_map.shape[0] > 0:
        ax.scatter(pcd_map[:,0], pcd_map[:,1], s=0.5, alpha=0.6)
    if pred_bboxes is not None and pred_bboxes.size != 0:
        for i, bb in enumerate(pred_bboxes):
            x, y, z, l, w, h, yaw = bb
            score = float(pred_scores[i]) if pred_scores is not None and i < len(pred_scores) else None
            draw_bbox_topdown(ax, (x,y), l, w, yaw, edgecolor='r', linewidth=1.2)
            if score is not None:
                ax.text(x, y, "{:.2f}".format(score), color='red', fontsize=8, ha='center', va='center')
    ax.set_xlabel("x (map)")
    ax.set_ylabel("y (map)")
    ax.set_aspect('equal', 'box')
    if title:
        ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=0, help="scenario id to use")
    parser.add_argument("--frame", type=int, default=9, help="frame index inside the case")
    parser.add_argument("--vehicle", type=int, default=None, help="vehicle id (ego id) to run perception on; if omitted script picks first vehicle present")
    parser.add_argument("--fusion", type=str, default="early", choices=["early","intermediate","late"], help="fusion method")
    parser.add_argument("--model", type=str, default="pointpillar", choices=["pixor","voxelnet","second","pointpillar","v2vnet","fpvrcnn"], help="OpenCOOD model name")
    parser.add_argument("--save_dir", type=str, default="./result/single_vehicle", help="where to save pickles/figures")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # build dataset and perception (same pattern as evaluate.py / OpencoodPerception)
    dataset = OPV2VDataset(root_path=os.path.join(data_root, "OPV2V"), mode="test")
    perception = OpencoodPerception(fusion_method=args.fusion, model_name=args.model)

    # load the case (multi_vehicle_case)
    case = dataset.get_case(args.scenario, tag="multi_frame", use_lidar=True, use_camera=False)
    if case is None:
        raise RuntimeError("Case id {} not found".format(args.scenario))

    frame = case[args.frame]
    vehicle_keys = list(frame.keys())
    if len(vehicle_keys) == 0:
        raise RuntimeError("No vehicles in case {} frame {}".format(args.scenario, args.frame))

    ego_id = args.vehicle if args.vehicle is not None else vehicle_keys[0]
    if ego_id not in frame:
        raise RuntimeError("vehicle_id {} not present in case {} frame {}. Available ids: {}".format(ego_id, args.scenario, args.frame, vehicle_keys))

    print("Using case {}, frame {}, ego vehicle id {}".format(args.scenario, args.frame, ego_id))

    # run perception on the original multi-vehicle case (no attack)
    pred_bboxes, pred_scores = perception.run(case[args.frame], ego_id=ego_id)

    # transform LiDAR to map frame for plotting
    # case[frame_id][vehicle]['lidar'] is in sensor frame; use pcd_sensor_to_map
    lidar = frame[ego_id]["lidar"]
    lidar_pose = frame[ego_id]["lidar_pose"]
    pcd_map = pcd_sensor_to_map(lidar, lidar_pose)

    # Save a pickle containing everything relevant (see description below)
    save_data = {
        "case_id": args.scenario,
        "frame_id": args.frame,
        "ego_id": ego_id,
        "lidar_pose": lidar_pose,
        "lidar_sensor": lidar,     # original sensor-frame points
        "lidar_map": pcd_map,      # transformed to map coordinates (x,y,z)
        "pred_bboxes": pred_bboxes, # Nx7 numpy array (center format after opencood conversions)
        "pred_scores": pred_scores, # N numpy
        "fusion": args.fusion,
        "model": args.model,
    }
    save_file = os.path.join(args.save_dir, "case_{:06d}_frame_{:02d}_ego_{}.pkl".format(args.scenario, args.frame, ego_id))
    with open(save_file, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved perception result to:", save_file)

    # quick visualization (top-down)
    fig_file = save_file.replace(".pkl", ".png")
    title = "case {} frame {} ego {}".format(args.scenario, args.frame, ego_id)
    topdown_plot(pcd_map, pred_bboxes, pred_scores, save_path=fig_file, title=title)
    print("Saved figure to:", fig_file)

if __name__ == "__main__":
    main()
