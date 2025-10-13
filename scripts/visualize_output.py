import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def rotation_matrix(roll, yaw, pitch):
    R = np.array([[np.cos(yaw)*np.cos(pitch),
                   np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll),
                   np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch),
                   np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll),
                   np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch),
                   np.cos(pitch)*np.sin(roll),
                   np.cos(pitch)*np.cos(roll)]])
    return R


def compute_xy_limits(points_xyz: np.ndarray, margin_ratio: float = 0.05):
    if points_xyz.shape[0] == 0:
        return (-10, 10), (-10, 10)
    x = points_xyz[:, 0]; y = points_xyz[:, 1]
    x_lo, x_hi = np.percentile(x, [0.5, 99.5])
    y_lo, y_hi = np.percentile(y, [0.5, 99.5])
    xm = (x_hi - x_lo) * margin_ratio
    ym = (y_hi - y_lo) * margin_ratio
    return (x_lo - xm, x_hi + xm), (y_lo - ym, y_hi + ym)


def box2d_corners_xy(center_xy, extent_lwh, yaw_deg):
    # Use full lengths as in the original bbox drawing: size = extent*2
    l, w = extent_lwh[0]*2.0, extent_lwh[1]*2.0
    cx, cy = center_xy
    yaw = np.deg2rad(yaw_deg)
    # rectangle corners in local frame (centered)
    dx = l/2.0; dy = w/2.0
    local = np.array([[ dx,  dy],
                      [ dx, -dy],
                      [-dx, -dy],
                      [-dx,  dy]])  # (4,2)
    R = np.array([[ np.cos(yaw), -np.sin(yaw)],
                  [ np.sin(yaw),  np.cos(yaw)]])
    world = (R @ local.T).T + np.array([cx, cy])
    return world


def draw_topdown_matplotlib(pointclouds_xyz,
                            labels,
                            out_path,
                            mark_poses=None,
                            pose_id_labels=None,
                            dpi=200):
    """
    pose_id_labels: optional list of (x, y, 'id-string') to annotate near pose markers.
    """
    pts = np.vstack(pointclouds_xyz) if len(pointclouds_xyz) else np.empty((0,3))
    xlim, ylim = compute_xy_limits(pts)

    fig, ax = plt.subplots(figsize=(12, 10))
    if pts.shape[0] > 0:
        ax.scatter(pts[:,0], pts[:,1], s=0.1, c="black", linewidths=0)

    # draw 2D boxes projected on XY
    for _, label in labels.items():
        center_xy = np.array(label["location"][:2])
        extent    = np.array(label["extent"])  # half sizes
        yaw_deg   = float(label["angle"][1])   # yaw in degrees
        corners = box2d_corners_xy(center_xy, extent, yaw_deg)
        poly = np.vstack([corners, corners[0]])  # close loop
        ax.plot(poly[:,0], poly[:,1], color="red", linewidth=1.2)

    # pose markers
    if mark_poses:
        mark_poses = np.asarray(mark_poses).reshape(-1,2)
        ax.scatter(mark_poses[:,0], mark_poses[:,1], s=40, c="tab:blue", marker="x")

    # id labels near poses
    if pose_id_labels:
        for (x, y, txt) in pose_id_labels:
            # slight offset so text doesn't sit exactly on the marker
            ax.text(x + 0.6, y + 0.6, str(txt),
                    fontsize=9, color="tab:blue",
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(args):
    if args.dataset not in ["train", "validate", "test"]:
        raise Exception("Wrong dataset")

    pkl_path = os.path.join(args.datadir, f"{args.dataset}.pkl")
    with open(pkl_path, 'rb') as f:
        meta = pickle.load(f)

    scenario_key = str(args.scenario)
    if scenario_key not in meta:
        some = list(meta.keys())[:10]
        raise Exception(f"Wrong scenario: {scenario_key}. Examples: {some}")

    # create output directory
    out_dir = f"output/{scenario_key}"
    os.makedirs(out_dir, exist_ok=True)

    # decide which frames to process
    if args.frame is None or args.frame < 0:
        frame_ids = sorted(meta[scenario_key]["data"].keys())
        print("Total frames:", len(frame_ids))
    else:
        if args.frame not in meta[scenario_key]["data"]:
            available = sorted(list(meta[scenario_key]["data"].keys()))
            raise Exception(f"Wrong frame: {args.frame}. Available: {available[:20]}{' ...' if len(available)>20 else ''}")
        frame_ids = [args.frame]

    # loop over frames
    for fid in frame_ids:
        frame_rec = meta[scenario_key]["data"][fid]
        labels = meta[scenario_key]["label"][fid]

        clouds_xyz = []
        poses_xy = []
        pose_id_labels = []

        for lidar_id, rec in frame_rec.items():
            # PCD path
            pcd_path = rec.get("lidar")
            if pcd_path is None:
                continue
            if not os.path.isabs(pcd_path):
                pcd_path = os.path.join(args.datadir, pcd_path)
            if not os.path.exists(pcd_path):
                print(f"[WARN] missing PCD: {pcd_path}")
                continue

            pcd = o3d.io.read_point_cloud(pcd_path)
            pts = np.asarray(pcd.points)
            if pts.size == 0:
                continue

            calib_entry = rec.get("calib")
            if isinstance(calib_entry, dict):
                calib = calib_entry
            else:
                raise TypeError(f"Unsupported calib entry type for lidar_id {lidar_id}: {type(calib_entry)}")

            lidar_pose = calib.get("lidar_pose", rec.get("lidar_pose"))
            if lidar_pose is None:
                continue
            R = rotation_matrix(*(np.array(lidar_pose[3:]) * np.pi / 180.0))
            pts_map = (R @ pts.T).T + np.array(lidar_pose[:3])

            clouds_xyz.append(pts_map)
            poses_xy.append(lidar_pose[:2])

            # collect id labels if requested
            if args.label_ids:
                pose_id_labels.append((lidar_pose[0], lidar_pose[1], str(lidar_id)))

        out_path = args.out if args.out and len(frame_ids) == 1 else f"{out_dir}/frame_{fid:03d}.png"

        draw_topdown_matplotlib(
            clouds_xyz,
            labels,
            out_path=out_path,
            mark_poses=poses_xy if args.poses else None,
            pose_id_labels=pose_id_labels if args.label_ids else None,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Save top-down OPV2V LiDAR figure (headless-safe).")
    ap.add_argument("--datadir", type=str, default="../data/OPV2V",
                    help="Path to dataset root containing train/validate/test.pkl")
    ap.add_argument("--dataset", type=str, default="test", help="train / validate / test")
    ap.add_argument("--scenario", type=str, required=True, help="Scenario id (exact key in the pkl)")
    ap.add_argument("--frame", type=int, default=None, help="Frame index; if omitted, all frames are processed")
    ap.add_argument("--out", type=str, default=None, help="Output image path (only used when a single frame is processed)")
    ap.add_argument("--dpi", type=int, default=400, help="Figure DPI")
    ap.add_argument("--poses", action="store_true", help="Mark agent LiDAR poses on map")
    ap.add_argument("--label-ids", action="store_true", help="Write vehicle IDs near each LiDAR pose")
    args = ap.parse_args()
    main(args)
