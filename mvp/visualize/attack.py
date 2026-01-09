import numpy as np
from mvp.data.util import rotation_matrix
import os
import matplotlib.pyplot as plt
import matplotlib

from .general import get_xylims
from mvp.config import model_3d_examples
from mvp.data.util import bbox_shift, bbox_rotate, pcd_sensor_to_map, bbox_sensor_to_map
from mvp.visualize.lidar_3d_export import lidar_to_map_xyz, concat_xyz, save_ascii_ply_xyz
from .general import draw_bbox_2d


def draw_attack(attack, normal_case, attack_case, perception_model_name, current_frame_id, mode="multi_frame", show=False, save=None):
    assert (perception_model_name in ["pixor", "pointpillar"])
    if mode == "multi_frame":
        frame_ids = [current_frame_id]
        frame_num = len(frame_ids)
        fig, axes = plt.subplots(frame_num, 2, figsize=(40, 20 * frame_num))

        # draw normal case first
        for idx, case in enumerate([normal_case, attack_case]):
            for frame_id in frame_ids:
                if frame_num <= 1:
                    ax = axes[idx]
                else:
                    ax = axes[frame_ids.index(frame_id)][idx]

                # save 3d point clouds
                xyz_list = []
                ego_vehicle_id = attack["attack_meta"]["ego_vehicle_id"]
                ego_vehicle_data = case[frame_id][ego_vehicle_id]
                xyz_map = lidar_to_map_xyz(ego_vehicle_data["lidar"], ego_vehicle_data["lidar_pose"])
                xyz_list.append(xyz_map)
                # for vehicle_id, vehicle_data in case[frame_id].items():
                #     xyz_map = lidar_to_map_xyz(vehicle_data["lidar"], vehicle_data["lidar_pose"])
                #     xyz_list.append(xyz_map)
                pointcloud_all = concat_xyz(xyz_list)
                out_path = os.path.splitext(save)[0] + ".ply"
                save_ascii_ply_xyz(out_path, pointcloud_all)
                
                # pointcloud_all = np.vstack([pcd_sensor_to_map(vehicle_data["lidar"], vehicle_data["lidar_pose"])[:,:3] for vehicle_id, vehicle_data in case[frame_id].items()])  # Draw fused point cloud
                pointcloud_all = pcd_sensor_to_map(ego_vehicle_data["lidar"], ego_vehicle_data["lidar_pose"])[:,:3]  # Draw ego point cloud
                xlim, ylim = get_xylims(pointcloud_all)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                # ax.set_aspect('equal', adjustable='box')
                ax.scatter(pointcloud_all[:,0], pointcloud_all[:,1], s=0.01, c="black")

                # label the location of victim and ego
                victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
                victim_vehicle_data = case[frame_id][victim_vehicle_id]
                ax.scatter(*ego_vehicle_data["lidar_pose"][:2].tolist(), s=100, c="green")
                ax.scatter(*victim_vehicle_data["lidar_pose"][:2].tolist(), s=100, c="red")

                # draw gt/result bboxes
                total_bboxes = []
                if "gt_bboxes" in ego_vehicle_data:
                    total_bboxes.append((bbox_sensor_to_map(ego_vehicle_data["gt_bboxes"], ego_vehicle_data["lidar_pose"]), ego_vehicle_data["object_ids"], "g"))
                if "result_bboxes" in ego_vehicle_data:
                    total_bboxes.append((bbox_sensor_to_map(ego_vehicle_data["result_bboxes"], ego_vehicle_data["lidar_pose"]), None, "b"))
                
                # label the position of replacement only in normal case
                if idx == 0:
                    # bbox = attack["attack_meta"]["bboxes"][frame_ids.index(frame_id)]
                    bbox = attack["attack_meta"]["bboxes"][frame_id]
                    bbox = bbox_sensor_to_map(bbox, ego_vehicle_data["lidar_pose"])
                    total_bboxes.append((bbox[None,:], None, 'red'))

                draw_bbox_2d(ax, total_bboxes)
    else:
        raise NotImplementedError()

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()


def draw_attack_trace(trace, show=False, save=None):
    pass
