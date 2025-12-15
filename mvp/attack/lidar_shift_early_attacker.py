import os
import random
import pickle
import numpy as np
import copy
import open3d as o3d

from .attacker import Attacker
from mvp.config import model_3d_path, model_3d_examples, data_root, scenario_maps
from mvp.data.util import rotation_matrix, get_point_indices_in_bbox, get_open3d_bbox, get_distance, bbox_sensor_to_map, pcd_sensor_to_map, pcd_map_to_sensor, sort_lidar_points
from mvp.tools.ray_tracing import get_model_mesh, ray_intersection, get_wall_mesh
from mvp.tools.ground_detection import get_ground_plane, get_ground_mesh
from mvp.tools.polygon_space import bbox_to_polygon


class LidarShiftEarlyAttacker(Attacker):
    def __init__(self, perception=None, dataset=None, advshape=False, sample=False, default_car_model="car_0200", attack_dataset="lidar_shift"):
        super().__init__()
        self.name = attack_dataset
        self.dataset = dataset
        self.perception = perception
        self.load_benchmark_meta()
        self.name = "lidar_shift_early"

        if advshape:
            self.name += "_AS"
        if sample:
            self.name += "_Sampled"
        self.advshape = advshape
        self.sample = sample

        # A 3D model as the fake car.
        self.default_car_model = default_car_model
        self.mesh = o3d.io.read_triangle_mesh(os.path.join(model_3d_path, "{}.ply".format(self.default_car_model)))
        # Divides the 3D model into 4 pieces.
        mesh_divide = pickle.load(open(os.path.join(model_3d_path, "spoof/mesh_divide.pkl"), "rb"))
        meshes = []
        # for vertex_indices in mesh_divide:
        #     meshes.append(self.mesh.select_by_index(vertex_indices))
        self.meshes = meshes

    def run(self, multi_frame_case, attack_opts):
        ego_id = attack_opts["ego_vehicle_id"]
        attack_info = [{} for _ in range(len(multi_frame_case))]
        for frame_id in attack_opts["frame_ids"]:
            single_vehicle_case = multi_frame_case[frame_id][ego_id]
            lidar_poses = {vehicle_id: multi_frame_case[frame_id][vehicle_id]["lidar_pose"] for vehicle_id in multi_frame_case[frame_id]}
            benign_sensor_locations = []
            for vehicle_id, vehicle_data in multi_frame_case[frame_id].items():
                if vehicle_id == ego_id:
                    continue
                benign_sensor_locations.append(
                    pcd_map_to_sensor(vehicle_data["lidar_pose"][:3][np.newaxis, :], multi_frame_case[frame_id][ego_id]["lidar_pose"])[0]
                )
            attack_opts["benign_sensor_locations"] = np.asarray(benign_sensor_locations)
            attack_opts["lidar_poses"] = lidar_poses
            new_case, info = self.run_core(single_vehicle_case, attack_opts)
            attack_info[frame_id].update({ego_id: info})
            multi_frame_case[frame_id][ego_id] = new_case
        return multi_frame_case, attack_info

    def run_core(self, single_vehicle_case, attack_opts):
        """ attack_opts: {
                "frame_ids": [-1],
                "ego_vehicle_id": int,
                "object_id": int,
                "shift_direction": float,
                "shift_distance": float,
            }
        """
        new_case = copy.deepcopy(single_vehicle_case)
        ego_pcd = new_case["lidar"]
        try:
            if "object_index" in attack_opts:
                object_index = attack_opts["object_index"]
            elif "object_id" in attack_opts:
                object_index = new_case["object_ids"].index(attack_opts["object_id"])
            bbox_to_remove = new_case["gt_bboxes"][object_index]  # Object's relative position to ego vehicle.
            bbox_to_remove[3:6] += 0.2
            bbox_to_spoof = np.copy(bbox_to_remove)
            # bbox_to_spoof[0] += np.cos(attack_opts["shift_direction"]) * attack_opts["shift_distance"]
            # bbox_to_spoof[1] += np.sin(attack_opts["shift_direction"]) * attack_opts["shift_distance"]
            # bbox_to_spoof[6] += attack_opts["rotation"]
        except:
            print("Case {}, Pair {}, Ego vehicle {}, Object vehicle {}: The target object is not available.".format(attack_opts["case_id"], attack_opts["pair_id"], attack_opts["ego_vehicle_id"], attack_opts["object_id"]))
            return new_case, {}

        # Remove
        points = ego_pcd[:, :3]
        distance = np.sum(points ** 2, axis=1) ** 0.5
        direction = points / np.tile(distance.reshape((-1, 1)), (1, 3))
        point_indices = get_point_indices_in_bbox(bbox_to_remove, ego_pcd[:,:3])
        rays = np.hstack([np.zeros((len(point_indices), 3)), direction[point_indices]])
        if rays.shape[0] == 0:
            print("The removed object is not visible.")
            return new_case, {}

        plane_model, _ = get_ground_plane(ego_pcd, method="ransac")
        ground_mesh = get_ground_mesh(plane_model)
        meshes = [ground_mesh]
        for i in range(new_case["gt_bboxes"].shape[0]):
            if i == object_index:
                continue
            bbox = new_case["gt_bboxes"][i]
            meshes.append(
                get_model_mesh(self.default_car_model, bbox)
            )
        
        intersect_points = ray_intersection(meshes, rays)
        if intersect_points.shape[0] == 0:
            print("Ray tracing failed.")
            return new_case, {}

        index_mask = (intersect_points[:,0] ** 2 < 10000)
        replace_indices = np.argwhere(index_mask > 0).reshape(-1)
        ego_pcd[point_indices[replace_indices],:3] = intersect_points[replace_indices]

        in_range_mask = (np.sqrt(np.sum(intersect_points[:,:2] ** 2, axis=1)) <= 100).astype(bool)
        replace_indices = point_indices[in_range_mask]
        replace_data = intersect_points[in_range_mask]
        ego_pcd[replace_indices,:3] = replace_data
        ignore_indices = point_indices[np.logical_not(in_range_mask)]
        remain_mask = np.ones(ego_pcd.shape[0]).astype(bool)
        remain_mask[ignore_indices] = False
        ego_pcd = ego_pcd[remain_mask]

        # Spoof
        points = ego_pcd[:, :3]
        distance = np.sum(points ** 2, axis=1) ** 0.5
        direction = points / np.tile(distance.reshape((-1, 1)), (1, 3))
        rays = np.hstack([np.zeros((direction.shape[0], 3)), direction])

        if self.sample:
            meshes = self.post_process_meshes(self.meshes, bbox_to_spoof)
            
            # Gets casted points on edges.
            replace_mask_list = []
            replace_data_list = []
            for i in range(len(meshes)):
                intersect_points = ray_intersection([meshes[i]], rays)
                in_range_mask = (intersect_points[:,0] ** 2 < 10000)
                replace_mask_list.append(in_range_mask)
                replace_data_list.append(intersect_points)

            # Estimate weights of four edges.
            mesh_weight = np.zeros(len(meshes))
            attacker_lidar_pose = attack_opts["lidar_poses"][attack_opts["attacker_vehicle_id"]]
            for vehicle_id, lidar_pose in attack_opts["lidar_poses"].items():
                if vehicle_id == attack_opts["attacker_vehicle_id"]:
                    continue
                lidar_offset = pcd_map_to_sensor(lidar_pose[np.newaxis, :3], attacker_lidar_pose)[0, :3]
                for i, mesh in enumerate(meshes):
                    vertices = np.asarray(mesh.vertices)
                    h_angle = np.arctan2(vertices[:, 1] - lidar_offset[1], vertices[:, 0] - lidar_offset[0])
                    v_angle = (vertices[:, 2] - lidar_offset[2]) / get_distance(vertices[:, :2], lidar_offset[:2])
                    mesh_weight[i] += ((h_angle.max() - h_angle.min()) / 0.005) * ((v_angle.max() - v_angle.min()) / 0.01)

            # Point sampling
            replace_data2 = []
            point_sampling_weight = np.vstack(replace_mask_list).T * mesh_weight
            replace_indices2 = np.argwhere(np.logical_or.reduce(replace_mask_list)).reshape(-1).astype(np.int32)
            for i in replace_indices2:
                replace_data2.append(
                    replace_data_list[
                        np.random.choice(mesh_weight.shape[0], p=point_sampling_weight[i]/np.sum(point_sampling_weight[i]))
                    ][i]
                )
            replace_data2 = np.array(replace_data2)
        else:
            car_mesh = get_model_mesh(self.default_car_model, bbox_to_spoof)
            intersect_points2 = ray_intersection([car_mesh], rays)
            index_mask = (intersect_points2[:,0] ** 2 < 10000) * (ego_pcd[:,0] / intersect_points2[:,0] > 1)
            replace_indices2 = np.where(index_mask > 0)[0]
            replace_data2 = intersect_points2[replace_indices2]

        ego_pcd[replace_indices2,:3] = replace_data2

        # Handles advshape and their intersection points.
        if self.advshape:
            insert_locations = self.sample_bbox_boundary_points(bbox_to_remove, spacing=0.2, height_offset=1.5)
            insert_location_distances = np.zeros(len(insert_locations))
            for benign_loc in attack_opts["benign_sensor_locations"]:
                insert_location_distances += 1 / get_distance(insert_locations, benign_loc)
            closest_indices = np.argsort(insert_location_distances)[::-1][:20]
            selected_indices = np.random.choice(closest_indices, size=5, replace=False)
            selected_insert_locations = insert_locations[selected_indices]
            advshape_meshes = [get_wall_mesh(np.array([*insert_location, 0.1, 0.1, 1.0, 0])) for insert_location in selected_insert_locations]
            advshape_data = ray_intersection(advshape_meshes, rays)
            advshape_indices = (advshape_data[:,0] ** 2 < 10000)
            advshape_data = advshape_data[advshape_indices]
            
            # NOTE: the advshape points are directly appended on the point cloud.
            append_data = advshape_data
            tmp_pcd = np.vstack([ego_pcd[:, :3], advshape_data])
            tmp_pcd, _ = sort_lidar_points(tmp_pcd)
            ego_pcd = np.hstack([tmp_pcd, np.ones((tmp_pcd.shape[0], 1))])
        else:
            append_data = None

        # Final merge
        replace_indices2_on_original = np.array([i for i in range(single_vehicle_case["lidar"].shape[0])])
        replace_indices2_on_original = np.delete(replace_indices2_on_original, ignore_indices, axis=0)[replace_indices2]
        # advshape_indices_on_original = np.array([i for i in range(single_vehicle_case["lidar"].shape[0])])
        # advshape_indices_on_original = np.delete(advshape_indices_on_original, ignore_indices, axis=0)[advshape_indices]
        final_replace_data = np.zeros((single_vehicle_case["lidar"].shape[0], 3))
        final_replace_data[replace_indices] = replace_data
        final_replace_data[replace_indices2_on_original] = replace_data2
        # final_replace_data[advshape_indices_on_original] = advshape_data
        final_replace_indices = np.union1d(replace_indices, replace_indices2_on_original)
        # final_replace_indices = np.union1d(np.union1d(replace_indices, replace_indices2_on_original), advshape_indices_on_original)
        final_replace_indices.sort()
        final_replace_data = final_replace_data[final_replace_indices]

        new_case["lidar"] = ego_pcd
        info = {
            "ignore_indices": ignore_indices,
            "replace_indices": final_replace_indices,
            "replace_data": final_replace_data,
            "append_data": append_data,
        }

        return new_case, info

    def post_process_meshes(self, meshes, bbox):
        new_meshes = []
        for mesh in meshes:
            x = copy.deepcopy(mesh)
            scale = np.min(bbox[3:6] / model_3d_examples[self.default_car_model][3:6]) 
            x = x.scale(scale, np.array([0, 0, 0]).T)
            x = x.rotate(rotation_matrix(0, bbox[6], 0), np.zeros(3).T)
            x = x.translate(bbox[:3])
            new_meshes.append(x)
        return new_meshes

    @staticmethod
    def sample_bbox_boundary_points(bbox, spacing=0.5, height_offset=1.0):
        """
        Sample boundary points of a rotated 3D bounding box with equal physical spacing.

        Parameters:
            bbox: [x, y, z, l, w, h, yaw]
            spacing: desired distance between sampled points
            height_offset: z offset from bbox center

        Returns:
            (N, 3) array of (x, y, z) sampled points on the top face perimeter
        """
        x, y, z, l, w, h, yaw = bbox

        # Local corners of the box (centered at origin, top face in x-y plane)
        half_l, half_w = l / 2, w / 2
        corners = np.array([
            [ half_l,  half_w],
            [ half_l, -half_w],
            [-half_l, -half_w],
            [-half_l,  half_w]
        ])

        # Compute segments and sample points
        boundary_points = []
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            edge_vec = end - start
            edge_len = np.linalg.norm(edge_vec)

            num_points = max(1, int(np.floor(edge_len / spacing)))
            t_vals = np.linspace(0, 1, num_points, endpoint=False)

            for t in t_vals:
                pt_local = (1 - t) * start + t * end
                boundary_points.append(pt_local)

        boundary_points = np.array(boundary_points)

        # Apply yaw rotation
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s],
                    [s,  c]])
        rotated_points = boundary_points @ R.T

        # Translate to global center and add fixed z
        rotated_points += np.array([x, y])
        z_vals = np.full((rotated_points.shape[0], 1), z + height_offset)
        points_3d = np.hstack([rotated_points, z_vals])

        return points_3d
