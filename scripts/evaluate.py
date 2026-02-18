"""
Checklist before every run:
1. Filename of console.log
2. Filename of evaluate.log
3. default_shift_model
4. IoU threshold

nohup python evaluate.py > ../console_14.log 2>&1 &

Evaluation result structure:
result/attack/model_name/case_id/pair_id/vehicle_id/frame_id & attack_info.pkl/

If an NPC vehicle is totally invisible in a collaborator's LiDAR, then the NPC is not in the vehicle list of that collaborator.

PIXOR:
bev-preprocessor: numpy -> model -> bev-postprocessor: tensor (non-differentiable)
opencood_perception.run() -> early_fusion_dataset.__getitem__() -> bev_preprocessor.preprocess() -> inference_utils.inference_early_fusion() -> pixor.forward() -> early_fusion_dataset.post_process() -> bev_postprocessor.post_process()
"""

import os, sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)
import pickle
import logging
import numpy as np
from collections import OrderedDict
import json

from mvp.config import data_root, model_3d_examples
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor, pcd_sensor_to_map, pcd_map_to_sensor, get_distance
from mvp.tools.iou import iou3d, iou2d
from mvp.tools.polygon_space import bbox_to_polygon
from mvp.tools.squeezeseg.interface import SqueezeSegInterface
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.lidar_seg import lidar_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.polygon_space import get_occupied_space, get_free_space, bbox_to_polygon
from mvp.visualize.attack import draw_attack
from mvp.visualize.defense import visualize_defense, draw_roc
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_shift_early_attacker import LidarShiftEarlyAttacker
from mvp.attack.lidar_spoof_early_attacker import LidarSpoofEarlyAttacker
from mvp.attack.lidar_spoof_intermediate_attacker import LidarSpoofIntermediateAttacker
from mvp.attack.lidar_spoof_late_attacker import LidarSpoofLateAttacker
from mvp.attack.lidar_remove_early_attacker import LidarRemoveEarlyAttacker
from mvp.attack.lidar_remove_intermediate_attacker import LidarRemoveIntermediateAttacker
from mvp.attack.lidar_remove_late_attacker import LidarRemoveLateAttacker
from mvp.defense.perception_defender import PerceptionDefender
from defenses.mate.run import run_mate_attack_evaluation

result_dir = os.path.normpath("/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/result")
os.makedirs(result_dir, exist_ok=True)

attack_frame_ids = [i for i in range(10)]
TOTAL_FRAMES = 10
SUCCESS_RATE_THRESHOLD = 0.8
VICTIM_IOU_THRESHOLD = 0.1
NONVICTIM_IOU_THRESHOLD = 0.3

# Resume controls: set any of these to resume processing from a checkpoint
# None to start from the beginning
resume_case_id = None
resume_pair_id = None

logging.basicConfig(
    filename=os.path.join(result_dir, "evaluate_adv_real_car_victim_alpha10.log"),
    filemode="a",
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
# logging.basicConfig(filename=os.path.join(result_dir, "evaluate_real_car.log"), filemode="a", level=logging.INFO, format="%(levelname)s: %(message)s")

dataset = OPV2VDataset(root_path=os.path.join(data_root, "OPV2V"), mode="test")

# CHANGE THE MODEL NAME HERE
default_shift_model = "adv_real_car_victim_alpha10"
# default_shift_model = "real_car"
# CHANGE THE ATTACK DATASET HERE
attack_dataset = "lidar_shift"
# CHANGE THE PRECEPTION MODEL NAME
# pointpillar, pixor
perception_model_name = "pixor"

perception_list = [
    OpencoodPerception(fusion_method="early", model_name=perception_model_name),
    # OpencoodPerception(fusion_method="intermediate", model_name="pointpillar"),
    # OpencoodPerception(fusion_method="late", model_name="pointpillar"),
]
perception_dict = OrderedDict([(x.name, x) for x in perception_list])

attacker_list = [
    LidarShiftEarlyAttacker(dataset=dataset, default_car_model=default_shift_model, attack_dataset=attack_dataset)
    # LidarSpoofEarlyAttacker(dataset, dense=0, sync=0, default_car_model=default_spoof_model, attack_dataset=attack_dataset),
    # LidarSpoofEarlyAttacker(dataset, dense=1, sync=0, default_car_model=default_spoof_model, attack_dataset=attack_dataset),
    # LidarSpoofEarlyAttacker(dataset, dense=2, sync=0, default_car_model=default_spoof_model, attack_dataset=attack_dataset),
    # LidarSpoofEarlyAttacker(dataset, dense=2, sync=1, default_car_model=default_spoof_model, attack_dataset=attack_dataset),
    # LidarSpoofEarlyAttacker(dataset, dense=3, sync=0, default_car_model=default_spoof_model, attack_dataset=attack_dataset),
    # LidarSpoofEarlyAttacker(dataset, dense=3, sync=1, default_car_model=default_spoof_model, attack_dataset=attack_dataset),
    # LidarRemoveEarlyAttacker(dataset, advshape=0, dense=0, sync=0),
    # LidarRemoveEarlyAttacker(dataset, advshape=0, dense=1, sync=0),
    # LidarRemoveEarlyAttacker(dataset, advshape=0, dense=2, sync=0),
    # LidarRemoveEarlyAttacker(dataset, advshape=1, dense=0, sync=0),
    # LidarRemoveEarlyAttacker(dataset, advshape=1, dense=1, sync=0),
    # LidarRemoveEarlyAttacker(dataset, advshape=1, dense=2, sync=0),
    # LidarRemoveEarlyAttacker(dataset, advshape=1, dense=2, sync=1),
    # LidarRemoveEarlyAttacker(dataset, advshape=1, dense=3, sync=0),
    # LidarRemoveEarlyAttacker(dataset, advshape=1, dense=3, sync=1),
    # LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=100, sync=0, init=False, online=False),
    # LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=0, init=False, online=False),
    # LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=False, online=False),
    # LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=False, online=True),
    # LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=0, init=True, online=False),
    # LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=True, online=False),
    # LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=True, online=True),
    # LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=100, sync=0, init=False, online=False),
    # LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=0, init=False, online=False),
    # LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=False, online=False),
    # LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=False, online=True),
    # LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=0, init=True, online=False),
    # LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=True, online=False),
    # LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=True, online=True),
    # LidarSpoofLateAttacker(perception_dict["pointpillar_late"], dataset),
    # LidarRemoveLateAttacker(perception_dict["pointpillar_late"], dataset),
]
attacker_dict = OrderedDict([(x.name, x) for x in attacker_list])

defender_list = [
    PerceptionDefender(),
]
defender_dict = OrderedDict([(x.name, x) for x in defender_list])

pickle_cache = OrderedDict()
pickle_cache_size = 20


def pickle_cache_load(file_path):
    file_path = os.path.normpath(file_path)
    if file_path in pickle_cache:
        return pickle_cache[file_path]
    else:
        data = pickle.load(open(file_path, 'rb'))
        if len(pickle_cache) >= pickle_cache_size:
            pickle_cache.popitem(last=False)
        pickle_cache[file_path] = data
        return data
    

def pickle_cache_dump(data, file_path):
    file_path = os.path.normpath(file_path)
    if file_path in pickle_cache:
        pickle_cache[file_path] = data
    pickle.dump(data, open(file_path, 'wb'))


def normal_case_iterator(f):
    def wrapper(*args, **kwargs):
        for case_id, case in dataset.case_generator(tag="multi_frame", index=True, use_lidar=True, use_camera=False):
            data_dir = os.path.join(result_dir, "normal/{:06d}".format(case_id))
            os.makedirs(data_dir, exist_ok=True)

            kwargs.update({
                "case_id": case_id,
                "case": case,
                "data_dir": data_dir,
            })
            f(*args, **kwargs)
    return wrapper


def attack_case_iterator(f):
    def wrapper(*args, **kwargs):
        attacker = args[0]
        resume_reached = (resume_case_id is None and resume_pair_id is None)
        for attack_id, attack in enumerate(attacker.attack_list):
            case_id = attack["attack_meta"]["case_id"]
            pair_id = attack["attack_meta"]["pair_id"]

            if not resume_reached:
                if resume_case_id is not None and case_id != resume_case_id:
                    continue
                if resume_pair_id is not None and pair_id != resume_pair_id:
                    continue
                resume_reached = True
                logging.info("Resuming from attack_id %d (case %s, pair %s)", attack_id, str(case_id), str(pair_id))

            data_dir = os.path.join(result_dir, "attack/{}/{}/case{:06d}/pair{:02d}".format(perception_model_name, default_shift_model, case_id, pair_id))
            os.makedirs(data_dir, exist_ok=True)
            case = dataset.get_case(case_id, tag="multi_frame", use_lidar=True, use_camera=False)

            kwargs.update({
                "case_id": case_id,
                "case": case,
                "pair_id": pair_id,
                "data_dir": data_dir,
                "attack_id": attack_id,
                "attack": attack,
            })
            f(*args, **kwargs)
    return wrapper


@normal_case_iterator
def normal_perception(case_id=None, case=None, data_dir=None):
    for perception_name, perception in perception_dict.items():
        save_file = os.path.join(data_dir, "{}.pkl".format(perception_name))
        if os.path.isfile(save_file):
            continue
        else:
            logging.info("Processing perception {} on normal case {}".format(perception.name, case_id))

        perception_feature = [{} for _ in range(TOTAL_FRAMES)]
        for frame_id in attack_frame_ids:
            for vehicle_id in list(case[frame_id].keys()):
                pred_bboxes, pred_scores = perception.run(case[frame_id], ego_id=vehicle_id)
                perception_feature[frame_id][vehicle_id] = {"pred_bboxes": pred_bboxes, "pred_scores": pred_scores}

        pickle_cache_dump(perception_feature, save_file)
        

@attack_case_iterator
def attack_perception(attacker, case_id=None, case=None, pair_id=None, data_dir=None, attack_id=None, attack=None):
    attack_opts = attack["attack_opts"]
    attack_opts["ego_vehicle_id"] = attack["attack_meta"]["ego_vehicle_id"]
    attack_opts["frame_ids"] = [i for i in range(10)]
    attack["attack_meta"]["attack_frame_ids"] = [9]

    data_dir = os.path.join(data_dir, str(attack_opts["ego_vehicle_id"]))
    os.makedirs(data_dir, exist_ok=True)
    save_file = os.path.join(data_dir, "attack_info.pkl")
    if os.path.isfile(save_file):
        return
    else:
        logging.info("Processing attack {} and attack case {}".format(attacker.name, attack_id))

    if (isinstance(attacker, LidarSpoofEarlyAttacker) or isinstance(attacker, LidarRemoveEarlyAttacker)) and attacker.dense == 2 and attacker.sync == 1:
        # Need to attack all frames here as the data is used for online intermediate-fusion attack.
        attack_opts["frame_ids"] = [i for i in range(10)]

    if (isinstance(attacker, LidarSpoofIntermediateAttacker) or isinstance(attacker, LidarRemoveIntermediateAttacker)):
        # Intermediate-fusion attacks need the result of ray casting.
        if attacker.init:
            attack_category = attacker.name.split('_')[1]
            attack_opts["attack_info"] = pickle_cache_load(os.path.join(data_dir, "../../lidar_{}_early_AS_DenseAll_Async/{:04d}/attack_info.pkl".format(attack_category, attack_id)))
        else:
            attack_opts["attack_info"] = [{} for _ in range(10)]
        if attacker.online:
            attack_opts["frame_ids"] = [i for i in range(1, 10)]

    new_case, attack_info = attacker.run(case, attack_opts)
    pickle_cache_dump(attack_info, save_file)

    if isinstance(attacker, LidarSpoofEarlyAttacker) or isinstance(attacker, LidarRemoveEarlyAttacker) or isinstance(attacker, LidarShiftEarlyAttacker):
        # Early-fusion attacks are block box attacks. We need to apply certain models to evaluate their performance.
        # for perception_name in ["pointpillar_early", "pointpillar_intermediate"]:
        for perception_name in [f"{perception_model_name}_early"]:
            perception = perception_dict[perception_name]
            perception_feature = [{} for _ in range(TOTAL_FRAMES)]
            for frame_id in attack_frame_ids:
                os.makedirs(os.path.join(data_dir, "frame{}".format(frame_id)), exist_ok=True)
                pred_bboxes, pred_scores = perception.run(new_case[frame_id], ego_id=attack_opts["ego_vehicle_id"])
                perception_feature[frame_id][attack_opts["ego_vehicle_id"]] = {"pred_bboxes": pred_bboxes, "pred_scores": pred_scores}
                perception_save_file = os.path.join(data_dir, "frame{}".format(frame_id), "{}.pkl".format(perception_name))
                pickle_cache_dump(perception_feature, perception_save_file)

                new_case[frame_id][attack_opts["ego_vehicle_id"]]["result_bboxes"] = pred_bboxes
                new_case[frame_id][attack_opts["ego_vehicle_id"]]["result_scores"] = pred_scores
                print("Case {}, Pair {}, Frame {}, Vehicle {}: Num of pred bboxes: {}".format(case_id, pair_id, frame_id, attack_opts["ego_vehicle_id"],len(pred_bboxes)))
                
                # Visualization
                dataset.load_feature(new_case, perception_feature)
                draw_attack(attack, case, new_case, perception_model_name, default_shift_model, current_frame_id=frame_id, mode="multi_frame", show=False, save=os.path.join(data_dir, "frame{}".format(frame_id), "{}.png".format(perception_name)))
    else:
        # Visualization
        dataset.load_feature(new_case, attack_info)
        draw_attack(attack, case, new_case, mode="multi_frame", show=False, save=os.path.join(data_dir, "visualization.png"))


def attack_evaluation_iou(attacker, perception_name):
    logging.info("Evaluating attack {} at perception {} on mesh model {}".format(attacker.name, perception_name, default_shift_model))
    case_number = len(attacker.attack_list)
    success_log = np.zeros((case_number, TOTAL_FRAMES)).astype(bool)
    valid_log = np.ones((case_number, TOTAL_FRAMES)).astype(bool)
    max_iou = np.zeros((case_number, TOTAL_FRAMES, 2)).astype(np.float32)
    best_score = np.zeros((case_number, TOTAL_FRAMES, 2)).astype(np.float32)

    save_dir = os.path.join(result_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    @attack_case_iterator
    def attack_evaluation_processor(attacker, perception_name, case_id=None, case=None, pair_id=None, data_dir=None, attack_id=None, attack=None):
        ego_id = attack["attack_meta"]["ego_vehicle_id"]
        victim_id = attack["attack_meta"]["victim_vehicle_id"]
        # attacker_id = attack["attack_meta"]["attacker_vehicle_id"]
        # attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bboxes"][-1], case[9][attacker_id]["lidar_pose"])
        # attack_bbox = bbox_map_to_sensor(attack_bbox, case[-1][ego_id]["lidar_pose"])

        # feature_data = pickle_cache_load(os.path.join(result_dir, "normal/{:06d}/{}.pkl".format(case_id, perception_name)))

        # pred_bboxes = feature_data[-1][ego_id]["pred_bboxes"]
        # pred_scores = feature_data[-1][ego_id]["pred_scores"]
        # for j, pred_bbox in enumerate(pred_bboxes):
        #     iou = iou3d(pred_bbox, attack_bbox)
        #     if iou > max_iou[attack_id, 0]:
        #         max_iou[attack_id, 0] = iou
        #         best_score[attack_id, 0] = pred_scores[j]

        # if "early" in attacker.name:
            # feature_data = pickle_cache_load(os.path.join(data_dir, str(ego_id), "{}.pkl".format(perception_name)))
        # else:
            # feature_data = pickle_cache_load(os.path.join(data_dir, "attack_info.pkl"))
        feature_data_path = os.path.join(data_dir, str(ego_id))
        attack_info = pickle_cache_load(os.path.join(feature_data_path, "attack_info.pkl"))
        
        # Calculate success rate by frames
        print(f"[INFO] Case {case_id}, Pair {pair_id}, Vehicle {ego_id}.")
        logging.info(f"Case {case_id}, Pair {pair_id}, Vehicle {ego_id}.")
        for frame_id in attack_frame_ids:
            frame_attack_info = attack_info[frame_id].get(ego_id, {})
            if not frame_attack_info:
                # Attack not applied in this frame. Exclude it from metrics.
                valid_log[attack_id][frame_id] = False
                logging.info(f"[INVALID] Frame {frame_id} is invalid.")
                continue

            attack_bbox = np.copy(attack["attack_meta"]["bboxes"][frame_id])
            attack_bbox[3:6] = model_3d_examples[default_shift_model][3:6]

            feature_data = pickle_cache_load(os.path.join(feature_data_path, f"frame{frame_id}", "{}.pkl".format(perception_name)))
            pred_bboxes = feature_data[frame_id][ego_id]["pred_bboxes"]
            pred_scores = feature_data[frame_id][ego_id]["pred_scores"]
            if pred_bboxes is None or len(pred_bboxes) == 0:
                # Prediction is not available in this frame. Exclude it from metrics.
                valid_log[attack_id][frame_id] = False
                logging.info(f"[INVALID] Frame {frame_id} has no predicted bboxes.")
                continue
            for j, pred_bbox in enumerate(pred_bboxes):
                if perception_model_name == "pointpillar":
                    iou = iou3d(pred_bbox, attack_bbox)
                elif perception_model_name == "pixor":
                    iou = iou2d(pred_bbox, attack_bbox)
                else:
                    raise NotImplementedError("Not support this perception model.")
                
                if iou > max_iou[attack_id, frame_id, 1]:
                    max_iou[attack_id, frame_id, 1] = iou
                    best_score[attack_id, frame_id, 1] = pred_scores[j]

            if attacker.name.startswith("lidar_spoof") and max_iou[attack_id, frame_id, 1] > 0:
                success_log[attack_id][frame_id] = True
            elif attacker.name.startswith("lidar_remove") and max_iou[attack_id, frame_id, 1] == 0:
                success_log[attack_id][frame_id] = True
            elif attacker.name.startswith("lidar_shift"):
                if ego_id == victim_id and max_iou[attack_id, frame_id, 1] <= VICTIM_IOU_THRESHOLD:
                    success_log[attack_id][frame_id] = True
                elif ego_id != victim_id and max_iou[attack_id, frame_id, 1] >= NONVICTIM_IOU_THRESHOLD:
                    success_log[attack_id][frame_id] = True

    attack_evaluation_processor(attacker, perception_name)

    # Aggregate success rate by (case_id, pair_id) combinations
    combination_index_map = {}
    combination_difficulty_map = {}
    for idx, attack in enumerate(attacker.attack_list):
        meta = attack["attack_meta"]
        combination_key = (meta["case_id"], meta["pair_id"])
        combination_index_map.setdefault(combination_key, []).append(idx)
        combination_difficulty_map.setdefault(combination_key, meta["difficulty"])
    combination_results = []
    for (case_id, pair_id), attack_indices in combination_index_map.items():
        ego_success_details = []
        for idx in attack_indices:
            ego_id = attacker.attack_list[idx]["attack_meta"]["ego_vehicle_id"]
            victim_id = attacker.attack_list[idx]["attack_meta"]["victim_vehicle_id"]
            frames_succeeded = int(np.sum(success_log[idx] & valid_log[idx]))
            valid_frames = int(np.sum(valid_log[idx]))
            ego_success = (float(frames_succeeded / valid_frames) >= SUCCESS_RATE_THRESHOLD) if valid_frames > 0 else True
            ego_success_details.append({
                "ego_id": ego_id,
                "is_victim": bool(ego_id==victim_id),
                "successful_frames_number": frames_succeeded,
                "valid_frames_number": valid_frames,
                "success rate": float(frames_succeeded / valid_frames) if valid_frames > 0 else 1.0,
                "ego_success": ego_success,
            })
        combo_success = all(item["ego_success"] for item in ego_success_details)
        combination_results.append({
            "case_id": case_id,
            "pair_id": pair_id,
            "difficulty": combination_difficulty_map.get((case_id, pair_id)),
            "combination_success": combo_success,
            "ego_details": ego_success_details,
        })
    combination_total = len(combination_results)
    combination_success_count = sum(1 for item in combination_results if item["combination_success"])
    combination_success_rate = combination_success_count / combination_total if combination_total > 0 else 0.0

    # Calculate success rate by whether the ego vehicle is the victim.
    victim_success_ego = victim_total_ego = non_victim_success_ego = non_victim_total_ego = 0
    victim_valid_frames = victim_success_frames = 0
    non_victim_valid_frames = non_victim_success_frames = 0
    difficulty_stats = {}
    difficulty_levels = sorted(diff for diff in {attack["attack_meta"]["difficulty"] for attack in attacker.attack_list} if diff is not None)
    for difficulty in difficulty_levels:
        difficulty_stats[difficulty] = {
            "combination_total": 0,
            "combination_success_count": 0,
            "combination_success_rate": 0.0,
            "victim_success_ego": 0,
            "victim_total_ego": 0,
            "victim_success_rate": 0.0,
            "non_victim_success_ego": 0,
            "non_victim_total_ego": 0,
            "non_victim_success_rate": 0.0,
            "valid_frames_total": 0,
            "success_frames_total": 0,
            "valid_frames_victim": 0,
            "success_frames_victim": 0,
            "valid_frames_non_victim": 0,
            "success_frames_non_victim": 0,
            "frame_success_rate": 0.0,
            "frame_success_rate_victim": 0.0,
            "frame_success_rate_non_victim": 0.0,
        }
    for idx, attack in enumerate(attacker.attack_list):
        ego_id = attack["attack_meta"]["ego_vehicle_id"]
        victim_id = attack["attack_meta"]["victim_vehicle_id"]
        difficulty = attack["attack_meta"]["difficulty"]
        assert(difficulty in difficulty_stats)
        is_victim = ego_id == victim_id
        frames_succeeded = int(np.sum(success_log[idx] & valid_log[idx]))
        valid_frames = int(np.sum(valid_log[idx]))
        ego_success = (float(frames_succeeded / valid_frames) >= SUCCESS_RATE_THRESHOLD) if valid_frames > 0 else True
        if is_victim:
            victim_total_ego += 1
            victim_success_ego += int(ego_success)
            victim_valid_frames += valid_frames
            victim_success_frames += frames_succeeded
            difficulty_stats[difficulty]["victim_total_ego"] += 1
            difficulty_stats[difficulty]["victim_success_ego"] += int(ego_success)
            difficulty_stats[difficulty]["valid_frames_victim"] += valid_frames
            difficulty_stats[difficulty]["success_frames_victim"] += frames_succeeded
        else:
            non_victim_total_ego += 1
            non_victim_success_ego += int(ego_success)
            non_victim_valid_frames += valid_frames
            non_victim_success_frames += frames_succeeded
            difficulty_stats[difficulty]["non_victim_total_ego"] += 1
            difficulty_stats[difficulty]["non_victim_success_ego"] += int(ego_success)
            difficulty_stats[difficulty]["valid_frames_non_victim"] += valid_frames
            difficulty_stats[difficulty]["success_frames_non_victim"] += frames_succeeded
        difficulty_stats[difficulty]["valid_frames_total"] += valid_frames
        difficulty_stats[difficulty]["success_frames_total"] += frames_succeeded
    victim_success_rate = victim_success_ego / victim_total_ego if victim_total_ego > 0 else 0.0
    non_victim_success_rate = non_victim_success_ego / non_victim_total_ego if non_victim_total_ego > 0 else 0.0

    # Calculate success rate by single frame
    valid_frames_total = int(np.sum(valid_log))
    success_frames_total = int(np.sum(success_log & valid_log))
    frame_success_rate = success_frames_total / valid_frames_total if valid_frames_total > 0 else 0.0
    frame_success_rate_victim = victim_success_frames / victim_valid_frames if victim_valid_frames > 0 else 0.0
    frame_success_rate_non_victim = non_victim_success_frames / non_victim_valid_frames if non_victim_valid_frames > 0 else 0.0

    # Aggregate combination success rate by difficulty.
    for combination_result in combination_results:
        difficulty = combination_result.get("difficulty")
        assert(difficulty in difficulty_stats)
        difficulty_stats[difficulty]["combination_total"] += 1
        difficulty_stats[difficulty]["combination_success_count"] += int(combination_result["combination_success"])

    for difficulty, stats in difficulty_stats.items():
        if stats["combination_total"] > 0:
            stats["combination_success_rate"] = stats["combination_success_count"] / stats["combination_total"]
        if stats["victim_total_ego"] > 0:
            stats["victim_success_rate"] = stats["victim_success_ego"] / stats["victim_total_ego"]
        if stats["non_victim_total_ego"] > 0:
            stats["non_victim_success_rate"] = stats["non_victim_success_ego"] / stats["non_victim_total_ego"]
        if stats["valid_frames_total"] > 0:
            stats["frame_success_rate"] = stats["success_frames_total"] / stats["valid_frames_total"]
        if stats["valid_frames_victim"] > 0:
            stats["frame_success_rate_victim"] = stats["success_frames_victim"] / stats["valid_frames_victim"]
        if stats["valid_frames_non_victim"] > 0:
            stats["frame_success_rate_non_victim"] = stats["success_frames_non_victim"] / stats["valid_frames_non_victim"]
    
    # Save the evaluation results
    # pickle_cache_dump({"success": success_log, "iou": max_iou, "score": best_score},
    #                   os.path.join(save_dir, "attack_result_{}_{}.pkl".format(attacker.name, perception_name)))
    evaluation_results = {
        "success": success_log.tolist(),
        "valid": valid_log.tolist(),
        "combination_results": combination_results,
        "combination_total": combination_total,
        "combination_success_count": combination_success_count,
        "combination_success_rate": combination_success_rate,
        "valid_frames_total": valid_frames_total,
        "success_frames_total": success_frames_total,
        "frame_success_rate": frame_success_rate,
        "valid_frames_victim": victim_valid_frames,
        "success_frames_victim": victim_success_frames,
        "frame_success_rate_victim": frame_success_rate_victim,
        "valid_frames_non_victim": non_victim_valid_frames,
        "success_frames_non_victim": non_victim_success_frames,
        "frame_success_rate_non_victim": frame_success_rate_non_victim,
        "victim_success_ego": victim_success_ego,
        "victim_total_ego": victim_total_ego,
        "victim_success_rate": victim_success_rate,
        "non_victim_success_ego": non_victim_success_ego,
        "non_victim_total_ego": non_victim_total_ego,
        "non_victim_success_rate": non_victim_success_rate,
        "difficulty_stats": difficulty_stats,
    }
    save_path = os.path.join(save_dir, "attack_result_{}_{}_{}.txt".format(default_shift_model, perception_name, attacker.name))
    with open(save_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    logging.info("Evaluation of attack {} at perception {}".format(
        attacker.name, perception_name))
    logging.info("Total cases: {}".format(
        success_log.shape[0]))
    logging.info("Total frame success: {}/{} ({:.4f})".format(
        success_frames_total, valid_frames_total, frame_success_rate))
    logging.info("Victim frame success: {}/{} ({:.4f})".format(
        victim_success_frames, victim_valid_frames, frame_success_rate_victim))
    logging.info("Non-victim frame success: {}/{} ({:.4f})".format(
        non_victim_success_frames, non_victim_valid_frames, frame_success_rate_non_victim))
    logging.info("Average IoU: {:.4f}, average score: {:.4f}".format(
        np.mean(max_iou[:, 1]), np.mean(best_score[:, 1])))
    logging.info("Combination (case, pair): {}/{} ({:.4f})".format(
        combination_success_count, combination_total, combination_success_rate))
    logging.info("Victim and non-victim success: victim {}/{} ({:.4f}), non-victim {}/{} ({:.4f})".format(
        victim_success_ego, victim_total_ego, victim_success_rate,
        non_victim_success_ego, non_victim_total_ego, non_victim_success_rate))
    for difficulty in sorted(difficulty_stats):
        stats = difficulty_stats[difficulty]
        logging.info(
            "Difficulty {}: combination {}/{} ({:.4f}), total frame {}/{} ({:.4f}), victim {}/{} ({:.4f}), non-victim {}/{} ({:.4f}), victim frame {}/{} ({:.4f}), non-victim frame {}/{} ({:.4f})".format(
                difficulty,
                stats["combination_success_count"],
                stats["combination_total"],
                stats["combination_success_rate"],
                stats["success_frames_total"],
                stats["valid_frames_total"],
                stats["frame_success_rate"],
                stats["victim_success_ego"],
                stats["victim_total_ego"],
                stats["victim_success_rate"],
                stats["non_victim_success_ego"],
                stats["non_victim_total_ego"],
                stats["non_victim_success_rate"],
                stats["success_frames_victim"],
                stats["valid_frames_victim"],
                stats["frame_success_rate_victim"],
                stats["success_frames_non_victim"],
                stats["valid_frames_non_victim"],
                stats["frame_success_rate_non_victim"],
            )
        )

def attack_evaluation_defenses(attacker, perception_name, defense_name: str):
    if defense_name == "mate":
        run_mate_attack_evaluation(
            attacker=attacker,
            dataset=dataset,
            result_dir=result_dir,
            perception_model_name=perception_model_name,
            default_shift_model=default_shift_model,
            perception_name=perception_name,
            attack_frame_ids=attack_frame_ids,
            pickle_cache_load=pickle_cache_load,
            logger=logging,
        )
    else:
        raise NotImplementedError("Only support MATE")


@normal_case_iterator
def occupancy_map(lidar_seg_api, case_id=None, case=None, data_dir=None):
    save_file = os.path.join(data_dir, "occupancy_map.pkl")
    if os.path.isfile(save_file):
        return
    else:
        logging.info("Processing occupancy map of case {}".format(case_id))

    occupancy_feature = [{} for _ in range(TOTAL_FRAMES)]
    for frame_id in attack_frame_ids:
        for vehicle_id, vehicle_data in case[frame_id].items():
            lidar, lidar_pose = vehicle_data["lidar"], vehicle_data["lidar_pose"]
            pcd = pcd_sensor_to_map(lidar, lidar_pose)

            lane_info = pickle_cache_load(os.path.join(data_root, "carla/{}_lane_info.pkl".format(vehicle_data["map"])))
            lane_areas = pickle_cache_load(os.path.join(data_root, "carla/{}_lane_areas.pkl".format(vehicle_data["map"])))
            lane_planes = pickle_cache_load(os.path.join(data_root, "carla/{}_ground_planes.pkl".format(vehicle_data["map"])))

            ground_indices, in_lane_mask, point_height = get_ground_plane(pcd, lane_info=lane_info, lane_areas=lane_areas, lane_planes=lane_planes, method="map")
            lidar_seg = lidar_segmentation(lidar, method="squeezeseq", interface=lidar_seg_api)
            
            object_segments = filter_segmentation(lidar, lidar_seg, lidar_pose, in_lane_mask=in_lane_mask, point_height=point_height, max_range=50)
            object_mask = np.zeros(pcd.shape[0]).astype(bool)
            if len(object_segments) > 0:
                object_indices = np.hstack(object_segments)
                object_mask[object_indices] = True

            ego_bbox = vehicle_data["ego_bbox"]
            ego_area = bbox_to_polygon(ego_bbox)
            ego_area_height = ego_bbox[5]

            ret = {
                "ego_area": ego_area,
                "ego_area_height": ego_area_height,
                "plane": None,
                "ground_indices": ground_indices,
                "point_height": point_height,
                "object_segments": object_segments,
            }

            height_thres = 0
            occupied_areas, occupied_areas_height = get_occupied_space(pcd, object_segments, point_height=point_height, height_thres=height_thres)
            free_areas = get_free_space(lidar, lidar_pose, object_mask, in_lane_mask=in_lane_mask, point_height=point_height, max_range=50, height_thres=height_thres, height_tolerance=0.2)
            ret["occupied_areas"] = occupied_areas
            ret["occupied_areas_height"] = occupied_areas_height
            ret["free_areas"] = free_areas
            
            occupancy_feature[frame_id][vehicle_id] = ret

    pickle_cache_dump(occupancy_feature, save_file)


# @attack_case_iterator
# def defense(attacker, defender, perception_name, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
#     if "early" in attacker.name:
#         save_file = os.path.join(data_dir, "{}_{}.pkl".format(defender.name, perception_name))
#         vis_file = os.path.join(data_dir, "{}_{}.png".format(defender.name, perception_name))
#     else:
#         save_file = os.path.join(data_dir, "{}.pkl".format(defender.name))
#         vis_file = os.path.join(data_dir, "{}.png".format(defender.name))
#     if os.path.isfile(save_file):
#         return
#     else:
#         logging.info("Processing defense {} against attack {} on attack case {}".format(defender.name, attacker.name, attack_id))
#     logging.info("Processing defense {} against attack {} on attack case {}".format(defender.name, attacker.name, attack_id))

#     if "early" in attacker.name:
#         perception_feature = pickle_cache_load(os.path.join(data_dir, "{}.pkl".format(perception_name)))
#     else:
#         perception_feature = pickle_cache_load(os.path.join(data_dir, "attack_info.pkl"))
#     case = dataset.load_feature(case, perception_feature)

#     occupancy_feature = pickle_cache_load(os.path.join(result_dir, "normal/{:06d}/occupancy_map.pkl".format(case_id)))
#     case = dataset.load_feature(case, occupancy_feature)

#     defend_opts = {"frame_ids": [9]}
#     new_case, score, metrics = defender.run(case, defend_opts)

#     pickle_cache_dump(metrics, save_file)
#     visualize_defense(case, metrics, show=False, save=vis_file)


# def defense_evaluation(attacker, defender, perception_name):
#     save_dir = os.path.join(result_dir, "evaluation")
#     os.makedirs(save_dir, exist_ok=True)
#     save_file = os.path.join(save_dir, "defense_result_{}_{}_{}.pkl".format(attacker.name, defender.name, perception_name))

#     defense_results = {
#         "spoof_error": [],
#         "spoof_label": [],
#         "spoof_location": [],
#         "remove_error": [],
#         "remove_label": [],
#         "remove_location": [],
#         "success": [],
#     }

#     @attack_case_iterator
#     def defense_evaluation_processor(attacker, defender, perception_name, case_id=None, case=None, data_dir=None, attack_id=None, attack=None, iou_thres=0.7, dist_thres=40):
#         if "early" in attacker.name:
#             defense_file = os.path.join(data_dir, "{}_{}.pkl".format(defender.name, perception_name))
#         else:
#             defense_file = os.path.join(data_dir, "{}.pkl".format(defender.name))
#         metrics = pickle_cache_load(defense_file)

#         attacker_vehicle_id = attack["attack_meta"]["attacker_vehicle_id"]
#         victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
#         attack_mode =  "spoof" if "spoof" in attacker.name else "remove"
#         attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bboxes"][-1], case[-1][attacker_vehicle_id]["lidar_pose"])

#         victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
#         for frame_id in attack_frame_ids:
#             vehicle_metrics = metrics[frame_id][victim_vehicle_id]

#         gt_bboxes = vehicle_metrics["gt_bboxes"]
#         pred_bboxes = vehicle_metrics["pred_bboxes"]
#         lidar_pose = vehicle_metrics["lidar_pose"]

#         # iou 2d
#         gt_bboxes[:, 2] = 0
#         gt_bboxes[:, 5] = 1
#         pred_bboxes[:, 2] = 0
#         pred_bboxes[:, 5] = 1

#         iou = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
#         for i, gt_bbox in enumerate(gt_bboxes):
#             for j, pred_bbox in enumerate(pred_bboxes):
#                 iou[i, j] = iou3d(gt_bbox, pred_bbox)

#         spoof_label = np.max(iou, axis=0) <= iou_thres
#         spoof_mask = np.logical_and(get_distance(pred_bboxes[:, :2], lidar_pose[:2]) > 1, get_distance(pred_bboxes[:, :2], lidar_pose[:2]) <= dist_thres)
#         remove_label = np.max(iou, axis=1) <= iou_thres
#         remove_mask = get_distance(gt_bboxes[:, :2], lidar_pose[:2]) <= dist_thres

#         spoof_error = np.zeros(pred_bboxes.shape[0])
#         spoof_location = np.zeros((pred_bboxes.shape[0], 2))
#         for error_area, error, gt_error, bbox_index in vehicle_metrics["spoof"]:
#             if error > spoof_error[bbox_index]:
#                 spoof_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
#                 spoof_error[bbox_index] = error

#         remove_error = np.zeros(gt_bboxes.shape[0])
#         remove_location = np.zeros((gt_bboxes.shape[0], 2))
#         for error_area, error, gt_error, bbox_index in vehicle_metrics["remove"]:
#             if bbox_index < 0:
#                 continue
#             if error > remove_error[bbox_index]:
#                 remove_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
#                 remove_error[bbox_index] = error

#         detected_location = spoof_location if attack_mode == "spoof" else remove_location
#         is_success = np.min(get_distance(detected_location, attack_bbox[:2])) < 2

#         defense_results["spoof_error"].append(spoof_error[spoof_mask])
#         defense_results["spoof_label"].append(spoof_label[spoof_mask])
#         defense_results["spoof_location"].append(spoof_location[spoof_mask])
#         defense_results["remove_error"].append(remove_error[remove_mask])
#         defense_results["remove_label"].append(remove_label[remove_mask])
#         defense_results["remove_location"].append(remove_location[remove_mask])
#         defense_results["success"].append(np.array([is_success]).astype(np.int8))

#     defense_evaluation_processor(attacker, defender, perception_name)

#     for key, data in defense_results.items():
#         defense_results[key] = np.concatenate(data).reshape(-1)

#     pickle_cache_dump(defense_results, save_file)
#     spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres = draw_roc(defense_results["spoof_error"], defense_results["spoof_label"],
#             save=os.path.join(save_dir, "roc_lidar_spoof_{}_{}_{}.png".format(attacker.name, defender.name, perception_name)))
#     remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres = draw_roc(defense_results["remove_error"], defense_results["remove_label"],
#             save=os.path.join(save_dir, "roc_lidar_remove_{}_{}_{}.png".format(attacker.name, defender.name, perception_name)))
    
#     attack_result = pickle_cache_load(os.path.join(save_dir, "attack_result_{}_{}.pkl".format(attacker.name, perception_name)))
#     success_rate = np.mean(attack_result["success"] * defense_results["success"])
    
#     logging.info("Evaluation of defense {} against attack {} on perception {} success rate {:.2f}: For spoofing attack, best TPR {:.2f}, best FPR {:.2f}, ROC AUC {:.2f}, best threshold {:.2f}; For removal attack, best TPR {:.2f}, best FPR {:.2f}, ROC AUC {:.2f}, best threshold {:.2f}." .format(
#         defender.name, attacker.name, perception_name, success_rate,
#         spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres, remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres
#     ))


def main():
    # First do perception on normal cases as the baseline.
    # logging.info("######################## Perception on normal cases ########################")
    # normal_perception()

    # Launch all attacks.
    logging.info("######################## Run attacks ########################")
    for attacker_name, attacker in attacker_dict.items():
        # attack_perception(attacker)
        if "early" in attacker_name:
            # for perception_name in ["pointpillar_early", "pointpillar_intermediate"]:
            for perception_name in [f"{perception_model_name}_early"]:
                # logging.info("######################## Run IoU evaluation ########################")
                # attack_evaluation_iou(attacker, perception_name)
                logging.info("######################## Run defenses evaluation ########################")
                attack_evaluation_defenses(attacker, perception_name, "mate")
        # else:
        #     attack_evaluation(attacker, attacker.perception.name)
    
    # # Precompute occupancy maps for defense.
    # logging.info("######################## Generating occupancy maps ########################")
    # lidar_seg_api = SqueezeSegInterface()
    # occupancy_map(lidar_seg_api)

    # # Launch the defense on each attack.
    # logging.info("######################## Launching defenses ########################")
    # for attacker_name, attacker in attacker_dict.items():
    #     for defender_name, defender in defender_dict.items():
    #         if "early" in attacker_name:
    #             for perception_name in ["pointpillar_early", "pointpillar_intermediate"]:
    #                 defense(attacker, defender, perception_name)
    #                 defense_evaluation(attacker, defender, perception_name)
    #         else:
    #             defense(attacker, defender, attacker.perception.name)
    #             defense_evaluation(attacker, defender, perception_name)


if __name__ == "__main__":
    main()
