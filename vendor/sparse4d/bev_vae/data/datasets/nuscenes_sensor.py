from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
from functools import cached_property

import numpy as np
import pandas as pd
from bev_vae.data.datasets.utils import (
    read_feather, read_json_file, read_img, 
    quaternion2matrix, extract_matrix_xy,
    matrix2quaternion, yaw2quaternion, 
    polygon_geoms_to_instances, filter_short_map_instances, 
    shift_fixed_num_sampled_points_v2)
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from bev_vae.data.transforms import denormalize
import torchvision.transforms.functional as F
from nuscenes.map_expansion.map_api import NuScenesMap


NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}


@dataclass
class SensorDataset:
    data_dir: Path
    split: str
    cam_names: List[str]
    other_sensors: List[str]
    class_num: int 
    scene_size: List[int] 
    pc: List[float] 
    with_condition: bool = False
    return_bev: bool = False
    with_bboxes: bool = False
    with_map_vectors: bool = False
    vertex_num: int = 20
    temporal: bool = False
    maptr_v2: bool = False

    def __post_init__(self) -> None:
        sensor_cache = read_feather(self.data_dir / "nusc/sensor_cache.feather")
        synchronization_cache = read_feather(self.data_dir / "nusc/synchronization_cache.feather")
        self.sensor_cache = sensor_cache.loc[sensor_cache['split'] == self.split]
        self.synchronization_cache = synchronization_cache.loc[synchronization_cache['split'] == self.split]
        if "LIDAR_TOP" in self.other_sensors:
            scene2frame = read_json_file(self.data_dir / "nusc/scene2frame.json")
            logs = self.sensor_cache['log_id'].unique().tolist()
            self.scene2frame = {scene: scene2frame[scene] for scene in logs}
            lidar_cache = self.sensor_cache[self.sensor_cache["sensor_name"] == "LIDAR_TOP"]
            lidar_synchronization_cache = self.synchronization_cache[self.synchronization_cache["sensor_name"] == "LIDAR_TOP"]
            frame_cache, frame_synchronization_cache = [], []
            for log_id, frame in self.scene2frame.items():
                scene_cache = lidar_cache[lidar_cache["log_id"] == log_id]
                scene_synchronization_cache = lidar_synchronization_cache[lidar_synchronization_cache["log_id"] == log_id]
                assert np.array_equal(scene_synchronization_cache["LIDAR_TOP"].values, scene_cache["timestamp_ns"].values)
                valid_frame = scene_cache["timestamp_ns"].isin(frame).values
                frame_cache.append(scene_cache[valid_frame])
                frame_synchronization_cache.append(scene_synchronization_cache[valid_frame])
            frame_cache = pd.concat(frame_cache, ignore_index=True)
            frame_synchronization_cache = pd.concat(frame_synchronization_cache, ignore_index=True)
            self.sensor_cache = pd.concat([self.sensor_cache[self.sensor_cache["sensor_name"] != "LIDAR_TOP"], frame_cache], ignore_index=True)
            self.synchronization_cache = pd.concat([self.synchronization_cache[self.synchronization_cache["sensor_name"] != "LIDAR_TOP"], frame_synchronization_cache], ignore_index=True)

        full_sensors = self.cam_names + self.other_sensors
        valid_sensors = sum([self.sensor_cache['sensor_name'] == sensor for sensor in full_sensors]) == 1
        self.sensor_cache = self.sensor_cache[valid_sensors]
        valid_synchronization_sensors = sum(
            [self.synchronization_cache['sensor_name'] == sensor for sensor in full_sensors]) == 1
        self.synchronization_cache = self.synchronization_cache[valid_synchronization_sensors]
        valid_list = [self.synchronization_cache[sensor].notna() for sensor in full_sensors]
        full_valid = sum(valid_list) == len(full_sensors)
        for sensor in full_sensors:
            masked = (self.synchronization_cache['sensor_name'] == sensor) & (~full_valid)
            item = self.synchronization_cache[masked].copy()
            item.insert(
                item.shape[1],
                "timestamp_ns", 
                item.loc[:, sensor].copy().apply(lambda x: x.asm8.astype(np.int64)))
            item = item[["split",  "log_id", "sensor_name", "timestamp_ns"]]
            self.sensor_cache = pd.merge(self.sensor_cache, item, indicator=True, how='outer')
            assert sum(masked) == sum(self.sensor_cache['_merge'] == 'both')
            self.sensor_cache = self.sensor_cache[self.sensor_cache['_merge'] != 'both']
            self.sensor_cache = self.sensor_cache.drop(['_merge'], axis=1)   
        self.synchronization_cache = self.synchronization_cache[full_valid]
        self.sensor_cache.set_index(["split", "log_id", "sensor_name", "timestamp_ns"], inplace=True)
        self.sensor_cache.sort_index(inplace=True)
        self.synchronization_cache.set_index(keys=["split", "log_id", "sensor_name"], inplace=True)
        self.synchronization_cache = self.synchronization_cache.loc[:, full_sensors]
        self.synchronization_cache.sort_index(inplace=True)   

        scene2sensor2stamp2token = read_json_file(self.data_dir / "nusc/scene2sensor2stamp2token.json")
        scene2sensor2intrinsic = read_json_file(self.data_dir / "nusc/scene2sensor2intrinsic.json")
        scene2sensor2extrinsic = read_json_file(self.data_dir / "nusc/scene2sensor2extrinsic.json")
        token2file = read_json_file(self.data_dir / "nusc/token2file.json")
        
        self.scene2sensor2stamp2token = {scene: {sensor: scene2sensor2stamp2token[scene][sensor] for sensor in self.sensors} for scene in self.logs}
        self.scene2sensor2intrinsic = {scene: {sensor: scene2sensor2intrinsic[scene][sensor] for sensor in self.sensors} for scene in self.logs}
        self.scene2sensor2extrinsic = {scene: {sensor: scene2sensor2extrinsic[scene][sensor] for sensor in self.sensors} for scene in self.logs}
        self.token2file = {token: token2file[token] for scene in self.logs for sensor in self.sensors for token in self.scene2sensor2stamp2token[scene][sensor].values()}
        if self.with_condition or self.with_bboxes:
            if self.class_num == 10:
                scene2stamp2annotation = read_json_file(self.data_dir / "nusc/scene2stamp2annotation.json")
            elif self.class_num == 7:
                scene2stamp2annotation = read_json_file(self.data_dir / "nusc/scene2stamp2annotation_7.json")
            else:
                raise ValueError("Unsupported class_num")
            self.scene2stamp2annotation = {log: scene2stamp2annotation[log] for log in self.logs}
        if self.with_bboxes:
            scene2stamp2velocity = read_json_file(self.data_dir / "nusc/scene2stamp2velocity.json")
            self.scene2stamp2velocity = {log: scene2stamp2velocity[log] for log in self.logs}
            token2ego = read_json_file(self.data_dir / "nusc/token2ego.json")
            self.token2ego = {token: token2ego[token] for scene in self.logs for token in self.scene2sensor2stamp2token[scene][self.refer_sensor].values()}
        if self.with_map_vectors or self.maptr_v2:
            scene2location = read_json_file(self.data_dir / "nusc/scene2location.json")
            self.scene2location = {log: scene2location[log] for log in self.logs}
            self.map_apis = {loc: NuScenesMap(dataroot=self.data_dir, map_name=loc) for loc in set(self.scene2location.values())}
            self.map2idx = {'road_divider': 0, 'lane_divider': 0, 'ped_crossing': 1, 'road_segment': 2, 'lane': 2}
            token2ego = read_json_file(self.data_dir / "nusc/token2ego.json")
            self.token2ego = {token: token2ego[token] for scene in self.logs for token in self.scene2sensor2stamp2token[scene][self.refer_sensor].values()}


        if self.temporal or self.maptr_v2:
            # Camera exposures and the reference LiDAR keyframe are asynchronous.
            # Keep poses for both so projections and temporal anchors use the SAME ego frame.
            token2ego = read_json_file(self.data_dir / "nusc/token2ego.json")
            self.token2ego = {token: token2ego[token] for scene in self.logs
                             for sensor in self.sensors
                             for token in self.scene2sensor2stamp2token[scene][sensor].values()}

    def __getitem__(self, idx: int) -> Dict:
        split, log_id, timestamp_ns = self.sensor_cache.xs(key=self.refer_sensor, level=2).iloc[idx].name
        log_sensor_records = self.sensor_cache.xs((split, log_id, self.refer_sensor)).index
        datum = {
            "idx": idx,
            "split": split,
            "log_id": log_id,
            "timestamp_ns": timestamp_ns,
            "idx_in_log":  np.where(log_sensor_records == timestamp_ns)[0].item(),
            "num_sweeps_in_log": len(log_sensor_records)}
        if self.return_bev:
            datum["latent"] = self._load_latent(log_id, timestamp_ns)
        else:
            datum["synchronized_imagery"] = self._load_synchronized_cams(log_id, timestamp_ns)
        if self.with_condition:
            datum["condition"] = self._load_conditions(log_id, timestamp_ns)
        if self.with_bboxes:
            # datum["cuboids"] = self._load_cuboids(log_id, timestamp_ns)
            datum["gt_bboxes"] = self._load_bboxes(log_id, timestamp_ns)
        if self.with_map_vectors and not self.maptr_v2:
            datum["gt_map_vectors"] = self._load_map_vectors(log_id, timestamp_ns)
        return datum
        
    def __iter__(self):
        self._ptr = 0
        return self

    def __next__(self):
        result = self.__getitem__(self._ptr)
        self._ptr += 1
        return result

    @cached_property
    def sensors(self) -> List[str]:
        return list(self.sensor_cache.index.unique("sensor_name"))

    @cached_property
    def logs(self) -> List[str]:
        return list(self.sensor_cache.index.unique("log_id"))

    @cached_property
    def num_logs(self) -> int:
        return len(self.logs)

    @cached_property
    def sensor_counts(self) -> pd.Series:
        return self.sensor_cache.index.get_level_values("sensor_name").value_counts()

    @property
    def num_sensors(self) -> int:
        return len(self.sensor_counts)

    @cached_property
    def refer_sensor(self) -> str:
        return "LIDAR_TOP" if "LIDAR_TOP" in self.sensor_counts.index else self.sensor_counts.idxmax()

    @cached_property
    def num_sweeps(self) -> int:
        return int(self.sensor_counts[self.refer_sensor])
    
    @cached_property
    def scene_points(self) -> np.ndarray:
        z = np.linspace(0.5, self.scene_size[0]-0.5, self.scene_size[0]) 
        y = np.linspace(0.5, self.scene_size[1]-0.5, self.scene_size[1]) 
        x = np.linspace(0.5, self.scene_size[2]-0.5, self.scene_size[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        scene_points = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        return scene_points
        
    @cached_property
    def pc_points(self) -> np.ndarray:
        pc_points = np.stack([
            self.scene_points[..., 0] / self.scene_size[2] * (self.pc[3] - self.pc[0]) + self.pc[0], 
            self.scene_points[..., 1] / self.scene_size[1] * (self.pc[4] - self.pc[1]) + self.pc[1], 
            self.scene_points[..., 2] / self.scene_size[0] * (self.pc[5] - self.pc[2]) + self.pc[2]], axis=-1)
        pc_points = np.concatenate([pc_points, np.ones_like(pc_points[..., :1])], axis=-1)
        return pc_points

    def __len__(self) -> int:
        return self.num_sweeps

    def find_closest_sensor(
        self,
        split: str,
        log_id: str,
        src_sensor_name: str,
        src_timestamp_ns: int,
        dst_sensor_name: str,
        ) -> Optional[Path]:
        src_timedelta_ns = pd.Timedelta(src_timestamp_ns)
        src2dst_records = self.synchronization_cache.loc[(split, log_id, src_sensor_name)].set_index(src_sensor_name, drop=False)
        assert src_timedelta_ns in src2dst_records.index
        dst_timestamp_ns = src2dst_records.loc[src_timedelta_ns, dst_sensor_name]
        assert not pd.isna(dst_timestamp_ns)
        timestamp_ns_str = str(dst_timestamp_ns.asm8.item())
        dst_token = self.scene2sensor2stamp2token[log_id][dst_sensor_name][timestamp_ns_str]
        return dst_token

    def _load_synchronized_cams(
        self, 
        log_id: str, 
        timestamp_ns: int
        ) -> Dict:
        cams = dict()
        reference_pose = None
        if self.temporal or self.maptr_v2:
            reference_token = self.scene2sensor2stamp2token[log_id][self.refer_sensor][str(timestamp_ns)]
            reference_pose = self._ego_pose_matrix(self.token2ego[reference_token])
        for cam_name in self.cam_names:
            cam_token = self.find_closest_sensor(self.split, log_id, self.refer_sensor, timestamp_ns, cam_name)
            cam_path = self.data_dir / self.token2file[cam_token]
            intrinsic = np.array(self.scene2sensor2intrinsic[log_id][cam_name])
            extrinsic_params = self.scene2sensor2extrinsic[log_id][cam_name]
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = Quaternion(extrinsic_params[:4]).rotation_matrix
            extrinsic[:3, 3] = extrinsic_params[4:]
            if reference_pose is not None:
                camera_pose = self._ego_pose_matrix(self.token2ego[cam_token])
                extrinsic = np.linalg.solve(reference_pose, camera_pose) @ extrinsic
            cams[cam_name] = {"path": cam_path.relative_to(self.data_dir).as_posix(), "img": read_img(cam_path), "intrinsic": intrinsic, "extrinsic": extrinsic}
        return cams

    @staticmethod
    def _ego_pose_matrix(params):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Quaternion(params[:4]).rotation_matrix
        matrix[:3, 3] = params[4:]
        return matrix
    
    def _load_latent(
        self, 
        log_id: str, 
        timestamp_ns: int
        ) -> Dict:
        latent = dict()
        for cam_name in self.cam_names:
            intrinsic = np.array(self.scene2sensor2intrinsic[log_id][cam_name])
            extrinsic_params = self.scene2sensor2extrinsic[log_id][cam_name]
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = Quaternion(extrinsic_params[:4]).rotation_matrix
            extrinsic[:3, 3] = extrinsic_params[4:]
            latent[cam_name] = {"intrinsic": intrinsic, "extrinsic": extrinsic}

        # latent_type = "bev-lidar" if self.with_condition or self.with_bboxes else "bev"
        latent_type = "bev-lidar" if self.with_condition or self.with_bboxes or self.with_map_vectors else "bev"
        latent_type = "bev-pre-lidar" if self.with_bboxes else latent_type
        bev_path = self.data_dir / latent_type / self.split / log_id / f"{timestamp_ns}.npy"
        latent["path"] = bev_path.relative_to(self.data_dir).as_posix()
        latent["bev"] = np.load(bev_path)
        return latent

    def _cuboid2mask(self, cuboids: List[List]) -> np.ndarray:
        scene_cuboid_mask = np.zeros((self.class_num, *self.scene_size), dtype=bool)
        for i, cuboid in enumerate(cuboids):
            cuboid_points = self.pc_points @  np.linalg.inv(quaternion2matrix(cuboid[:7])).T
            cuboid_points = cuboid_points[..., :-1]
            lwh = np.array(cuboid[7:-1])
            valid = np.all((lwh / -2.0 < cuboid_points) & (cuboid_points < lwh / 2.0), axis=-1)
            if np.sum(valid) > 0:
                valid_points = self.scene_points[valid].astype(np.int64)
                x, y, z = valid_points.T  
                scene_cuboid_mask[cuboid[-1], z, y, x] = True 
        return scene_cuboid_mask.astype(np.float32)
    
    def _load_cuboids(self, log_id: str, timestamp_ns: int) -> List[List]:
        cuboids = self.scene2stamp2annotation[log_id][str(timestamp_ns)]
        cuboids = [cuboid for cuboid in cuboids if self.pc[0] < cuboid[4] < self.pc[3] and self.pc[1] < cuboid[5] < self.pc[4]]
        return cuboids

    def _load_conditions(self, log_id: str, timestamp_ns: int) -> Dict:
        return self._cuboid2mask(self._load_cuboids(log_id, timestamp_ns))    

    def _load_bboxes(self, log_id: str, timestamp_ns: int) -> Dict:
        padded_bboxes = np.concatenate([np.zeros((256, 10)), np.ones((256, 1)) * -1], axis=-1) 
        cuboids = np.asarray(self.scene2stamp2annotation[log_id][str(timestamp_ns)], dtype=np.float32)
        velocities = np.asarray(self.scene2stamp2velocity[log_id][str(timestamp_ns)], dtype=np.float32)
        if cuboids.size == 0:
            return padded_bboxes.astype(np.float32)

        cuboids = np.atleast_2d(cuboids)
        velocities = np.atleast_2d(velocities)
        assert len(cuboids) == len(velocities)

        valid = (self.pc[0] < cuboids[:, 4]) & (cuboids[:, 4] < self.pc[3]) & (self.pc[1] < cuboids[:, 5]) & (cuboids[:, 5] < self.pc[4])
        cuboids = cuboids[valid]
        velocity = velocities[valid, :2].copy()

        # nuScenes reports NaN velocity when an annotation does not have
        # enough neighbouring samples for a reliable estimate.  Sparse4D's
        # official dataset adapter replaces those values with zero before
        # constructing regression and denoising targets.  Keep the same
        # contract here and also reject infinities defensively.
        invalid_velocity = ~np.isfinite(velocity).all(axis=-1)
        velocity[invalid_velocity] = 0.0

        if len(cuboids) > 0:
            yaw = np.stack([[(Quaternion(cuboids[i, :4])).yaw_pitch_roll[0]] for i in range(len(cuboids))])
            bboxes = np.concatenate([cuboids[:, 4:7], np.log(cuboids[:, 7:10]), np.sin(yaw), np.cos(yaw), velocity, cuboids[:, 10:11]], axis=-1)
            padded_bboxes[:len(bboxes)] = bboxes
        return padded_bboxes.astype(np.float32)
    
    def _load_map_instances(self, log_id: str, timestamp_ns: int, pc=None) -> Dict:
        nusc_map = self.map_apis[self.scene2location[log_id]]
        ego_params = self.token2ego[self.scene2sensor2stamp2token[log_id][self.refer_sensor][str(timestamp_ns)]]
        if self.maptr_v2:
            from bev_vae.data.maptr_targets import extract_map_instances
            return extract_map_instances(nusc_map, ego_params, self.pc if pc is None else pc)
        patch_box = (ego_params[4], ego_params[5] , self.pc[3] - self.pc[0], self.pc[4] - self.pc[1])
        patch_angle = Quaternion(ego_params[:4]).yaw_pitch_roll[0] / np.pi * 180
        geom_list = nusc_map.get_map_geom(patch_box, patch_angle, self.map2idx.keys())    
        idx2geom = {idx: [] for idx in set(self.map2idx.values())}
        for map, geoms in geom_list:
            idx2geom[self.map2idx[map]].extend(geoms)
        instances = []
        patch_size=(self.pc[3] - self.pc[0], self.pc[4] - self.pc[1])
        for idx, geoms in idx2geom.items():
            instance_list = geoms if idx == 0 else polygon_geoms_to_instances(geoms, patch_size)
            for inst in instance_list:
                instances.append((inst, idx))
        return filter_short_map_instances(instances)
    
    def _load_map_vectors(self, log_id: str, timestamp_ns: int) -> Dict:
        padded_vectors =  np.concatenate([np.zeros((256, 40)), np.ones((256, 2)) * -1], axis=-1) 
        instances = self._load_map_instances(log_id, timestamp_ns)
        if len(instances) == 0:
            return padded_vectors.astype(np.float32)
        vectors = shift_fixed_num_sampled_points_v2(instances, self.vertex_num, self.pc)
        padded_vectors[:len(vectors)] = vectors
        return padded_vectors.astype(np.float32)
