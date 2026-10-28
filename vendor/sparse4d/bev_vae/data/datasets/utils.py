from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List
import json
import pandas as pd
import numpy as np
import cv2
from pyarrow import feather
from pyquaternion import Quaternion
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from bev_vae.data.transforms import denormalize
import math
from shapely.geometry import box, LineString, MultiLineString, MultiPolygon, Polygon
from shapely import ops
from io import BytesIO


def read_feather(
    path: Path, columns: Optional[Tuple[str, ...]] = None
) -> pd.DataFrame:

    with path.open("rb") as file_handle:
        dataframe: pd.DataFrame = feather.read_feather(
            file_handle, columns=columns, memory_map=True
        )
    return dataframe


def read_json_file(fpath: Path) -> Dict[str, Any]:
    with fpath.open("rb") as f:
        data: Dict[str, Any] = json.load(f)
        return data

def read_img(img_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def matrix2quaternion(matrix: np.ndarray) -> List[float]:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    # quaternion = Quaternion(matrix=rotation, atol=1e-07)
    quaternion = Quaternion(matrix=rotation, atol=1e-05)
    params = quaternion.q.tolist() + translation.tolist()
    return params

def quaternion2matrix(params: List[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(params[:4]).rotation_matrix
    matrix[:3, 3] = params[4:]
    return matrix.astype(np.float32)

def extract_matrix_xy(matrix):
    rotation = matrix[:3, :3]
    z_angle = np.arctan2(rotation[1, 0], rotation[0, 0])
    rotation_xy = np.array([[np.cos(z_angle), -np.sin(z_angle)], [np.sin(z_angle),  np.cos(z_angle)]])
    translation_xy = matrix[:2, 3]
    matrix_xy = np.eye(3)
    matrix_xy[:2, :2] = rotation_xy
    matrix_xy[:2, 2] = translation_xy
    return matrix_xy.astype(np.float32)

def extract_matrix_xy_3d(matrix):
    rotation = matrix[:3, :3]
    z_angle = np.arctan2(rotation[1, 0], rotation[0, 0])
    rotation_xy = np.array([[np.cos(z_angle), -np.sin(z_angle)], [np.sin(z_angle),  np.cos(z_angle)]])
    translation_xyz = matrix[:3, 3]
    matrix_xy_3d = np.eye(4)
    matrix_xy_3d[:2, :2] = rotation_xy
    matrix_xy_3d[:3, 3] = translation_xyz
    return matrix_xy_3d.astype(np.float32)

def yaw2quaternion(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw)
    yaw = np.atleast_2d(yaw)
    half_yaw = yaw / 2
    w = np.cos(half_yaw)
    x = np.zeros_like(half_yaw)
    y = np.zeros_like(half_yaw)
    z = np.sin(half_yaw)
    quat = np.concatenate([w, x, y, z], axis=-1)
    return quat

def draw_box(draw: ImageDraw.Draw, corners, color, width=1):
    for i in range(4):
        draw.line((*corners[i], *corners[i + 4]), fill=color, width=width)

    def draw_rect(face, width):
        prev = face[-1]
        for curr in face:
            draw.line((*prev, *curr), fill=color, width=width)
            prev = curr

    draw_rect(corners[:4], width)
    draw_rect(corners[4:], width)

    center_bottom_forward = np.mean(corners[2:4], axis=0)
    center_bottom = np.mean(corners[[2, 3, 7, 6]], axis=0)
    draw.line((*center_bottom, *center_bottom_forward), fill=color, width=width)


def render_cuboids(cuboids, x, img_metas, eps=1e-6, only_xy_rotation=True):
    pil_imgs = [F.to_pil_image(img) for img in denormalize(x)]
    draw_objs = [ImageDraw.Draw(img) for img in pil_imgs]
    for b in range(len(cuboids)):
        for cuboid in cuboids[b]:
            rgba = plt.cm.jet(cuboid[-1].cpu().numpy() / 10)  # 用最后一维作为颜色权重
            color = tuple([int(255 * c) for c in rgba[:3]])
            extrinsic = img_metas['extrinsic'][b]
            intrinsic = img_metas['intrinsic'][b]
            geometric = img_metas['geometric'][b]
            l, w, h = cuboid[7:-1]
            x_corners = l / 2 * torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], device=x.device)
            y_corners = w / 2 * torch.tensor([1, -1, -1, 1, 1, -1, -1, 1], device=x.device)
            z_corners = h / 2 * torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], device=x.device)
            corners = torch.stack([x_corners, y_corners, z_corners], dim=-1)
            corners = torch.cat([corners, torch.ones_like(corners[..., :1])], dim=-1)

            quaternion = cuboid[:7].cpu().numpy()
            matrix = extract_matrix_xy_3d(quaternion2matrix(quaternion)) if only_xy_rotation else quaternion2matrix(quaternion)
            matrix = torch.tensor(matrix, device=x.device)

            corners = corners @ matrix.T
            corners = torch.inverse(extrinsic)[:, None] @ corners[..., None]
            corners = corners[..., 0]

            corners = intrinsic[:, None] @ corners[..., :3, None]
            valid_z = corners.squeeze(-1)[..., -1] > eps
            corners = geometric[:, None] @ corners
            corners = corners[..., 0]
            corners = corners[..., :2] / torch.maximum(corners[..., -1:], torch.ones_like(corners[..., -1:]) * eps)

            valid_x = (0. < corners[..., 0]) & (corners[..., 0] < img_metas['img_shape'][1])
            valid_y = (0. < corners[..., 1]) & (corners[..., 1] < img_metas['img_shape'][0])
            valid = valid_x & valid_y & valid_z

            for cam_id in range(img_metas['cam_num']):
                if torch.sum(valid[cam_id]) >= 2:
                    draw_box(draw_objs[b*img_metas['cam_num'] + cam_id], corners[cam_id].cpu().numpy(), color)
    return pil_imgs


def concat_images(images):
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    canvas = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        canvas.paste(img, (x_offset, 0))
        x_offset += img.width
    return canvas

def bbox2cuboid(bboxes: torch.Tensor) -> torch.Tensor:
    yaw = torch.atan2(bboxes[..., 6:7], bboxes[..., 7:8])
    half_yaw = yaw / 2
    w = torch.cos(half_yaw)
    x = torch.zeros_like(half_yaw)
    y = torch.zeros_like(half_yaw)
    z = torch.sin(half_yaw)
    q = torch.cat([w, x, y, z], dim=-1)
    # cuboids = torch.cat([q, bboxes[..., :3], bboxes[..., 3:6].exp(), bboxes[..., -1:]], dim=-1)
    cuboids = torch.cat([q, bboxes[..., :3], bboxes[..., 3:6].exp(), bboxes[..., 10:11]], dim=-1)
    return cuboids

def concat_vertical(images):
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    canvas = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for img in images:
        canvas.paste(img, (0, y_offset))
        y_offset += img.height
    return canvas


def line_geom_to_instances(line_geom):
    instances = []
    for line in line_geom:
        if line.is_empty:
            continue
        if line.geom_type == 'MultiLineString':
            for single_line in line.geoms:
                if not single_line.is_empty:
                    instances.append(single_line)
        elif line.geom_type == 'LineString':
            instances.append(line)
        else:
            raise NotImplementedError(f"Unsupported geometry type: {line.geom_type}")

    return instances


def polygon_geoms_to_instances(polygon_geom, patch_size, eps=0.2):
    union_segments = ops.unary_union(polygon_geom)
    if union_segments.is_empty:
        return []

    if union_segments.geom_type != 'MultiPolygon':
        union_segments = MultiPolygon([union_segments])

    max_x = patch_size[1] / 2
    max_y = patch_size[0] / 2
    local_patch = box(-max_x + eps, -max_y + eps, max_x - eps, max_y - eps)

    exteriors, interiors, lines = [], [], []
    for poly in union_segments.geoms:
        exteriors.append(poly.exterior)
        interiors.extend(list(poly.interiors))

    for ext in exteriors:
        line = LineString(list(ext.coords)[::-1] if ext.is_ccw else list(ext.coords))
        clipped = line.intersection(local_patch)
        if clipped.is_empty:
            continue
        if isinstance(clipped, MultiLineString):
            clipped = ops.linemerge(clipped)
        lines.append(clipped)

    for inter in interiors:
        line = LineString(list(inter.coords) if inter.is_ccw else list(inter.coords)[::-1])
        clipped = line.intersection(local_patch)
        if clipped.is_empty:
            continue
        if isinstance(clipped, MultiLineString):
            clipped = ops.linemerge(clipped)
        lines.append(clipped)
    return line_geom_to_instances(lines)


def filter_short_map_instances(instances, min_length=5.0):
    return [
        (line, idx)
        for line, idx in instances
        if isinstance(line, LineString)
        and not (len(line.coords) == 2 and line.length < min_length)
    ]


def shift_fixed_num_sampled_points_v2(instance_list, vertex_num, pc):
    instances = []
    for i, (instance, c) in enumerate(instance_list):
        distances = np.linspace(0, instance.length, vertex_num)
        poly_pts = np.array(list(instance.coords))
        start_pts, end_pts = poly_pts[0], poly_pts[-1]
        is_poly = np.allclose(start_pts, end_pts)
        shift_pts_list = []
        shift_num = poly_pts.shape[0] - 1
        final_shift_num = vertex_num - 1
        label = np.array([c, i])
        if is_poly:
            pts_to_shift = poly_pts[:-1,:]
            for shift_right_i in range(shift_num):
                shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                shift_pts = np.concatenate((shift_pts, shift_pts[:1]), axis=0)
                shift_instance = LineString(shift_pts)
                sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                sampled_points[:, 0] = (sampled_points[:, 0] - pc[0]) / (pc[3] - pc[0])
                sampled_points[:, 1] = (sampled_points[:, 1] - pc[1]) / (pc[4] - pc[1])
                sampled_points = np.clip(sampled_points, 0.0, 1.0)
                shift_pts_list.append(np.concatenate([sampled_points.flatten(), label]))
        else:
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            sampled_points[:, 0] = (sampled_points[:, 0] - pc[0]) / (pc[3] - pc[0])
            sampled_points[:, 1] = (sampled_points[:, 1] - pc[1]) / (pc[4] - pc[1])
            sampled_points = np.clip(sampled_points, 0.0, 1.0)
            flip_sampled_points = np.flip(sampled_points, axis=0)
            shift_pts_list.append(np.concatenate([sampled_points.flatten(), label]))
            shift_pts_list.append(np.concatenate([flip_sampled_points.flatten(), label]))

        multi_shifts_pts = np.stack(shift_pts_list, axis=0)
        if multi_shifts_pts.shape[0] > final_shift_num:
            index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
            multi_shifts_pts = multi_shifts_pts[index]
        instances.append(multi_shifts_pts)
    instances = np.concatenate(instances, axis=0)
    return instances


def extract_label_vector_ordered_vertex(map_vectors, vertex_num):
    # map_vectors: (B, vector_num, vertex_num*2 + 2) with last two dims for label and instance id
    valid = map_vectors[..., -1] != -1 # B model.vector_num or padded_vector_num
    instances_list = [map_vectors[i, valid[i], -1].to(torch.int64) for i in range(valid.size(0))]
    labels_list = [map_vectors[i, valid[i], -2].to(torch.int64) for i in range(valid.size(0))]
    vectors_list = [map_vectors[i, valid[i], :-2] for i in range(valid.size(0))]
    vector_labels, vector_ordered_vertexes = [], []
    for i in range(valid.size(0)):
        if instances_list[i].numel() == 0:
            vector_labels.append(torch.empty((0,), device=valid.device, dtype=torch.int64))
            vector_ordered_vertexes.append(torch.empty((0, vertex_num, vertex_num * 2), device=valid.device))
            continue
        vector_num = instances_list[i].max() + 1
        vector_label = torch.ones((vector_num), device=valid.device, dtype=torch.int64) * -1
        vector_ordered_vertex = torch.ones((vector_num, vertex_num, vertex_num * 2), device=valid.device) * -1
        for j in range(vector_num):
            instance_mask = instances_list[i] == j
            vector_vertex = vectors_list[i][instance_mask]
            vector_label[j] = labels_list[i][instance_mask][0]
            vector_ordered_vertex[j, :vector_vertex.size(0)] = vector_vertex
        vector_labels.append(vector_label)
        vector_ordered_vertexes.append(vector_ordered_vertex)
    return vector_labels, vector_ordered_vertexes


def plot_map_instances(instances, category=3, linewidth=2, figsize=(8, 8)):
    cmap = plt.get_cmap("tab20")
    color_map = {i: cmap(i % 20) for i in range(category)}
    fig, ax = plt.subplots(figsize=figsize)
    for geom, cat in instances:
        x, y = geom.xy
        x, y = -np.array(y), np.array(x)
        plt.plot(x, y, color=color_map[cat], linewidth=linewidth)
        plt.scatter(x, y, color=color_map[cat], s=10)

    plt.xlim(-51.2, 51.2)
    plt.ylim(-51.2, 51.2)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)  
    buf.seek(0)
    img = Image.open(buf)
    return img.resize((512, 512))


def plot_maps(map_vectors, vertex_num, pc=None):
    map_imgs = []
    for i in range(map_vectors.size(0)):
        vectors = map_vectors[i]
        vectors = vectors[vectors[..., -1] != -1]
        vertexes = vectors[..., :-2].view(-1, vertex_num, 2)
        if pc is not None:
            vertexes[..., 0:1] = vertexes[..., 0:1] * (pc[3] - pc[0]) + pc[0]
            vertexes[..., 1:2] = vertexes[..., 1:2] * (pc[4] - pc[1]) + pc[1]
        labels = vectors[..., -2].to(torch.int64)
        pd_map_instances = []
        for j in range(vertexes.shape[0]):
            geom = LineString(vertexes[j].detach().cpu().numpy())
            pd_map_instances.append((geom, labels[j].item()))
        map_imgs.append(plot_map_instances(pd_map_instances))
    return map_imgs
