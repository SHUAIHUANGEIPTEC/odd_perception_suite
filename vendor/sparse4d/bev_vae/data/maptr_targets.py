"""MapTRv2 targets in BEV-VAE's reference ego frame (x forward, y left).

The explicit order mask keeps padded permutations out of Hungarian matching.
Geometry routines work with both Shapely 1.x and 2.x without mutating rings.
"""
import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, linemerge

MAP_CLASSES = ("divider", "ped_crossing", "boundary")


def parts(geom, kind):
    if geom.is_empty:
        return []
    if isinstance(geom, kind):
        return [geom]
    return [p for g in getattr(geom, "geoms", ()) for p in parts(g, kind)]


def sample_line(line, count):
    return np.asarray([line.interpolate(d).coords[0][:2]
                       for d in np.linspace(0, line.length, count)], dtype=np.float32)


def extract_map_instances(map_api, ego_params, pc):
    from pyquaternion import Quaternion
    yaw = Quaternion(ego_params[:4]).yaw_pitch_roll[0]
    cx, cy = (pc[0] + pc[3]) / 2, (pc[1] + pc[4]) / 2
    world_x = ego_params[4] + np.cos(yaw) * cx - np.sin(yaw) * cy
    world_y = ego_params[5] + np.sin(yaw) * cx + np.cos(yaw) * cy
    # nuScenes API takes height (y) BEFORE width (x).
    patch = (world_x, world_y, pc[4] - pc[1], pc[3] - pc[0])
    layers = ["road_divider", "lane_divider", "ped_crossing", "road_segment", "lane"]
    geoms = dict(map_api.explorer.get_map_geom(patch, np.degrees(yaw), layers))
    geoms = {k: [affinity.translate(g, cx, cy) for g in v] for k, v in geoms.items()}
    instances = [(line, 0) for k in layers[:2] for geom in geoms[k]
                 for line in parts(geom, LineString)]
    for label, names, inset in ((1, ["ped_crossing"], -0.2),
                                (2, ["road_segment", "lane"], 0.2)):
        merged = unary_union([g for name in names for g in geoms[name]])
        roi = box(pc[0] + inset, pc[1] + inset, pc[3] - inset, pc[4] - inset)
        for polygon in parts(merged, Polygon):
            polygon = orient(polygon, sign=-1.0)  # clockwise exterior, ccw holes
            for ring in [polygon.exterior, *polygon.interiors]:
                clipped = LineString(ring.coords).intersection(roi)
                if clipped.geom_type == "MultiLineString":
                    clipped = linemerge(clipped)
                instances.extend((line, label) for line in parts(clipped, LineString))
    # Keep short lane markings; silently dropping <5m lines biases the metric.
    return [(line, label) for line, label in instances if line.length > 1e-4]


def transform_instances(instances, pc, angle=0.0):
    roi = box(pc[0], pc[1], pc[3], pc[4])
    output = []
    for line, label in instances:
        line = affinity.rotate(line, angle, origin=(0, 0), use_radians=True)
        output.extend((part, label) for part in parts(line.intersection(roi), LineString)
                      if part.length > 1e-4)
    return output


def extraction_range(pc, angle):
    """Fetch enough unaugmented geometry to cover a rotated target ROI."""
    if abs(angle) < 1e-12:
        return list(pc)
    corners = np.asarray([[pc[0], pc[1]], [pc[0], pc[4]],
                          [pc[3], pc[1]], [pc[3], pc[4]]])
    c, s = np.cos(angle), np.sin(angle)
    source = corners @ np.asarray([[c, -s], [s, c]])  # row-vector inverse rotation
    lower, upper = source.min(0), source.max(0)
    # The boundary extractor insets its patch by .2m; retain a safety margin
    # so this artificial edge cannot remove geometry entering the final ROI.
    return [lower[0]-.3, lower[1]-.3, pc[2], upper[0]+.3, upper[1]+.3, pc[5]]


def point_orders(line, count):
    """Official v2: ring vertex shifts, or both directions for open lines.

    Use evenly spaced shifts when capped instead of random validation targets.
    """
    coords = np.asarray(line.coords)[:, :2]
    if np.allclose(coords[0], coords[-1]):
        vertices = coords[:-1]
        indices = np.linspace(0, len(vertices) - 1, min(len(vertices), count - 1), dtype=int)
        return np.stack([sample_line(LineString(np.concatenate(
            (np.roll(vertices, i, axis=0), np.roll(vertices, i, axis=0)[:1]))), count) for i in indices])
    sampled = sample_line(line, count)
    return np.stack([sampled, sampled[::-1]])


def build_map_targets(instances, pc, scene_size, vertex_num=20, max_instances=128,
                      intrinsic=None, extrinsic=None, geometric=None,
                      image_shape=None, pv_stride=32, ground_height=0.0):
    if vertex_num < 3:
        raise ValueError("MapTR needs at least three points")
    if len(instances) > max_instances:
        raise ValueError(f"{len(instances)} map instances exceed map_max_instances={max_instances}; increase it")
    lo = np.asarray(pc[:2], np.float32)
    extent = np.asarray(pc[3:5], np.float32) - lo
    m, p, s = max_instances, vertex_num, vertex_num - 1
    labels = np.full(m, -1, np.int64)
    points = np.zeros((m, p, 2), np.float32)
    orders = np.zeros((m, s, p, 2), np.float32)
    order_mask = np.zeros((m, s), bool)
    legacy = np.zeros((m, 2 * p + 2), np.float32)
    legacy[:, -2:] = -1
    h, w = scene_size[-2:]
    bev = np.zeros((h, w), np.uint8)
    pv = None
    if intrinsic is not None:
        ih, iw = image_shape
        ph, pw = (ih + pv_stride - 1) // pv_stride, (iw + pv_stride - 1) // pv_stride
        pv = np.zeros((len(intrinsic), 1, ph, pw), np.uint8)
    for i, (line, label) in enumerate(instances):
        labels[i] = label
        points[i] = np.clip((sample_line(line, p) - lo) / extent, 0, 1)
        permutations = np.clip((point_orders(line, p) - lo) / extent, 0, 1)
        orders[i, :len(permutations)] = permutations
        order_mask[i, :len(permutations)] = True
        legacy[i] = np.concatenate((points[i].ravel(), [label, i]))
        dense = sample_line(line, max(2, int(np.ceil(line.length / 0.25)) + 1))
        pixels = np.rint((dense - lo) / extent * [w - 1, h - 1]).astype(np.int32)
        cv2.polylines(bev, [pixels], False, 1, thickness=3)
        if pv is not None:
            xyz = np.column_stack((dense, np.full(len(dense), ground_height), np.ones(len(dense))))
            for cam in range(len(pv)):
                camera = (np.linalg.inv(extrinsic[cam]) @ xyz.T).T[:, :3]
                # Clip each segment to the front half-space before perspective division.
                for a, b in zip(camera[:-1], camera[1:]):
                    a, b = a.copy(), b.copy()
                    if max(a[2], b[2]) <= 0.1:
                        continue
                    if a[2] < 0.1:
                        a += (b - a) * ((0.1 - a[2]) / (b[2] - a[2]))
                    if b[2] < 0.1:
                        b += (a - b) * ((0.1 - b[2]) / (a[2] - b[2]))
                    uv = (geometric[cam] @ intrinsic[cam] @ np.stack((a, b)).T).T
                    uv = uv[:, :2] / uv[:, 2:3] / pv_stride
                    uv = np.rint(np.clip(uv, -1e6, 1e6)).astype(np.int32)
                    visible, start, end = cv2.clipLine((0, 0, pw, ph), tuple(uv[0]), tuple(uv[1]))
                    if visible:
                        cv2.line(pv[cam, 0], start, end, 1, thickness=1)
    result = dict(gt_map_labels=labels, gt_map_points=points, gt_map_orders=orders,
                  gt_map_order_mask=order_mask, gt_map_vectors=legacy,
                  gt_map_bev_seg=bev[None].astype(np.float32))
    if pv is not None:
        result["gt_map_pv_seg"] = pv.astype(np.float32)
    return result
