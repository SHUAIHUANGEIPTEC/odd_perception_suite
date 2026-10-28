from typing import Any, Dict

import numpy as np
import torch


def bbox2result(bboxes: torch.Tensor, egos=None) -> torch.Tensor:
    xyz = bboxes[..., :3]
    wlh = bboxes[..., 3:6].exp()
    sin = bboxes[..., 6:7]
    cos = bboxes[..., 7:8]
    v2 = bboxes[..., 8:10]
    cls = bboxes[..., 10:11]

    if egos is not None:
        rot = egos[:, :2, :2]
        trans = egos[:, :2, 2]
        xyz = torch.cat([xyz[..., :2] @ rot.transpose(1, 2) + trans[:, None], xyz[..., 2:3]], dim=-1)
        v2 = v2 @ rot.transpose(1, 2)
        heading = torch.cat([cos, sin], dim=-1) @ rot.transpose(1, 2)
        yaw = torch.atan2(heading[..., 1:], heading[..., :1])
    else:
        yaw = torch.atan2(sin, cos)

    return torch.cat([xyz, wlh, yaw, v2, cls], dim=-1)


def filter_bboxes_by_range(bboxes: torch.Tensor, idx2range: Dict[int, float]) -> torch.Tensor:
    bboxes = bboxes.clone()
    cls = bboxes[..., -1].long()
    max_cfg_idx = max(idx2range.keys()) if idx2range else -1
    max_tensor_idx = int(cls.max().item()) if cls.numel() > 0 else -1
    max_idx = max(max_cfg_idx, max_tensor_idx)
    max_dist = torch.full((max_idx + 2,), float("inf"), device=bboxes.device, dtype=bboxes.dtype)
    for idx, class_range in idx2range.items():
        max_dist[idx] = class_range

    dist = torch.norm(bboxes[..., :2], dim=-1)
    mask = (cls >= 0) & (dist >= max_dist[cls])
    bboxes[..., -1] = cls.masked_fill(mask, -1).to(bboxes.dtype)
    return bboxes


def center_distance(pd_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    return torch.norm(pd_bbox[:2] - gt_bbox[:2])


def velocity_l2(pd_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    return torch.norm(pd_bbox[7:9] - gt_bbox[7:9])


def scale_iou(pd_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    intersection = torch.prod(torch.minimum(pd_bbox[3:6], gt_bbox[3:6]))
    pd_volume = torch.prod(pd_bbox[3:6])
    gt_volume = torch.prod(gt_bbox[3:6])
    return intersection / (pd_volume + gt_volume - intersection).clamp(min=1e-6)


def yaw_diff(pd_bbox: torch.Tensor, gt_bbox: torch.Tensor, period: float) -> torch.Tensor:
    diff = (pd_bbox[6] - gt_bbox[6] + period / 2) % period - period / 2
    return torch.abs(diff)


def attr_acc(pd_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    pd_attr = torch.norm(pd_bbox[7:9], p=2) > 0.2
    gt_attr = torch.norm(gt_bbox[7:9], p=2) > 0.2
    return (pd_attr == gt_attr).float()


def cummean_nan(values: np.ndarray) -> np.ndarray:
    if np.isnan(values).all():
        return np.ones(len(values))
    sum_values = np.nancumsum(values.astype(float))
    count_values = np.cumsum(~np.isnan(values))
    return np.divide(sum_values, count_values, out=np.zeros_like(sum_values), where=count_values != 0)


def calc_ap(precision: np.ndarray, min_recall_ind: int, min_precision: float) -> float:
    precision = precision[min_recall_ind:]
    precision = precision - min_precision
    precision[precision < 0] = 0
    return float(precision.mean() / (1.0 - min_precision))


def calc_tp(metric: np.ndarray, min_recall_ind: int, max_recall_ind: int) -> float:
    if max_recall_ind < min_recall_ind:
        return 1.0
    return float(metric[min_recall_ind : max_recall_ind + 1].mean())


def _default_tp_errors() -> Dict[str, float]:
    return {
        "trans_err": 1.0,
        "scale_err": 1.0,
        "orient_err": 1.0,
        "vel_err": 1.0,
        "attr_err": 1.0,
    }


def _recall_indices(confidence: np.ndarray, eval_det_cfg: Dict[str, Any]) -> tuple[int, int]:
    nelem = int(eval_det_cfg["nelem"])
    min_recall_ind = int(round((nelem - 1) * eval_det_cfg["min_recall"])) + 1
    non_zero = np.nonzero(confidence)[0]
    max_recall_ind = int(non_zero[-1]) if len(non_zero) > 0 else 0
    return min_recall_ind, max_recall_ind


def _interpolate_pr(
    confidence: np.ndarray,
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    gt_count: int,
    eval_det_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    tp = np.cumsum(true_positive).astype(float)
    fp = np.cumsum(false_positive).astype(float)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(max(gt_count, 1))
    recall_interp = np.linspace(0, 1, int(eval_det_cfg["nelem"]))
    precision_interp = np.interp(recall_interp, recall, precision, right=0)
    confidence_interp = np.interp(recall_interp, recall, confidence, right=0)
    return precision_interp, confidence_interp


def _calc_class_dist_metrics(
    pd_bboxes_list: list[torch.Tensor],
    pd_scores_list: list[torch.Tensor],
    gt_bboxes_list: list[torch.Tensor],
    gt_count: int,
    class_name: str,
    dist_th: float,
    eval_det_cfg: Dict[str, Any],
) -> Dict[str, float]:
    true_positive = []
    false_positive = []
    confidence = []
    match_confidence = []
    match_data = {key: [] for key in _default_tp_errors()}

    for pd_bboxes, pd_scores, gt_bboxes in zip(pd_bboxes_list, pd_scores_list, gt_bboxes_list):
        if pd_scores.numel() > 1:
            assert torch.all(pd_scores[:-1] >= pd_scores[1:]), "PD scores are not sorted in descending order!"

        if pd_bboxes.numel() == 0:
            continue

        if gt_bboxes.numel() == 0:
            dist = pd_bboxes.new_empty((pd_bboxes.shape[0], 0))
        else:
            dist = torch.norm(pd_bboxes[:, None, :2] - gt_bboxes[None, :, :2], dim=-1)

        for pd_idx in range(pd_bboxes.shape[0]):
            confidence.append(pd_scores[pd_idx])
            if dist.shape[1] == 0:
                is_match = False
            else:
                min_dist, gt_idx = torch.min(dist[pd_idx], dim=0)
                is_match = min_dist < dist_th

            if not is_match:
                true_positive.append(0)
                false_positive.append(1)
                continue

            dist[:, gt_idx] = torch.inf
            true_positive.append(1)
            false_positive.append(0)
            pd_bbox = pd_bboxes[pd_idx]
            gt_bbox = gt_bboxes[gt_idx]
            match_confidence.append(pd_scores[pd_idx])
            match_data["trans_err"].append(center_distance(pd_bbox, gt_bbox))
            match_data["scale_err"].append(1 - scale_iou(pd_bbox, gt_bbox))

            period = torch.pi if class_name == "barrier" else 2 * torch.pi
            if class_name == "traffic_cone":
                match_data["orient_err"].append(torch.tensor(torch.nan, device=pd_bboxes.device))
            else:
                match_data["orient_err"].append(yaw_diff(pd_bbox, gt_bbox, period))

            if class_name in ["barrier", "traffic_cone"]:
                match_data["vel_err"].append(torch.tensor(torch.nan, device=pd_bboxes.device))
                match_data["attr_err"].append(torch.tensor(torch.nan, device=pd_bboxes.device))
            else:
                match_data["vel_err"].append(velocity_l2(pd_bbox, gt_bbox))
                match_data["attr_err"].append(1 - attr_acc(pd_bbox, gt_bbox))

    if len(confidence) == 0:
        return {"ap": 0.0, **_default_tp_errors()}

    confidence_np = torch.stack(confidence).detach().cpu().numpy()
    order = np.argsort(confidence_np)[::-1]
    confidence_np = confidence_np[order]
    true_positive_np = np.asarray(true_positive)[order]
    false_positive_np = np.asarray(false_positive)[order]

    precision, confidence_interp = _interpolate_pr(
        confidence_np,
        true_positive_np,
        false_positive_np,
        gt_count,
        eval_det_cfg,
    )
    min_recall_ind, max_recall_ind = _recall_indices(confidence_interp, eval_det_cfg)
    metrics = {
        "ap": calc_ap(precision, min_recall_ind, eval_det_cfg["min_precision"]),
    }

    if len(match_confidence) == 0:
        metrics.update(_default_tp_errors())
        return metrics

    match_confidence_np = torch.stack(match_confidence).detach().cpu().numpy()
    match_order = np.argsort(match_confidence_np)[::-1]
    match_confidence_np = match_confidence_np[match_order]
    for metric_name, values in match_data.items():
        values_np = torch.stack(values).detach().cpu().numpy()[match_order]
        metric = cummean_nan(values_np)
        metric_interp = np.interp(confidence_interp[::-1], match_confidence_np[::-1], metric[::-1])[::-1]
        metrics[metric_name] = calc_tp(metric_interp, min_recall_ind, max_recall_ind)
    return metrics


def calc_det_metrics(
    total_pd_bboxes: torch.Tensor,
    total_gt_bboxes: torch.Tensor,
    eval_det_cfg: Dict[str, Any],
) -> Dict[str, float]:
    idx2cat = {int(idx): cat for idx, cat in eval_det_cfg["idx2cat"].items()}
    idx2range = {idx: eval_det_cfg["class_range"][cat] for idx, cat in idx2cat.items()}

    total_pd_scores = total_pd_bboxes[..., -1]
    total_pd_bboxes = bbox2result(total_pd_bboxes[..., :-1])
    total_gt_bboxes = bbox2result(total_gt_bboxes)
    total_pd_bboxes = filter_bboxes_by_range(total_pd_bboxes, idx2range)
    total_gt_bboxes = filter_bboxes_by_range(total_gt_bboxes, idx2range)

    cat2dist2metric = {}
    for idx in sorted(idx2range):
        class_name = idx2cat[idx]
        cat2dist2metric[class_name] = {}

        pd_cls_mask_list = [bboxes[:, -1].long() == idx for bboxes in total_pd_bboxes]
        pd_bboxes_list = [bboxes[mask] for bboxes, mask in zip(total_pd_bboxes, pd_cls_mask_list)]
        pd_scores_list = [scores[mask] for scores, mask in zip(total_pd_scores, pd_cls_mask_list)]

        gt_cls_mask_list = [bboxes[:, -1].long() == idx for bboxes in total_gt_bboxes]
        gt_bboxes_list = [bboxes[mask] for bboxes, mask in zip(total_gt_bboxes, gt_cls_mask_list)]
        gt_count = int(sum(mask.sum().item() for mask in gt_cls_mask_list))

        for dist_th in eval_det_cfg["dist_ths"]:
            cat2dist2metric[class_name][dist_th] = _calc_class_dist_metrics(
                pd_bboxes_list,
                pd_scores_list,
                gt_bboxes_list,
                gt_count,
                class_name,
                dist_th,
                eval_det_cfg,
            )

    cls_aps = {
        f"AP_{cat}": float(np.mean([metric["ap"] for metric in dist2metric.values()]))
        for cat, dist2metric in cat2dist2metric.items()
    }
    mean_ap = float(np.mean(list(cls_aps.values()))) if cls_aps else 0.0
    tp_errors = {
        metric: float(np.mean([dist2metric[eval_det_cfg["dist_th_tp"]][error] for dist2metric in cat2dist2metric.values()]))
        for error, metric in eval_det_cfg["error2metric"].items()
    }
    tp_scores = {metric: max(1 - value, 0.0) for metric, value in tp_errors.items()}
    nd_score = float(eval_det_cfg["mean_ap_weight"] * mean_ap + np.sum(list(tp_scores.values())))
    nd_score = nd_score / float(eval_det_cfg["mean_ap_weight"] + len(tp_scores))
    return {"mAP": mean_ap, "NDS": nd_score, **tp_errors, **cls_aps}
