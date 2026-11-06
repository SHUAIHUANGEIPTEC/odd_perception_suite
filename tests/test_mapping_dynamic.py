"""DynamicRollup / SensorDefaults unit tests (no torch)."""

from tests.conftest import make_det


def test_rollup_vru_and_two_wheeler(rollup):
    dets = [make_det("pedestrian"), make_det("pedestrian"), make_det("bicycle")]
    fields = rollup.map(dets)
    assert fields["pedestrians_present"] is True
    assert fields["two_wheelers_present"] is True
    assert fields["animals_over_0_3m"] == 0


def test_rollup_obstacles_are_static_only(rollup):
    dets = [make_det("traffic_cone")] * 5 + [make_det("barrier")] + [make_det("car")] * 3
    fields = rollup.map(dets)
    assert fields["obstacles_over_0_3m"] == 6  # cones + barrier; cars excluded
    assert fields["pedestrians_present"] is False


def test_traffic_vehicles_never_count_as_obstacles(rollup):
    dets = [make_det(c) for c in ("car", "truck", "bus", "trailer", "construction_vehicle")]
    fields = rollup.map(dets)
    assert fields["obstacles_over_0_3m"] == 0


def test_score_threshold_filters(rollup):
    dets = [make_det("pedestrian", score=0.2), make_det("traffic_cone", score=0.9)]
    fields = rollup.map(dets)
    assert fields["pedestrians_present"] is False
    assert fields["obstacles_over_0_3m"] == 1


def test_empty_detections(rollup):
    fields = rollup.map([])
    assert fields == {
        "pedestrians_present": False,
        "two_wheelers_present": False,
        "obstacles_over_0_3m": 0,
        "animals_over_0_3m": 0,
        "surrounding_vehicles": 0,
    }


def test_surrounding_vehicles_counts_traffic_role(rollup):
    dets = [make_det("car"), make_det("truck"), make_det("pedestrian")]
    fields = rollup.map(dets)
    assert fields["surrounding_vehicles"] == 2
    assert fields["obstacles_over_0_3m"] == 0


def test_unknown_class_ignored_by_default(rollup):
    fields = rollup.map([make_det("spaceship")])
    assert fields["obstacles_over_0_3m"] == 0


def test_sensors_from_detection_mean(sensors):
    dets = [make_det("car", score=0.8), make_det("car", score=0.6)]
    fields = sensors.map(dets)
    assert abs(fields["confidence_camera"] - 0.7) < 1e-9
    assert fields["camera_degraded"] is False


def test_sensors_degraded_when_low_scores(sensors):
    dets = [make_det("car", score=0.1), make_det("car", score=0.2)]
    fields = sensors.map(dets)
    assert fields["camera_degraded"] is True
    assert fields["lidar_degraded"] is True


def test_sensors_defaults_without_detections(sensors):
    fields = sensors.map(None)
    assert fields["confidence_camera"] == 0.95
    assert fields["confidence_lidar"] == 0.92
    assert fields["camera_degraded"] is False
