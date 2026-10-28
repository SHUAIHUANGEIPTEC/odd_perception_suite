"""EnvMapper unit tests (no torch)."""

from tests.conftest import make_attr


def test_visibility_bands(env_mapper):
    expected = {"LT50": 40, "50_200": 120, "200_500": 350, "GE500": 2000}
    for band, meters in expected.items():
        env = env_mapper.map(make_attr(visibility=band))
        assert env["visibility_m"] == meters


def test_lighting_representatives(env_mapper):
    expected = {"Day": 5000, "NightLit": 150, "DawnDusk": 80, "NightUnlit": 20}
    for cls, lux in expected.items():
        env = env_mapper.map(make_attr(lighting=cls))
        assert env["illumination_lux"] == lux


def test_baseline_when_clear_weather(env_mapper):
    env = env_mapper.map(make_attr(weather=["Sunny", "Cloudy"]))
    assert env["rain_mm_per_h"] == 0.0
    assert env["snow_level"] == "none"
    assert env["fog"] is False


def test_rain_sets_precip(env_mapper):
    env = env_mapper.map(make_attr(weather=["Rain"]))
    assert env["rain_mm_per_h"] == 4.0
    assert env["fog"] is False


def test_snow_conservative_default(env_mapper):
    env = env_mapper.map(make_attr(weather=["Snow"]))
    assert env["snow_level"] == "moderate"


def test_fog_flag_and_visibility_clamp(env_mapper):
    env = env_mapper.map(make_attr(weather=["Fog"], visibility="GE500"))
    assert env["fog"] is True
    assert env["visibility_m"] == 350  # fog resolves the GE500 contradiction down


def test_haze_dust_clamp(env_mapper):
    env = env_mapper.map(make_attr(weather=["Haze"], visibility="GE500"))
    assert env["visibility_m"] == 200
    env = env_mapper.map(make_attr(weather=["Dust"], visibility="200_500"))
    assert env["visibility_m"] == 200


def test_multi_label_weather_triggers_independently(env_mapper):
    env = env_mapper.map(make_attr(weather=["Rain", "Fog", "Snow"], visibility="LT50"))
    assert env["rain_mm_per_h"] == 4.0
    assert env["snow_level"] == "moderate"
    assert env["fog"] is True
    assert env["visibility_m"] == 40  # LT50 already below the fog clamp


def test_debug_exposes_road_tasks(env_mapper):
    dbg = env_mapper.debug(make_attr(road_condition="Wet", road_material="Concrete"))
    assert dbg == {"road_condition": "Wet", "road_material": "Concrete"}
