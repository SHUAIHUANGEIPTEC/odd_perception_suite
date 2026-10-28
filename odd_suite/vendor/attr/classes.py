"""Task class taxonomy, copied verbatim from odd_multitask_yolo/data.py.

Kept torch-free so the mapping/pipeline core can import it without heavy deps.
"""

CLASSES = {
    "weather": ["Sunny", "Cloudy", "Rain", "Snow", "Fog", "Haze", "Dust"],
    "visibility": ["LT50", "50_200", "200_500", "GE500"],
    "lighting": ["Day", "NightLit", "DawnDusk", "NightUnlit"],
    "road_condition": ["Dry", "Wet", "Snow-covered", "Other"],
    "road_material": ["Asphalt", "Concrete", "Cobblestone", "Other"],
}
