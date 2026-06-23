"""車流密度課程解析的 assert 測試(純函式,無需 CARLA)。
跑:  .\.venv\Scripts\python.exe -m carla_rl.scripts.test_curriculum_parse
"""
from carla_rl.scripts.train_sac import parse_curriculum, curriculum_vehicles

p = parse_curriculum("0:0,30000:50,60000:100")
assert p == [(0, 0), (30000, 50), (60000, 100)]
assert curriculum_vehicles(p, 0) == 0
assert curriculum_vehicles(p, 29999) == 0
assert curriculum_vehicles(p, 30000) == 50
assert curriculum_vehicles(p, 59999) == 50
assert curriculum_vehicles(p, 100000) == 100
# 亂序輸入也要排序正確
assert parse_curriculum("60000:100,0:0,30000:50") == [(0, 0), (30000, 50), (60000, 100)]

print("curriculum parse tests PASS")
