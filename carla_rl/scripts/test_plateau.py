"""plateau.detect assert 測試。跑: .\\.venv\\Scripts\\python.exe -m carla_rl.scripts.test_plateau"""
from carla_rl.scripts.plateau import detect

TOTAL = 100000

# 1) 持續進步:不停
steps = list(range(0, 60000, 500))
rets = [s / 100.0 for s in steps]
stop, why = detect(steps, rets, TOTAL)
assert not stop, why

# 2) 平台:20000 步後封頂、平盤到 ~60000(無新高 > 20% 總步數 = 20000)→ 停(PLATEAU)
steps = list(range(0, 60000, 500))
rets = [min(s, 20000) / 100.0 for s in steps]
stop, why = detect(steps, rets, TOTAL)
assert stop and 'PLATEAU' in why, why

# 3) 下降:到高點後持續下滑超過 15% 總步數 → 停(DECLINE 或 PLATEAU 皆可,反正該停)
steps = list(range(0, 60000, 500))
rets = [(s / 100.0 if s <= 20000 else (200 - (s - 20000) / 100.0)) for s in steps]
stop, why = detect(steps, rets, TOTAL)
assert stop, why

# 4) 樣本太少:warmup,不停
stop, why = detect([0, 100, 200], [1, 2, 3], TOTAL)
assert not stop and why == 'warmup', why

print("plateau tests PASS")
