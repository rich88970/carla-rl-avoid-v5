"""Default EasyCarla-RL environment parameters, shared by all scripts.

Matches the upstream demo (easycarla_demo.py) except where noted.

所有腳本共用的環境預設參數(集中管理,改條件走 make_params,不要各處 inline dict)。
關鍵:number_of_walkers 固定 0(此機源碼版生行人會 SkeletalMesh abort 當機);traffic='off'
代表紅綠燈凍結綠燈(資料集條件);desired_speed 為速度追蹤目標(m/s)。
"""

DEFAULT_PARAMS = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,  # keep 0: walker spawning crashed the local source-built server (SkeletalMesh assert)
    'dt': 0.1,
    'ego_vehicle_filter': 'vehicle.tesla.model3',
    'surrounding_vehicle_spawned_randomly': True,
    'port': 2000,
    'town': 'Town03',
    'max_time_episode': 1000,
    'max_waypoints': 12,
    'visualize_waypoints': False,  # off by default: debug points pollute recorded videos
    'desired_speed': 8,  # m/s
    'max_ego_spawn_times': 200,
    'view_mode': 'follow',
    'traffic': 'off',  # 'off' = all lights frozen green; flip to 'on' for traffic-light training
    'lidar_max_range': 50.0,
    'max_nearby_vehicles': 5,
}


def make_params(**overrides):
    params = dict(DEFAULT_PARAMS)
    params.update(overrides)
    return params
