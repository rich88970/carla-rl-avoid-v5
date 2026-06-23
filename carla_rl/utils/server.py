"""Manage the local CARLA server process.

The source-built server intermittently dies with a UE4 SkeletalMesh assert
during actor churn (see carla_rl/README.md), so long runs must be able to
restart it and reconnect.
"""

import os
import socket
import subprocess
import time

# The official packaged 0.9.15 build was tried (2026-06-11) and CANNOT load
# Town03 on this machine: its 2023-cooked shaders fail fatally under NVIDIA
# driver 596.36 on DX12, DX11, and Vulkan alike ("Shader compilation failures
# are Fatal" / access violations). The source build works because its shaders
# were compiled locally — so it stays the primary server, with the intermittent
# SkeletalMesh assert handled by restart_server / the evaluator's retry loop.
#
# 伺服器路徑一律由環境變數提供(不留任何本機絕對路徑;見 INSTALL.md):
#   CARLA_UE4_EDITOR — UE4Editor.exe 完整路徑
#   CARLA_UPROJECT   — CarlaUE4.uproject 完整路徑
def require_env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'缺少必要環境變數 {name},請依 INSTALL.md 設定。')
    return value


SERVER_PROCESS_NAMES = (
    'UE4Editor',
    'CarlaUE4',
    'CarlaUE4-Win64-Shipping',
    'CrashReportClient',
    'UnrealCEFSubProcess',
)


def is_port_open(port=2000, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        return sock.connect_ex((host, port)) == 0


def kill_server():
    for name in SERVER_PROCESS_NAMES:
        subprocess.run(
            ['taskkill', '/F', '/IM', f'{name}.exe'],
            capture_output=True,
        )


def launch_server(port=2000, offscreen=False, town='Town03', nullrhi=False):
    # Boot directly into the training town: the crash-prone moment for this
    # server is the load_world map switch right after boot, and the env's
    # world-reuse patch skips load_world when the town is already up.
    # -unattended: crash without a modal dialog so the supervisor can act.
    # nullrhi: no rendering at all — the SkeletalMesh render assert that kills
    # the server cannot fire, and training does not need rendering (LiDAR and
    # collision are physics-based). Cameras/videos need a rendered server.
    ue4_editor = require_env_path('CARLA_UE4_EDITOR')
    uproject = require_env_path('CARLA_UPROJECT')
    # 畫質等級:預設 Low(穩、省 GPU);高畫質錄影時設 CARLA_QUALITY=Epic(更好的材質/陰影/光照,
    # 但 GPU 負載大增、高車流密度下更易當機)。
    quality = os.environ.get('CARLA_QUALITY', 'Low')
    args = [ue4_editor, uproject, f'/Game/Carla/Maps/{town}', '-game']
    args += [
        f'-carla-rpc-port={port}',
        f'-quality-level={quality}',
        '-nosound',
        '-unattended',
        '-NoCrashDialog',
    ]
    if nullrhi:
        args.append('-nullrhi')
    elif offscreen:
        args.append('-RenderOffScreen')
    else:
        args += ['-windowed', '-ResX=800', '-ResY=600']
    return subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def wait_for_server(port=2000, timeout=600):  # cold boots after reboot exceed 360 s
    # RPC 埠(port)先開,但「串流埠」(streaming port = rpc+1,送相機/LiDAR 感測資料)
    # 還沒就緒;若此時就連線附掛感測器,會狂噴 "streaming client: connection failed"
    # 把整個 run 卡死(restart 後重連失敗的根因)。故須等「兩個埠都開」再 settle。
    deadline = time.time() + timeout
    rpc_ready = False
    while time.time() < deadline:
        if not rpc_ready:
            if is_port_open(port):
                rpc_ready = True
                continue  # 立刻接著檢查串流埠,不再多睡一輪
        elif is_port_open(port + 1):  # 串流埠就緒
            time.sleep(8)  # settle:讓 RPC 與串流子系統都完全初始化
            return True
        time.sleep(5)
    return False


def restart_server(port=2000, offscreen=False, town='Town03', nullrhi=False):
    """Kill any existing server, launch a fresh one, block until ready."""
    kill_server()
    time.sleep(5)
    launch_server(port, offscreen, town, nullrhi)
    if not wait_for_server(port):
        raise RuntimeError(f'CARLA server did not open port {port} in time')


def ensure_server(port=2000, offscreen=False, town='Town03', nullrhi=False):
    if not is_port_open(port):
        restart_server(port, offscreen, town, nullrhi)
