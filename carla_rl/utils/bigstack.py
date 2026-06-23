"""Run a function on a thread with an enlarged stack.

On Windows, mixing the CARLA client with an imported PyTorch overflows the
default ~1-2 MB thread stack inside carla RPC loops (observed: deterministic
'Windows fatal exception: stack overflow' in _clear_all_actors on the second
env.reset() whenever torch was loaded). Python can't grow the main thread's
stack, but worker threads honor threading.stack_size().

關鍵中文說明:本機只要「同時 import torch + 連 CARLA」,在第二次 env.reset() 會必當機
(Windows stack overflow,exit 253 無 traceback)。主執行緒無法加大堆疊,故把整個訓練/評估
主程式丟到一條 64MB 堆疊的 worker thread 跑(run_with_big_stack)。train_sac / run_eval 都靠它。
"""

import threading


def run_with_big_stack(fn, stack_mb=64):
    result = {}

    def target():
        try:
            result['value'] = fn()
        except BaseException as exc:  # noqa: BLE001 — must capture SystemExit too
            result['error'] = exc

    threading.stack_size(stack_mb * 1024 * 1024)
    try:
        thread = threading.Thread(target=target, name='bigstack-main')
        thread.start()
        thread.join()
    finally:
        threading.stack_size(0)
    # re-raise on the main thread so the process exit code reflects the failure.
    # Without this, a SystemExit/exception in the worker is silently swallowed
    # and the process exits 0 — which made a refused resume look like success.
    if 'error' in result:
        raise result['error']
    return result.get('value')
