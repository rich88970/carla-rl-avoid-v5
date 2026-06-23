# Launch the CARLA 0.9.15 server (source build via UE4Editor in game mode).
#
# NOTE: the official packaged 0.9.15 build was tried on 2026-06-11 and cannot
# load Town03 on this machine — its pre-cooked shaders crash fatally under
# NVIDIA driver 596.36 on DX12, DX11, and Vulkan. The source build works
# because its shaders were compiled locally. See CHANGELOG.md.
#
# Usage: .\start_carla.ps1 [-OffScreen]
param(
    [switch]$OffScreen
)

# 路徑一律由環境變數提供(不留任何本機絕對路徑;見 INSTALL.md)
if (-not $env:CARLA_UE4_EDITOR) { throw "缺少 CARLA_UE4_EDITOR,請依 INSTALL.md 設定。" }
if (-not $env:CARLA_UPROJECT)   { throw "缺少 CARLA_UPROJECT,請依 INSTALL.md 設定。" }
$editor = $env:CARLA_UE4_EDITOR
$uproject = $env:CARLA_UPROJECT
# Boot directly into Town03: the map switch after boot is the crash-prone path
$args = @("`"$uproject`"", "/Game/Carla/Maps/Town03", "-game", "-carla-rpc-port=2000", "-quality-level=Low", "-nosound")
if ($OffScreen) {
    $args += "-RenderOffScreen"
} else {
    $args += @("-windowed", "-ResX=800", "-ResY=600")
}

Write-Host "Starting CARLA server (source build): $editor"
Start-Process -FilePath $editor -ArgumentList $args
