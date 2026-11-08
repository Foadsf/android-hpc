# SPDX-License-Identifier: Apache-2.0
# Build the command-line benchmark for Android so it can be run with `adb shell`.
#   .\scripts\build-android-cli.ps1                # arm64-v8a
#   .\scripts\build-android-cli.ps1 -Abi x86_64    # emulator
# Then:  adb push dist\hpc_cli-arm64-v8a /data/local/tmp/hpc_cli
#        adb shell chmod +x /data/local/tmp/hpc_cli ; adb shell /data/local/tmp/hpc_cli --preset small
param(
  [string]$Abi = "arm64-v8a",
  [string]$Ndk = $(if ($env:ANDROID_NDK_HOME) { $env:ANDROID_NDK_HOME } else { Get-ChildItem "$env:ANDROID_HOME\ndk" -Directory | Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName })
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root "build-android-$Abi"
$ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
if (-not $ninja) { $ninja = Join-Path $env:ANDROID_HOME "cmake\3.22.1\bin\ninja.exe" }
cmake -S $root -B $build -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" `
  "-DCMAKE_TOOLCHAIN_FILE=$Ndk\build\cmake\android.toolchain.cmake" `
  "-DANDROID_ABI=$Abi" "-DANDROID_PLATFORM=android-26" "-DANDROID_STL=c++_static" -DHPC_BUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build $build --target hpc_cli
New-Item -ItemType Directory -Force (Join-Path $root "dist") | Out-Null
Copy-Item (Join-Path $build "core\hpc_cli") (Join-Path $root "dist\hpc_cli-$Abi") -Force
Write-Host "-> dist\hpc_cli-$Abi"
