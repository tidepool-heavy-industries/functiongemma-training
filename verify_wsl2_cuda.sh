#!/usr/bin/env bash

echo "🔍 WSL2 CUDA Setup Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check 1: WSL2 driver library
echo "1️⃣  Checking WSL2 NVIDIA driver library..."
if [ -f /usr/lib/wsl/lib/libcuda.so.1 ]; then
    echo "   ✅ Found: /usr/lib/wsl/lib/libcuda.so.1"
    ls -lh /usr/lib/wsl/lib/libcuda.so.1

    # Check if it's a symlink (should be)
    if [ -L /usr/lib/wsl/lib/libcuda.so.1 ]; then
        echo "   ✅ Correctly configured as symlink"
    else
        echo "   ⚠️  WARNING: Should be a symlink, not a regular file"
        echo "      This is a known WSL2 bug. You may need to recreate it from Windows."
    fi
else
    echo "   ❌ NOT FOUND: /usr/lib/wsl/lib/libcuda.so.1"
    echo "      Install NVIDIA drivers on your Windows host"
fi
echo ""

# Check 2: nvidia-smi
echo "2️⃣  Checking nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ nvidia-smi is available"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "   ❌ nvidia-smi not found in PATH"
    echo "      Install NVIDIA drivers on Windows and restart WSL"
fi
echo ""

# Check 3: CUDA libraries in WSL
echo "3️⃣  Checking for other CUDA libraries in WSL..."
wsl_libs=$(ls /usr/lib/wsl/lib/libcuda* 2>/dev/null | wc -l)
if [ "$wsl_libs" -gt 0 ]; then
    echo "   ✅ Found $wsl_libs CUDA libraries:"
    ls -lh /usr/lib/wsl/lib/libcuda* 2>/dev/null | awk '{print "      " $9}'
else
    echo "   ⚠️  No CUDA libraries found in /usr/lib/wsl/lib/"
fi
echo ""

# Check 4: Nix installed
echo "4️⃣  Checking Nix installation..."
if command -v nix &> /dev/null; then
    echo "   ✅ Nix is installed"
    nix --version

    # Check if flakes are enabled
    if nix flake --version &> /dev/null 2>&1; then
        echo "   ✅ Flakes are enabled"
    else
        echo "   ⚠️  Flakes not enabled. Add to ~/.config/nix/nix.conf:"
        echo "      experimental-features = nix-command flakes"
    fi
else
    echo "   ❌ Nix not installed"
    echo "      Install with: sh <(curl -L https://nixos.org/nix/install) --daemon"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Next Steps:"
echo ""
if [ ! -f /usr/lib/wsl/lib/libcuda.so.1 ]; then
    echo "⚠️  Install NVIDIA drivers on Windows host"
    echo "   Download from: https://www.nvidia.com/Download/index.aspx"
    echo "   After install, restart WSL: wsl.exe --shutdown"
fi

if ! command -v nix &> /dev/null; then
    echo "⚠️  Install Nix to use the flake environment"
else
    echo "✅ Ready to run: nix develop"
    echo "   This will download CUDA toolkit and set up the environment"
fi
echo ""
