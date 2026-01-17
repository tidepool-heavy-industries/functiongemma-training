{
  description = "FunctionGemma 270M Fine-tuning Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" "x86_64-linux" ] (system:
      let
        overlays = [ (import rust-overlay) ];

        pkgs = import nixpkgs {
          inherit system overlays;
          config = {
            allowUnfree = true; # Required for CUDA packages
            cudaSupport = system == "x86_64-linux";
          };
        };

        isDarwin = system == "aarch64-darwin";
        isLinux = system == "x86_64-linux";

        # Rust toolchain with appropriate features
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
          targets = if isDarwin then [ "aarch64-apple-darwin" ] else [ "x86_64-unknown-linux-gnu" ];
        };

        # Python with minimal packages
        # Heavy ML packages (torch, transformers, etc.) will be installed via pip in .venv
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          pip
          virtualenv
          # Basic scientific libs (for native dependencies)
          numpy
          # Development tools
          ipython
          pytest
        ]);

        # Platform-specific libraries
        darwinLibs = with pkgs.darwin.apple_sdk.frameworks; [
          Security
          Metal
          MetalKit
          MetalPerformanceShaders
          Foundation
          CoreGraphics
          CoreVideo
          AppKit
        ];

        linuxLibs = with pkgs; [
          cudaPackages.cuda_nvcc
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          cudaPackages.nccl
          # Do NOT include nvidia_x11 on WSL2 - we use the Windows host driver
          # backendStdenv provides GCC-compatible compiler automatically
        ];

        # Common build dependencies
        buildInputs = with pkgs; [
          # Rust toolchain
          rustToolchain
          pkg-config
          openssl

          # Python environment
          pythonEnv

          # Rust bindgen support
          llvmPackages.libclang.lib

          # IPC and debugging
          socat
          netcat

          # Build essentials
          cmake
          git

          # Platform-specific libraries
        ] ++ (if isDarwin then darwinLibs else linuxLibs);

        # Environment variables
        shellEnv = if isDarwin then {
          # macOS Metal acceleration
          DYLD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath darwinLibs}";
          METAL_DEVICE_BACKEND = "metal";

          # Rust compilation flags
          RUSTFLAGS = "-C link-arg=-undefined -C link-arg=dynamic_lookup";

          # HuggingFace cache
          HF_HOME = "$HOME/.cache/huggingface";
          TRANSFORMERS_CACHE = "$HOME/.cache/huggingface/transformers";

          # Mistral.rs features
          MISTRALRS_FEATURES = "metal";

        } else {
          # Linux CUDA acceleration
          # backendStdenv handles compiler compatibility automatically
          CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          CUDNN_PATH = "${pkgs.cudaPackages.cudnn}";

          # HuggingFace cache
          HF_HOME = "$HOME/.cache/huggingface";
          TRANSFORMERS_CACHE = "$HOME/.cache/huggingface/transformers";

          # Mistral.rs features
          MISTRALRS_FEATURES = "cuda";

          # Rust compilation flags for CUDA
          RUSTFLAGS = "-C link-args=-Wl,-rpath,${pkgs.cudaPackages.cudatoolkit}/lib";
        };

        # Shell hook for additional setup
        shellHook = ''
          # Rust bindgen needs to find libclang
          export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"

          ${if isLinux then ''
            # WSL2 Driver Passthrough: Prepend Windows host driver path
            # Also add GCC stdcxx library for PyTorch in venv
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

            # Verify WSL2 driver is accessible
            if [ ! -f /usr/lib/wsl/lib/libcuda.so.1 ]; then
              echo "⚠️  WARNING: WSL2 NVIDIA driver not found at /usr/lib/wsl/lib/libcuda.so.1"
              echo "   Ensure NVIDIA drivers are installed on your Windows host."
            fi

            # Save library path for venv activation
            echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"" > .nix-lib-path
          '' else ""}

          echo "🚀 FunctionGemma Development Environment"
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          echo "Platform: ${system}"
          ${if isDarwin then ''
            echo "Acceleration: Metal (Apple Silicon)"
          '' else ''
            echo "Acceleration: CUDA (NVIDIA)"
            echo "CUDA Toolkit: $(${pkgs.cudaPackages.cudatoolkit}/bin/nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
            if command -v nvidia-smi &> /dev/null; then
              echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
            fi
          ''}
          echo ""
          echo "📦 Toolchains:"
          echo "  Rust:    $(rustc --version | awk '{print $2}')"
          echo "  Python:  $(python --version | awk '{print $2}')"
          ${if isLinux then ''
            echo "  GCC:     $(gcc --version | head -n1 | awk '{print $3}') (CUDA-compatible via backendStdenv)"
          '' else ""}
          echo ""
          echo "📁 Cache Directory:"
          echo "  HF_HOME=$HF_HOME"
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

          # Create cache directories
          mkdir -p $HF_HOME

          # Set up Python virtual environment for ML packages
          if [ ! -d .venv ]; then
            echo ""
            echo "📦 Creating Python virtual environment..."
            python -m venv .venv
            echo "✅ Virtual environment created at .venv"
            echo ""
            echo "Next steps:"
            echo "  1. source .venv/bin/activate && source .nix-lib-path"
            echo "  2. pip install torch transformers accelerate bitsandbytes"
            echo "  3. python test_functiongemma.py"
          else
            echo ""
            echo "💡 To use Python venv with Nix libraries:"
            echo "   source .venv/bin/activate && source .nix-lib-path"
          fi
        '';

      in
      {
        devShells.default = (if isLinux then
          # Use CUDA-compatible stdenv for Linux
          pkgs.mkShell.override { stdenv = pkgs.cudaPackages.backendStdenv; }
        else
          # Use default stdenv for Darwin
          pkgs.mkShell
        ) {
          inherit buildInputs shellHook;

          # Export all environment variables
          inherit (shellEnv)
            HF_HOME
            TRANSFORMERS_CACHE
            MISTRALRS_FEATURES;

          # Platform-specific exports
        } // (if isDarwin then {
          inherit (shellEnv)
            DYLD_LIBRARY_PATH
            METAL_DEVICE_BACKEND
            RUSTFLAGS;
        } else {
          inherit (shellEnv)
            CUDA_PATH
            CUDNN_PATH
            RUSTFLAGS;
        });

        # Formatter for nix files
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
