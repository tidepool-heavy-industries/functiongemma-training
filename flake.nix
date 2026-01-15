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

        # Use GCC 12 for CUDA compatibility (GCC 15 is too new)
        cudaGcc = pkgs.gcc12;

        linuxLibs = with pkgs; [
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          cudaPackages.nccl
          # Do NOT include nvidia_x11 on WSL2 - we use the Windows host driver
          libgcc
          glibc
          glibc.dev  # C headers for nvcc
          stdenv.cc.cc.lib
          cudaGcc  # GCC 12 for CUDA compatibility
        ];

        # Common build dependencies
        buildInputs = with pkgs; [
          # Rust toolchain
          rustToolchain
          pkg-config
          openssl

          # Python environment
          pythonEnv

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
          # Note: LD_LIBRARY_PATH is also set in shellHook to include /usr/lib/wsl/lib
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
            cudaGcc.cc.lib
            pkgs.stdenv.cc.cc.lib
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.nccl
          ]);

          CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          CUDNN_PATH = "${pkgs.cudaPackages.cudnn}";

          # HuggingFace cache
          HF_HOME = "$HOME/.cache/huggingface";
          TRANSFORMERS_CACHE = "$HOME/.cache/huggingface/transformers";

          # Mistral.rs features
          MISTRALRS_FEATURES = "cuda";

          # NVCC configuration - tell it which gcc to use and where to find headers
          NVCC_CCBIN = "${cudaGcc}/bin/gcc";
          C_INCLUDE_PATH = "${pkgs.glibc.dev}/include";
          CPLUS_INCLUDE_PATH = "${cudaGcc}/include/c++/${cudaGcc.version}:${pkgs.glibc.dev}/include";

          # Rust compilation flags for CUDA
          RUSTFLAGS = "-C link-args=-Wl,-rpath,${pkgs.cudaPackages.cudatoolkit}/lib";
        };

        # Shell hook for additional setup
        shellHook = ''
          ${if isLinux then ''
            # WSL2 Driver Passthrough: Add Windows host driver path
            # This allows Nix-built CUDA programs to access the NVIDIA driver
            export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${pkgs.lib.makeLibraryPath [
              cudaGcc.cc.lib
              pkgs.stdenv.cc.cc.lib
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.nccl
            ]}

            # Verify WSL2 driver is accessible
            if [ ! -f /usr/lib/wsl/lib/libcuda.so.1 ]; then
              echo "⚠️  WARNING: WSL2 NVIDIA driver not found at /usr/lib/wsl/lib/libcuda.so.1"
              echo "   Ensure NVIDIA drivers are installed on your Windows host."
            fi

            # Save environment variables for venv activation
            cat > .nix-lib-path <<EOF
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
export CUDNN_PATH="${pkgs.cudaPackages.cudnn}"
export NVCC_CCBIN="${cudaGcc}/bin/gcc"
export C_INCLUDE_PATH="${pkgs.glibc.dev}/include"
export CPLUS_INCLUDE_PATH="${cudaGcc}/include/c++/${cudaGcc.version}:${pkgs.glibc.dev}/include"
EOF
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
        devShells.default = pkgs.mkShell {
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
            LD_LIBRARY_PATH
            CUDA_PATH
            CUDNN_PATH
            NVCC_CCBIN
            C_INCLUDE_PATH
            CPLUS_INCLUDE_PATH
            RUSTFLAGS;
        });

        # Formatter for nix files
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
