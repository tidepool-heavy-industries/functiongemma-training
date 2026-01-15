# Nix Flake Development Environment Setup

This repository uses Nix Flakes to provide a reproducible, cross-platform development environment for FunctionGemma 270M fine-tuning.

## Prerequisites

### 1. Install Nix with Flakes support

**macOS or Linux:**
```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

Enable flakes by adding to `~/.config/nix/nix.conf` (create if it doesn't exist):
```
experimental-features = nix-command flakes
```

### 2. Install direnv

**macOS:**
```bash
nix profile install nixpkgs#direnv
```

**Linux (WSL2):**
```bash
nix profile install nixpkgs#direnv
```

Then add to your shell configuration (`~/.bashrc` or `~/.zshrc`):
```bash
eval "$(direnv hook bash)"  # for bash
# or
eval "$(direnv hook zsh)"   # for zsh
```

## Quick Start

1. **Clone this repository** (or navigate to it):
   ```bash
   cd /path/to/gemma
   ```

2. **Allow direnv** (first time only):
   ```bash
   direnv allow
   ```

   This will automatically download and set up all dependencies. The first run will take several minutes as Nix builds the environment.

3. **Verify the environment**:
   ```bash
   rustc --version    # Should show stable Rust
   ghc --version      # Should show GHC 9.6+
   python --version   # Should show Python 3.11
   ```

## Platform-Specific Notes

### macOS (aarch64-darwin)

- **Metal acceleration** is automatically configured
- Apple frameworks (Security, Metal, Foundation, CoreGraphics) are included
- Environment variable `METAL_DEVICE_BACKEND=metal` is set

### Linux WSL2 (x86_64-linux)

- **CUDA support** requires unfree packages (already enabled in `flake.nix`)
- NVIDIA drivers must be installed on Windows host
- Verify CUDA with:
  ```bash
  nvidia-smi  # Should show GPU info
  ```

## Handling Unfree Packages (CUDA)

CUDA packages are marked as "unfree" in nixpkgs. This flake automatically enables them via:

```nix
config.allowUnfree = true;
```

If you need to enable this globally, add to `~/.config/nixpkgs/config.nix`:
```nix
{
  allowUnfree = true;
}
```

## Project Structure

```
gemma/
├── flake.nix           # Main Nix Flake definition
├── .envrc              # Direnv configuration
├── haskell-orchestrator/  # Haskell code (to be created)
├── rust-inference/     # Rust mistral.rs wrapper (to be created)
├── python-executor/    # Python environment for agent code (to be created)
└── .venv/              # Python virtual env (auto-created)
```

## Python Virtual Environment

The Nix shell provides Python 3.11 with core ML libraries (torch, transformers, numpy). For additional packages like `unsloth` that have complex native dependencies:

```bash
source .venv/bin/activate
pip install unsloth bitsandbytes
```

The `.venv` is automatically created on first shell entry.

## Debugging IPC (Unix Domain Sockets)

The environment includes `socat` for debugging socket communication between Haskell and Rust:

```bash
# Listen on a socket and print traffic
socat -v UNIX-LISTEN:/tmp/test.sock -

# Forward between sockets
socat UNIX-LISTEN:/tmp/bridge.sock UNIX-CONNECT:/tmp/actual.sock
```

## Building Rust with mistral.rs

The environment sets `MISTRALRS_FEATURES` automatically:
- macOS: `metal`
- Linux: `cuda`

Example `Cargo.toml` features:
```toml
[dependencies]
mistralrs = { version = "0.x", features = ["metal"] }  # macOS
# or
mistralrs = { version = "0.x", features = ["cuda"] }   # Linux
```

## Updating the Environment

To update all dependencies:
```bash
nix flake update
```

To update a specific input:
```bash
nix flake update nixpkgs
```

## Troubleshooting

### Environment not loading
```bash
direnv reload
```

### Clear Nix cache
```bash
nix-collect-garbage -d
```

### Flake evaluation errors
```bash
nix flake check
```

### CUDA not found (Linux)
Verify NVIDIA drivers on Windows host:
```powershell
nvidia-smi  # Run in PowerShell
```

In WSL2, ensure `/usr/lib/wsl/lib` is in your library path (the flake handles this).

## Cache Directory

Models are cached in:
```
$HOME/.cache/huggingface/
```

This is shared across both platforms and persists between shell sessions.

## Next Steps

1. Create Haskell orchestrator project:
   ```bash
   mkdir haskell-orchestrator
   cd haskell-orchestrator
   cabal init
   ```

2. Create Rust inference wrapper:
   ```bash
   mkdir rust-inference
   cd rust-inference
   cargo init
   ```

3. Set up Python executor:
   ```bash
   mkdir python-executor
   cd python-executor
   source ../.venv/bin/activate
   pip install unsloth
   ```

## Resources

- [Nix Flakes Manual](https://nixos.org/manual/nix/stable/command-ref/new-cli/nix3-flake.html)
- [FunctionGemma Documentation](https://huggingface.co/google/functiongemma-270m)
- [mistral.rs Documentation](https://github.com/EricLBuehler/mistral.rs)
