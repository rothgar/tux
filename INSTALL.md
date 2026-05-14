# Installing tux

Pick whichever delivery you prefer. All paths give you the same `tux` CLI.

## Nix (recommended on NixOS / nix-darwin)

```sh
# one-shot run, no install
nix run github:rothgar/tux -- "install neovim"

# or install into your profile
nix profile install github:rothgar/tux
```

Inside a checkout:

```sh
nix build .#tux
./result/bin/tux --version
```

The Nix package wraps `tux` so it can find `grim`, `scrot`, `wl-paste`,
`xclip`, and `fd` even on minimal systems.

## Homebrew (macOS, Linuxbrew)

> Status: formula skeleton lives at `packaging/homebrew/tux.rb`. **Not
> yet published to a tap and not yet tested on macOS.** First install
> from this checkout will surface anything missing.

```sh
brew install --build-from-source ./packaging/homebrew/tux.rb
```

To consume it like a real tap once published:

```sh
brew tap rothgar/tux
brew install tux
```

## Container (ghcr.io)

Best suited for the headless / daemon path. Desktop integrations
(clipboard, screenshot) only work if you mount the host's wayland or X
socket into the container.

```sh
# pull the published image (after the first tagged release)
docker pull ghcr.io/rothgar/tux:latest

# answer a single prompt; stdin → tux → stdout
echo "what does systemd-analyze show?" \
  | docker run --rm -i -v tux-data:/data ghcr.io/rothgar/tux:latest

# run the daemon (mount XDG_RUNTIME_DIR in to share its socket)
docker run -d --name tuxd \
  -v tux-data:/data \
  -v "$XDG_RUNTIME_DIR:/run/user/1000" \
  ghcr.io/rothgar/tux:latest daemon serve
```

## Build from source

```sh
git clone https://github.com/rothgar/tux && cd tux
nix develop                     # or install cmake, clang, libclang yourself
cargo build --release -p tux-cli
./target/release/tux --version
```

To build without llama.cpp (mock backend only — useful for fast iteration
or hosts without `cmake`):

```sh
cargo build --no-default-features
```

## After installing

```sh
tux init                    # detect hardware, download a model, write system.json
tux init --with-vision      # also download a vision model + mmproj
tux "install neovim"        # try it out
```
