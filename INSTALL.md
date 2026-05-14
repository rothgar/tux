# Installing tux

Pick whichever delivery you prefer. All paths give you the same `tux` CLI.

## Nix (recommended on NixOS / nix-darwin)

```sh
# install into your profile
nix profile install github:rothgar/tux
```

Inside a checkout:

```sh
nix build .#tux
./result/bin/tux --version
```

The Nix package wraps `tux` so it can find `grim`, `scrot`, `wl-paste`,
`xclip`, and `fd` even on minimal systems.

### Home-Manager

If you manage your user environment with [home-manager](https://github.com/nix-community/home-manager),
add `tux` as a flake input and install the package + a user systemd
unit for the daemon:

```nix
# flake.nix
{
  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager.url = "github:nix-community/home-manager";
    tux = {
      url = "github:rothgar/tux";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, home-manager, tux, ... }: {
    homeConfigurations."me" = home-manager.lib.homeManagerConfiguration {
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
      modules = [
        ({ pkgs, ... }: {
          home.packages = [ tux.packages.x86_64-linux.tux ];

          # Run `tuxd` as a user service so the model stays loaded.
          systemd.user.services.tuxd = {
            Unit.Description = "tux local AI assistant daemon";
            Service = {
              ExecStart = "${tux.packages.x86_64-linux.tux}/bin/tux daemon serve";
              Restart   = "on-failure";
            };
            Install.WantedBy = [ "default.target" ];
          };
        })
      ];
    };
  };
}
```

After switching, run `tux init` once to download a model, then
`systemctl --user start tuxd.service`.

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
```

### Running the daemon as a container

The daemon needs two host paths:

1. **The model directory** — keeps multi-GB GGUF files on the host so
   container rebuilds don't trigger a re-download. The image expects
   `XDG_DATA_HOME=/data`, so models live at `/data/tux/models/` inside.
2. **The unix socket directory** — `tuxd` binds
   `$XDG_RUNTIME_DIR/tux.sock`; the host CLI reads the same path to
   talk to it.

Download a model on the host first (so you don't need to bind a TTY
into the daemon container):

```sh
# one-shot init using a host-mounted model dir
mkdir -p ~/.local/share/tux
docker run --rm -it \
  -v "$HOME/.local/share/tux:/data/tux" \
  ghcr.io/rothgar/tux:latest init
```

Then start the long-running daemon, sharing both the model dir and the
runtime socket dir:

```sh
docker run -d --name tuxd \
  --user "$(id -u):$(id -g)" \
  -v "$HOME/.local/share/tux:/data/tux" \
  -v "$XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR" \
  -e XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
  ghcr.io/rothgar/tux:latest daemon serve
```

With those mounts in place, a host-side `tux "..."` will auto-detect
`$XDG_RUNTIME_DIR/tux.sock` and route through the containerized daemon.
Verify with:

```sh
tux daemon status
```

For Podman, the same flags work — drop `--user` if you're running
rootless (UIDs already match the host).

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
