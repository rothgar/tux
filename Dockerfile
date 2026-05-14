# tux container image.
#
# Caveat: tux is primarily a *desktop* assistant — it shells out to grim /
# scrot for screenshots, wl-paste / xclip for the clipboard, and (later)
# pkexec/sudo for installs. Inside a container these only work if you
# also share the host's wayland/X socket and clipboard. The container is
# most useful for the headless paths:
#
#   - `tux daemon serve` running over a unix socket mount
#   - eval / batch usage where the model answers prompts piped in on
#     stdin and writes plain text to stdout
#
# For interactive desktop use, prefer the `nix build .#tux` package or
# the homebrew formula.

# ---- builder ------------------------------------------------------------
FROM rust:1.83-bookworm AS builder

# llama-cpp-sys-2 needs cmake + clang + libclang at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake clang libclang-dev pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY tux-core ./tux-core
COPY tux-cli  ./tux-cli
COPY tux-gui  ./tux-gui

# bindgen needs LIBCLANG_PATH on systems where it's not auto-discovered.
ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib

# Build the CLI only — GUI needs gtk4 system libs we deliberately leave
# out of the container (it would never have a display anyway).
RUN cargo build --release -p tux-cli

# ---- runtime ------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# - ca-certificates: tux init downloads model files over HTTPS.
# - libgomp1, libstdc++: linked by llama.cpp's release build.
# - grim/scrot/wl-clipboard/xclip/fd: optional helpers tux shells out to
#   when you mount a desktop session in. Cheap to include.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libstdc++6 \
        grim scrot wl-clipboard xclip fd-find \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/fdfind /usr/local/bin/fd

COPY --from=builder /src/target/release/tux /usr/local/bin/tux

# Models live in /data so callers can mount a volume and survive a
# container rebuild without re-downloading several GB.
ENV XDG_DATA_HOME=/data
VOLUME ["/data"]

ENTRYPOINT ["tux"]
