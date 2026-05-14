# tux

Local-AI Linux assistant. Talk or type, get help with system settings, files,
screenshots, and installs. Distro-agnostic; learns the host from local config.

## Status

v0:

- `tux-core` — `Backend` trait, `MockBackend`, real **llama.cpp backend**
  (feature `llama`, default on) using `llama-cpp-2` with the ChatML / Qwen2.5
  prompt template; `Tool` trait + screenshot tool; `SystemContext`
  (distro / DE / session detection); natural-language `Agent` loop with
  model-driven tool calls (no slash commands).
- `tux-cli` — pipe-friendly Unix shape. Input is always natural language.
  No REPL, no slash commands, no streaming. Final answer on stdout; trace
  and warnings on stderr.
- `tux-gui` — single-window GTK4 shell with transcript + entry, runs the same
  agent on a background tokio runtime.

## Setup

```sh
nix develop
cargo build --release
./target/release/tux init                 # downloads model, writes knowledge cache,
                                           #   installs + starts the systemd user unit
./target/release/tux init --with-vision   # ALSO downloads Qwen3-VL-2B + mmproj so
                                           #   the model can inspect screenshots
./target/release/tux why is text rendering blurry on my external monitor
```

`tux init` does four things in one go:

1. **Pick + download** a text model based on detected cores / RAM.
2. **Validate** by loading it once.
3. **Persist distro knowledge** to `$XDG_DATA_HOME/tux/system.json` so the
   model never has to re-derive how to install packages on this host.
4. **Install + enable** `~/.config/systemd/user/tuxd.service` so the daemon
   starts on login (skip with `--no-daemon`).

`tux init` reads `/proc/meminfo` and `available_parallelism()`, then picks
from a built-in [model registry](file:///dev/null) — currently:

| id              | size   | when picked                       |
| --------------- | ------ | --------------------------------- |
| `qwen3.5-4b-q4` | 2.5 GB | ≥6 cores and ≥6 GB RAM (default)  |
| `qwen3.5-2b-q4` | 1.3 GB | 4–5 cores or 3–6 GB RAM           |
| `qwen3.5-0.8b-q4` | 0.6 GB | <4 cores or <3 GB RAM (fallback) |

Override with `tux init --model qwen3.5-2b-q4`. The download streams to
`<id>.gguf.partial`, atomically renames on success, then validates by
loading once and symlinks `default.gguf` → the chosen file.

You can still pass `--model /path/to/anything.gguf` (or set `TUX_MODEL`)
to skip the registry. Without a model, the mock backend runs so you can
still exercise the agent loop.

## Daemon

To avoid paying the model load cost on every invocation, run the daemon:

```sh
tux daemon serve &        # foreground server; bind to $XDG_RUNTIME_DIR/tux.sock
tux daemon status         # check it's alive
tux daemon stop           # ask it to exit
```

When the socket exists, `tux <prompt>` auto-uses it; otherwise it falls
back to an in-process model load. Same protocol either way — your scripts
don't change.

The socket is created mode `0600` (user-only). Wire format is
newline-delimited JSON, one request per connection — debuggable with
`socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/tux.sock`.

A typical setup is to manage `tuxd` via a systemd user unit:

```ini
# ~/.config/systemd/user/tuxd.service
[Service]
ExecStart=%h/.cargo/bin/tux daemon serve
Restart=on-failure
[Install]
WantedBy=default.target
```

```sh
systemctl --user enable --now tuxd.service
```

## Performance without a discrete GPU

A 4B Q4_K_M model is ~2.5 GB on disk, ~3 GB resident. CPU-only throughput
on x86 with AVX2:

| Hardware                                  | tok/s   | 200-tok answer |
| ----------------------------------------- | ------- | -------------- |
| Modern desktop (Ryzen 7 / i7, 8C/16T)     | 15–25   | ~10–15 s       |
| Laptop P-series (Ryzen 7000U / i7-13xxH)  | 8–15    | ~15–25 s       |
| Older laptop (4–6 cores)                  | 4–8     | ~30–50 s       |

For a 2–3× speedup on iGPUs (Intel Iris, AMD APUs, Intel Arc-integrated)
build with the **`vulkan`** feature:

```sh
nix develop
cargo build --release --features vulkan
```

(The nix dev shell already pulls in `vulkan-loader` and `vulkan-headers`.)

## Layout

```
tux/
├── flake.nix          # nix dev shell (rust, gtk4, pkg-config, clang, grim, scrot)
├── Cargo.toml         # workspace; default-members = core + cli (gui needs gtk4 libs)
├── tux-core/
├── tux-cli/
└── tux-gui/
```

## Build

```sh
nix develop                              # rust + gtk4 + llama.cpp build deps

cargo build                              # core + cli with llama backend
cargo build -p tux-gui                   # GTK4 GUI with llama backend
cargo build --no-default-features        # mock-only, no system deps needed
```

## Run

```sh
tux install neovim                                # args become the prompt
tux why is text rendering blurry on my monitor
echo "install vim" | tux                          # piped stdin
cat error.log | tux "what's wrong with this?"     # args + stdin combined
journalctl -xe --since '5 min ago' | tux "summarize the failures"
tux info                                          # host facts + tools
```

The model decides when to use a tool. Ask "take a screenshot so you can see
what I'm looking at" and it will emit a `<tool name="screenshot">{}</tool>`
call which the agent runs and feeds back for a final answer.

Stdout is the answer; stderr is the tool-call trace and warnings — pipe with
confidence:

```sh
tux summarize this dmesg < /var/log/dmesg | tee answer.txt
```

## Tests

```sh
# fast, hermetic — uses ScriptedBackend, no model required
cargo test --no-default-features -p tux-core

# eval suite — runs real prompts through the actual model
nix develop
export TUX_MODEL="${XDG_DATA_HOME:-$HOME/.local/share}/tux/models/default.gguf"
cargo test --features llama --release -p tux-core --test eval -- --nocapture --test-threads=1
```

The unit tests cover the agent loop (direct answers, single tool call, multi-hop,
hop limit, unknown tool, system-prompt grounding) using a `ScriptedBackend`.
The eval suite asks the real model common Linux questions and checks for
sensible behavior — answers about installs, distro-aware replies grounded in
host facts, screenshot tool invocation, on-topic diagnostics, and that the
`<think>` block doesn't leak. It auto-skips when `TUX_MODEL` is unset.

## System knowledge

Detected on every run from `/etc/os-release` + `$XDG_*` env vars; the
package-management slice is keyed off `distro_id` and lives in
[knowledge.rs](file:///home/rothgar/src/tux/tux-core/src/knowledge.rs)
for: NixOS, Arch / Manjaro / EndeavourOS, Debian / Ubuntu / Mint / Pop,
Fedora / RHEL / Rocky / Alma, openSUSE, Alpine, Void.

`tux init` snapshots the result to `$XDG_DATA_HOME/tux/system.json` —
edit it freely; the runtime prefers the file over the static table.

## Vision

`tux init --with-vision` downloads **Qwen3-VL-2B** (~1.5 GB) plus its
projector (`mmproj`, ~700 MB), symlinks both as
`default.gguf` / `default.mmproj`, and the runtime auto-loads the
multimodal context. From then on, when a tool returns an image path
(e.g. `screenshot` or anything that puts `image_path` / `path`
ending in `.png` into `data`), the agent attaches that image to the
*next* model turn so it can be inspected.

Memory: text model + vision model both loaded ≈ 4.5 GB resident.

## Roadmap (next)

1. More tools: `find_file`, `read_setting`, `install_package` (with elevated
   privilege prompt via `pkexec`).
3. Auto-detect chat template from the GGUF metadata so non-ChatML models
   work without code changes.
4. Confirmation prompts for destructive tools — read from `/dev/tty` so they
   work even when stdin is piped.
5. Global hotkey activation (D-Bus / portal) — GUI only.
6. Voice input via whisper.cpp — GUI only.
7. Background watchers (journald, failing units, disk thresholds) running
   inside `tuxd` with `notify-send` integration.
