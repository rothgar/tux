{
  description = "tux - local AI Linux assistant";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Common native build deps for anything that compiles llama.cpp.
        # llama-cpp-sys-2 invokes cmake + clang and uses bindgen
        # (LIBCLANG_PATH below).
        llamaNativeBuildInputs = with pkgs; [
          pkg-config
          cmake
          clang
          llvmPackages.libclang
        ];

        # tux-cli itself is text-only — it doesn't link the GUI libs.
        tux-cli = pkgs.rustPlatform.buildRustPackage {
          pname = "tux";
          version = "0.2.0";

          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            # Skip target/ and editor turds so the source hash is stable
            # across local dev rebuilds.
            filter = path: type:
              let baseName = baseNameOf (toString path);
              in !(builtins.elem baseName [ "target" ".git" "result" ]);
          };

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          # Only build the CLI; the GUI is its own thing and pulls gtk4.
          cargoBuildFlags = [ "-p" "tux-cli" ];
          # Default features include `llama` (see tux-cli/Cargo.toml).
          # Tests under --features llama need a model file, so skip them
          # in the package build; they're already covered by `cargo test
          # --no-default-features` in CI.
          doCheck = false;

          # makeWrapper lets us patch tux's runtime PATH so it can find
          # grim/scrot/wl-paste/xclip/fd even on minimal systems.
          nativeBuildInputs = llamaNativeBuildInputs ++ [ pkgs.makeWrapper ];

          # bindgen wants this even under nix — the path is exported by
          # the dev shell too, see shellHook below.
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          postInstall = ''
            wrapProgram $out/bin/tux \
              --prefix PATH : ${pkgs.lib.makeBinPath (with pkgs; [
                grim slurp scrot wl-clipboard xclip fd
              ])}
          '';

          meta = with pkgs.lib; {
            description = "Local AI assistant for Linux (CLI)";
            homepage = "https://github.com/rothgar/tux";
            license = with licenses; [ mit asl20 ];
            mainProgram = "tux";
            platforms = platforms.linux;
          };
        };

      in {
        packages = {
          default = tux-cli;
          tux = tux-cli;
        };

        # `nix run .#tux -- "install neovim"`
        apps.default = {
          type = "app";
          program = "${tux-cli}/bin/tux";
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            # rust toolchain
            rustc
            cargo
            rustfmt
            clippy
            rust-analyzer

            # build deps
            pkg-config
            cmake
            clang
            llvmPackages.libclang  # bindgen for llama-cpp-2

            # gtk4 + linux ui deps
            gtk4
            glib
            gdk-pixbuf
            graphene
            cairo
            pango
            harfbuzz

            # screenshot helpers (wayland + x11)
            grim
            slurp
            scrot

            # clipboard helpers (wayland + x11)
            wl-clipboard
            xclip

            # file search (find_file tool prefers fd over find)
            fd

            # vulkan (opt-in iGPU acceleration via `cargo build --features vulkan`)
            vulkan-loader
            vulkan-headers
            shaderc
          ];

          # bindgen needs LIBCLANG_PATH
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          shellHook = ''
            echo "tux dev shell ready (rust $(rustc --version | cut -d' ' -f2), gtk4)"
          '';
        };
      });
}
