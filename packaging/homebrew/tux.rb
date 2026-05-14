# Homebrew formula for tux. UNTESTED — written from a NixOS workstation
# with no macOS to validate against. Treat as a starting point: the first
# `brew install --build-from-source ./packaging/homebrew/tux.rb` on a Mac
# will surface anything missing.
#
# To publish properly, copy this file into a homebrew tap repository
# (`homebrew-tux`) and update `url` + `sha256` per release.
class Tux < Formula
  desc "Local AI assistant for Linux & macOS (CLI)"
  homepage "https://github.com/rothgar/tux"
  # Replace with the real tarball URL + sha256 for each release.
  url "https://github.com/rothgar/tux/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/rothgar/tux.git", branch: "main"

  depends_on "cmake" => :build       # llama-cpp-sys-2 build
  depends_on "rust"  => :build
  # llvm provides libclang for bindgen. On macOS the system clang from
  # Xcode CLT is fine for compiling, but bindgen needs a `libclang.dylib`
  # it can dlopen — homebrew llvm exposes one at a known path.
  depends_on "llvm"  => :build

  def install
    # bindgen looks here.
    ENV["LIBCLANG_PATH"] = "#{Formula["llvm"].opt_lib}"

    # Build the CLI only. tux-gui needs GTK4 which homebrew users on
    # macOS shouldn't have to drag in just to use the CLI.
    system "cargo", "install", *std_cargo_args(path: "tux-cli")
  end

  test do
    # `tux --version` is the cheapest smoke test — no model needed.
    assert_match version.to_s, shell_output("#{bin}/tux --version")

    # `tux info` should exit 0 and print the backend name. With no model
    # configured it'll fall back to the mock backend, which is fine.
    assert_match "backend:", shell_output("#{bin}/tux info")
  end
end
