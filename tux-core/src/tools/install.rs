//! `install_package` tool — install a package using the host's package
//! manager, picked from the persisted `DistroKnowledge`.
//!
//! Two safety properties:
//!
//! 1. **Package name validation.** The name is interpolated into a shell
//!    command (`install_cmd` is a template like `sudo pacman -S {pkg}`),
//!    so we restrict it to a strict charset to keep a chatty model from
//!    smuggling in `; rm -rf` or backticks.
//!
//! 2. **Confirmation read from /dev/tty.** Stdin may be a pipe (`echo
//!    "install foo" | tux`), so a raw `read_line` on stdin would either
//!    consume the prompt body or fail outright. Opening `/dev/tty`
//!    directly always talks to the controlling terminal when one exists.
//!    Headless / non-interactive runs (no controlling tty) refuse to
//!    install — explicit policy: never silently change system state.
//!
//! Escalation: if the template starts with `sudo ` and `pkexec` is on
//! `PATH`, we swap in `pkexec` for a GUI auth prompt (better UX from a
//! desktop assistant). Otherwise we run the template as-is, which falls
//! back to the user's terminal `sudo` prompt.

use super::{Tool, ToolResult};
use crate::context::SystemContext;
use async_trait::async_trait;
use serde_json::json;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use tokio::process::Command;

#[derive(Default)]
pub struct InstallPackageTool;

/// Allow letters, digits, and `._+-` (covers every real package-name
/// scheme we know: `python3`, `gtk4-1.0`, `lib32-glibc`, `g++`, etc.).
/// Forbid leading `-` so the value can't masquerade as a flag.
fn is_valid_package_name(s: &str) -> bool {
    if s.is_empty() || s.starts_with('-') {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-'))
}

/// Substitute `{pkg}` in the template. If the template starts with
/// `sudo ` and `pkexec` is on PATH, swap escalation. Returns
/// `(rendered_command, used_pkexec)`.
fn render_command(template: &str, pkg: &str, pkexec_available: bool) -> (String, bool) {
    let rendered = template.replace("{pkg}", pkg);
    if pkexec_available {
        if let Some(rest) = rendered.strip_prefix("sudo ") {
            return (format!("pkexec {rest}"), true);
        }
    }
    (rendered, false)
}

fn pkexec_on_path() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|d| d.join("pkexec").is_file())
}

/// Prompt on `/dev/tty` and read a single line response. Returns `Ok(false)`
/// (decline) if there's no controlling tty — we treat "no human present"
/// as "do not proceed".
fn confirm_via_tty(prompt: &str) -> anyhow::Result<bool> {
    let tty_path = PathBuf::from("/dev/tty");
    let mut writer = match OpenOptions::new().read(true).write(true).open(&tty_path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("no /dev/tty for confirmation ({e}); declining install");
            return Ok(false);
        }
    };
    write!(writer, "{prompt}")?;
    writer.flush()?;
    let reader_handle = writer.try_clone()?;
    let mut reader = BufReader::new(reader_handle);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let ans = line.trim().to_ascii_lowercase();
    Ok(matches!(ans.as_str(), "y" | "yes"))
}

#[async_trait]
impl Tool for InstallPackageTool {
    fn name(&self) -> &'static str {
        "install_package"
    }

    fn description(&self) -> &'static str {
        "Install a package using the host's package manager. Args: \
         {\"package\": \"name\"}. Asks the user for confirmation on the \
         terminal before doing anything."
    }

    async fn invoke(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let pkg = args
            .get("package")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("install_package requires {{\"package\":\"name\"}}"))?;

        anyhow::ensure!(
            is_valid_package_name(pkg),
            "invalid package name {pkg:?} (allowed: letters, digits, and . _ + -)"
        );

        let ctx = SystemContext::detect();
        let knowledge = ctx.knowledge.ok_or_else(|| {
            anyhow::anyhow!(
                "no distro knowledge available — run `tux init` to populate system.json, \
                 or add an entry for this distro to tux-core/src/knowledge.rs"
            )
        })?;

        let (command, used_pkexec) =
            render_command(&knowledge.install_cmd, pkg, pkexec_on_path());

        let prompt = format!("tux: install `{pkg}` with: {command}\nproceed? [y/N]: ");
        if !confirm_via_tty(&prompt)? {
            return Ok(ToolResult {
                tool: "install_package".into(),
                summary: format!("install of `{pkg}` cancelled"),
                data: json!({
                    "cancelled": true,
                    "package": pkg,
                    "command": command,
                }),
            });
        }

        let status = Command::new("sh")
            .arg("-c")
            .arg(&command)
            .status()
            .await
            .map_err(|e| anyhow::anyhow!("spawn install command: {e}"))?;

        anyhow::ensure!(
            status.success(),
            "install command exited with {status}: `{command}`"
        );

        Ok(ToolResult {
            tool: "install_package".into(),
            summary: format!(
                "installed `{pkg}` via {}",
                if used_pkexec { "pkexec" } else { "sudo" }
            ),
            data: json!({
                "package": pkg,
                "command": command,
                "used_pkexec": used_pkexec,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_normal_package_names() {
        for name in [
            "vim", "neovim", "python3", "gtk4-1.0", "lib32-glibc", "g++", "clang.16",
            "foo_bar",
        ] {
            assert!(is_valid_package_name(name), "should accept {name:?}");
        }
    }

    #[test]
    fn rejects_shell_metacharacters() {
        for bad in [
            "", "vim;rm", "vim rm", "vim`whoami`", "vim$(id)", "vim&&id", "--evil", "vim|cat",
            "vim>out", "vim\nrm",
        ] {
            assert!(!is_valid_package_name(bad), "should reject {bad:?}");
        }
    }

    #[test]
    fn renders_template_substituting_pkg() {
        let (cmd, used) = render_command("sudo pacman -S {pkg}", "vim", false);
        assert_eq!(cmd, "sudo pacman -S vim");
        assert!(!used);
    }

    #[test]
    fn rewrites_sudo_to_pkexec_when_available() {
        let (cmd, used) = render_command("sudo apt install {pkg}", "vim", true);
        assert_eq!(cmd, "pkexec apt install vim");
        assert!(used);
    }

    #[test]
    fn leaves_non_sudo_template_alone_even_with_pkexec() {
        // nix-env -iA nixos.{pkg} doesn't need root.
        let (cmd, used) = render_command("nix-env -iA nixos.{pkg}", "vim", true);
        assert_eq!(cmd, "nix-env -iA nixos.vim");
        assert!(!used);
    }
}
