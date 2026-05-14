//! Persistent system knowledge — facts about the host that the model
//! shouldn't have to re-derive every turn (which package manager, how to
//! install/remove things, the escalation tool, etc.).
//!
//! Two layers:
//!
//! 1. A curated static table keyed by `distro_id` (`KNOWN_DISTROS`) — covers
//!    the common Linux distributions out of the box.
//! 2. A user-editable JSON file at `$XDG_DATA_HOME/tux/system.json` written
//!    by `tux init`. If present it wins over the static table, so the user
//!    can tweak commands (`doas` instead of `sudo`, custom paths, etc.).

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DistroKnowledge {
    /// Human label, e.g. "pacman", "apt", "nix-env".
    pub package_manager: String,
    /// Command template — `{pkg}` is replaced with the package name.
    pub install_cmd: String,
    pub remove_cmd: String,
    pub search_cmd: String,
    pub update_cmd: String,
    /// How to escalate to root: typically `sudo`, sometimes `doas`/`pkexec`.
    pub escalation: String,
    /// Service manager invocation prefix, e.g. `systemctl`, `rc-service`.
    pub service_manager: String,
    /// Useful config paths the model should know about.
    pub config_paths: Vec<String>,
    /// Free-form notes injected into the system prompt — distro quirks
    /// the model needs to be aware of.
    pub notes: Vec<String>,
}

impl DistroKnowledge {
    pub fn as_prompt_block(&self) -> String {
        let mut s = String::from("Package management & system commands:\n");
        s.push_str(&format!("- package manager: {}\n", self.package_manager));
        s.push_str(&format!("- install: {}\n", self.install_cmd));
        s.push_str(&format!("- remove:  {}\n", self.remove_cmd));
        s.push_str(&format!("- search:  {}\n", self.search_cmd));
        s.push_str(&format!("- update:  {}\n", self.update_cmd));
        s.push_str(&format!("- escalate: {}\n", self.escalation));
        s.push_str(&format!("- services: {}\n", self.service_manager));
        if !self.config_paths.is_empty() {
            s.push_str(&format!("- config: {}\n", self.config_paths.join(", ")));
        }
        for note in &self.notes {
            s.push_str(&format!("- note: {note}\n"));
        }
        s
    }
}

/// Curated facts for distros we recognize. Keep entries minimal and accurate;
/// the model fills in the rest from general Linux knowledge.
fn known(id: &str) -> Option<DistroKnowledge> {
    Some(match id {
        "nixos" => DistroKnowledge {
            package_manager: "nix".into(),
            install_cmd: "nix-env -iA nixos.{pkg}".into(),
            remove_cmd: "nix-env -e {pkg}".into(),
            search_cmd: "nix search nixpkgs {pkg}".into(),
            update_cmd: "sudo nixos-rebuild switch".into(),
            escalation: "sudo".into(),
            service_manager: "systemctl".into(),
            config_paths: vec!["/etc/nixos/configuration.nix".into()],
            notes: vec![
                "prefer adding packages to /etc/nixos/configuration.nix and rebuilding over imperative installs"
                    .into(),
                "user-level: home-manager if installed, else nix-env".into(),
            ],
        },
        "arch" | "endeavouros" | "manjaro" => DistroKnowledge {
            package_manager: "pacman".into(),
            install_cmd: "sudo pacman -S {pkg}".into(),
            remove_cmd: "sudo pacman -Rns {pkg}".into(),
            search_cmd: "pacman -Ss {pkg}".into(),
            update_cmd: "sudo pacman -Syu".into(),
            escalation: "sudo".into(),
            service_manager: "systemctl".into(),
            config_paths: vec!["/etc/pacman.conf".into()],
            notes: vec!["AUR helpers (yay, paru) handle community packages if installed".into()],
        },
        "debian" | "ubuntu" | "linuxmint" | "pop" => DistroKnowledge {
            package_manager: "apt".into(),
            install_cmd: "sudo apt install {pkg}".into(),
            remove_cmd: "sudo apt remove {pkg}".into(),
            search_cmd: "apt search {pkg}".into(),
            update_cmd: "sudo apt update && sudo apt upgrade".into(),
            escalation: "sudo".into(),
            service_manager: "systemctl".into(),
            config_paths: vec!["/etc/apt/sources.list".into(), "/etc/apt/sources.list.d/".into()],
            notes: vec![],
        },
        "fedora" | "rhel" | "centos" | "rocky" | "almalinux" => DistroKnowledge {
            package_manager: "dnf".into(),
            install_cmd: "sudo dnf install {pkg}".into(),
            remove_cmd: "sudo dnf remove {pkg}".into(),
            search_cmd: "dnf search {pkg}".into(),
            update_cmd: "sudo dnf upgrade".into(),
            escalation: "sudo".into(),
            service_manager: "systemctl".into(),
            config_paths: vec!["/etc/dnf/dnf.conf".into()],
            notes: vec!["RPMFusion repo needed for many media codecs".into()],
        },
        "opensuse-tumbleweed" | "opensuse-leap" | "opensuse" => DistroKnowledge {
            package_manager: "zypper".into(),
            install_cmd: "sudo zypper install {pkg}".into(),
            remove_cmd: "sudo zypper remove {pkg}".into(),
            search_cmd: "zypper search {pkg}".into(),
            update_cmd: "sudo zypper dup".into(),
            escalation: "sudo".into(),
            service_manager: "systemctl".into(),
            config_paths: vec!["/etc/zypp/".into()],
            notes: vec![],
        },
        "alpine" => DistroKnowledge {
            package_manager: "apk".into(),
            install_cmd: "sudo apk add {pkg}".into(),
            remove_cmd: "sudo apk del {pkg}".into(),
            search_cmd: "apk search {pkg}".into(),
            update_cmd: "sudo apk upgrade".into(),
            escalation: "sudo".into(),
            service_manager: "rc-service".into(),
            config_paths: vec!["/etc/apk/repositories".into()],
            notes: vec!["uses OpenRC, not systemd".into()],
        },
        "void" => DistroKnowledge {
            package_manager: "xbps".into(),
            install_cmd: "sudo xbps-install -S {pkg}".into(),
            remove_cmd: "sudo xbps-remove {pkg}".into(),
            search_cmd: "xbps-query -Rs {pkg}".into(),
            update_cmd: "sudo xbps-install -Su".into(),
            escalation: "sudo".into(),
            service_manager: "sv".into(),
            config_paths: vec!["/etc/xbps.d/".into()],
            notes: vec!["uses runit for services, not systemd".into()],
        },
        _ => return None,
    })
}

fn cache_path() -> Option<PathBuf> {
    let dirs = ProjectDirs::from("dev", "tux", "tux")?;
    Some(dirs.data_dir().join("system.json"))
}

/// Load knowledge for `distro_id`. Prefers the on-disk cache, falls back to
/// the curated static table. Returns `None` for distros we don't know about
/// (the prompt simply omits the block — the model uses general knowledge).
pub fn for_distro(distro_id: &str) -> Option<DistroKnowledge> {
    if let Some(path) = cache_path() {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(k) = serde_json::from_str::<DistroKnowledge>(&content) {
                return Some(k);
            }
        }
    }
    known(distro_id)
}

/// Persist `knowledge` to the cache. Called by `tux init` so the user can
/// inspect / edit `~/.local/share/tux/system.json`.
pub fn save(knowledge: &DistroKnowledge) -> anyhow::Result<PathBuf> {
    let path = cache_path()
        .ok_or_else(|| anyhow::anyhow!("could not resolve project dirs for cache"))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(knowledge)?;
    fs::write(&path, json)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_common_distros() {
        for id in [
            "nixos", "arch", "manjaro", "debian", "ubuntu", "fedora", "alpine", "void",
            "opensuse-tumbleweed",
        ] {
            assert!(known(id).is_some(), "missing knowledge for {id}");
        }
    }

    #[test]
    fn unknown_distro_returns_none() {
        assert!(known("frankenix").is_none());
    }

    #[test]
    fn arch_install_cmd_uses_sudo_pacman() {
        let k = known("arch").unwrap();
        assert!(k.install_cmd.contains("sudo pacman -S"));
    }

    #[test]
    fn nixos_notes_warn_about_imperative_installs() {
        let k = known("nixos").unwrap();
        assert!(k.notes.iter().any(|n| n.contains("configuration.nix")));
    }
}
