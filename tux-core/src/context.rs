//! System context: lightweight facts about the host that get prepended to
//! the system prompt so the model can give distro-aware answers.

use crate::knowledge::{self, DistroKnowledge};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemContext {
    pub distro_id: Option<String>,
    pub distro_pretty_name: Option<String>,
    pub desktop: Option<String>,
    pub session_type: Option<String>,
    pub shell: Option<String>,
    pub kernel: Option<String>,
    /// Directories on the user's `$PATH` that live under `$HOME` and exist
    /// — i.e. places the user can drop a script without `sudo`. The model
    /// uses this to pick a sensible install location (`~/bin`,
    /// `~/.local/bin`, etc.) when asked to create a helper script.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub path_dirs: Vec<String>,
    /// Persistent distro-specific knowledge (package manager commands etc.).
    /// Loaded from the static table or the on-disk cache. Skipped from the
    /// prompt block when `None`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub knowledge: Option<DistroKnowledge>,
}

impl SystemContext {
    /// Best-effort detection. Never fails — missing facts are simply `None`.
    pub fn detect() -> Self {
        let mut ctx = Self::default();

        if let Ok(content) = fs::read_to_string("/etc/os-release") {
            for line in content.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    let v = v.trim_matches('"').to_string();
                    match k {
                        "ID" => ctx.distro_id = Some(v),
                        "PRETTY_NAME" => ctx.distro_pretty_name = Some(v),
                        _ => {}
                    }
                }
            }
        }

        ctx.desktop = env::var("XDG_CURRENT_DESKTOP").ok();
        ctx.session_type = env::var("XDG_SESSION_TYPE").ok();
        ctx.shell = env::var("SHELL").ok();

        if let Ok(uname) = fs::read_to_string("/proc/sys/kernel/osrelease") {
            ctx.kernel = Some(uname.trim().to_string());
        }

        if let Some(id) = ctx.distro_id.as_deref() {
            ctx.knowledge = knowledge::for_distro(id);
        }

        ctx.path_dirs = user_path_dirs();

        ctx
    }

    /// Render as a compact block to embed in a system prompt.
    pub fn as_prompt_block(&self) -> String {
        let mut out = String::from("Host facts:\n");
        let push = |out: &mut String, k: &str, v: &Option<String>| {
            if let Some(v) = v {
                out.push_str(&format!("- {}: {}\n", k, v));
            }
        };
        push(&mut out, "distro", &self.distro_pretty_name);
        push(&mut out, "distro_id", &self.distro_id);
        push(&mut out, "desktop", &self.desktop);
        push(&mut out, "session", &self.session_type);
        push(&mut out, "shell", &self.shell);
        push(&mut out, "kernel", &self.kernel);
        if !self.path_dirs.is_empty() {
            out.push_str(&format!(
                "- writable PATH dirs: {}\n",
                self.path_dirs.join(", ")
            ));
        }
        if let Some(k) = &self.knowledge {
            out.push('\n');
            out.push_str(&k.as_prompt_block());
        }
        out
    }
}

/// Return the subset of `$PATH` directories that (a) live under `$HOME`
/// and (b) exist. These are the locations where the user can install a
/// helper script without root. Order is preserved (PATH order matters —
/// earlier entries shadow later ones).
pub(crate) fn user_path_dirs() -> Vec<String> {
    let Some(home) = env::var_os("HOME") else {
        return Vec::new();
    };
    let Some(path) = env::var_os("PATH") else {
        return Vec::new();
    };
    user_path_dirs_from(&home, &path, |p| p.is_dir())
}

/// Pure variant of [`user_path_dirs`] that takes `HOME`, `PATH`, and an
/// existence predicate. Lets tests run without poking process-wide env.
fn user_path_dirs_from<F>(
    home: &std::ffi::OsStr,
    path: &std::ffi::OsStr,
    exists: F,
) -> Vec<String>
where
    F: Fn(&std::path::Path) -> bool,
{
    let home_buf = std::path::PathBuf::from(home);
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for dir in env::split_paths(path) {
        if dir.starts_with(&home_buf) && exists(&dir) {
            let s = dir.display().to_string();
            if seen.insert(s.clone()) {
                out.push(s);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn join_paths<I: IntoIterator<Item = &'static str>>(parts: I) -> OsString {
        let bufs: Vec<std::path::PathBuf> = parts.into_iter().map(Into::into).collect();
        env::join_paths(bufs).unwrap()
    }

    #[test]
    fn keeps_only_home_dirs_that_exist_preserving_order() {
        let home = OsString::from("/home/alice");
        let path = join_paths([
            "/usr/bin",
            "/home/alice/.local/bin",
            "/home/alice/bin",
            "/home/alice/missing",
            "/home/bob/bin", // different user
            "/home/alice/bin", // duplicate
        ]);
        let exists = |p: &std::path::Path| {
            !p.to_string_lossy().contains("missing") && !p.to_string_lossy().contains("/bob/")
        };
        let dirs = user_path_dirs_from(&home, &path, exists);
        assert_eq!(
            dirs,
            vec![
                "/home/alice/.local/bin".to_string(),
                "/home/alice/bin".to_string(),
            ]
        );
    }

    #[test]
    fn empty_when_no_home_dirs_in_path() {
        let dirs = user_path_dirs_from(
            std::ffi::OsStr::new("/home/alice"),
            &join_paths(["/usr/bin", "/usr/local/bin"]),
            |_| true,
        );
        assert!(dirs.is_empty());
    }

    #[test]
    fn prompt_block_includes_path_dirs_when_present() {
        let mut ctx = SystemContext::default();
        ctx.path_dirs = vec!["/home/alice/bin".into(), "/home/alice/.local/bin".into()];
        let block = ctx.as_prompt_block();
        assert!(block.contains("writable PATH dirs:"));
        assert!(block.contains("/home/alice/bin"));
    }

    #[test]
    fn prompt_block_omits_path_dirs_line_when_empty() {
        let ctx = SystemContext::default();
        let block = ctx.as_prompt_block();
        assert!(!block.contains("writable PATH dirs"));
    }
}
