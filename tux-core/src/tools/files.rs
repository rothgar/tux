//! File tools: find, read, and edit text files.
//!
//! These together let the agent inspect and modify configuration on the
//! host — most importantly its own `~/.local/share/tux/system.json`, but
//! also dotfiles, `/etc/nixos/configuration.nix`, etc.
//!
//! Safety choices:
//!
//! - `find_file` and `read_file` are read-only and need no confirmation.
//! - `read_file` caps payload at 64 KiB and refuses files containing null
//!   bytes (avoids dumping a binary into the model context).
//! - `edit_file` does an *exact-string* search/replace (no regex, no fuzzy
//!   match), requires `old` to appear exactly once (so the model can't
//!   accidentally rewrite the wrong occurrence), and confirms on
//!   `/dev/tty` showing a short diff preview before touching disk. Writes
//!   are atomic (temp file + rename).
//!
//! The model is not constrained to a path allow-list. This is a
//! local-user assistant: the user already has full FS access from their
//! shell, and a path allow-list is brittle (every distro lays files
//! out differently). The /dev/tty confirmation is the safety net.

use super::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::json;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use tokio::process::Command;

const READ_CAP_BYTES: usize = 64 * 1024;
const FIND_DEFAULT_LIMIT: usize = 50;
const FIND_MAX_LIMIT: usize = 500;

#[derive(Default)]
pub struct FindFileTool;

#[derive(Default)]
pub struct ReadFileTool;

#[derive(Default)]
pub struct EditFileTool;

#[derive(Default)]
pub struct WriteFileTool;

// ---- shared helpers -----------------------------------------------------

/// Expand a leading `~/` or bare `~` against `$HOME`. Other paths pass
/// through unchanged.
fn expand_path(s: &str) -> PathBuf {
    if s == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(s)
}

fn on_path(bin: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|d| d.join(bin).is_file())
}

/// Prompt on `/dev/tty` and read a yes/no. Returns `Ok(false)` if there
/// is no controlling tty (we treat "no human present" as "do not edit").
fn confirm_via_tty(prompt: &str) -> anyhow::Result<bool> {
    let mut writer = match OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
    {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("no /dev/tty for confirmation ({e}); declining edit");
            return Ok(false);
        }
    };
    write!(writer, "{prompt}")?;
    writer.flush()?;
    let reader = writer.try_clone()?;
    let mut buf = String::new();
    BufReader::new(reader).read_line(&mut buf)?;
    let ans = buf.trim().to_ascii_lowercase();
    Ok(matches!(ans.as_str(), "y" | "yes"))
}

/// Truncate a snippet for display in a confirmation prompt. Multi-line
/// content is collapsed to first line + `…` so the prompt stays one
/// screen.
fn snippet(s: &str, max_chars: usize) -> String {
    let first_line = s.lines().next().unwrap_or("");
    let trimmed: String = first_line.chars().take(max_chars).collect();
    let mut out = trimmed;
    if first_line.chars().count() > max_chars || s.lines().count() > 1 {
        out.push('…');
    }
    out
}

// ---- find_file ----------------------------------------------------------

#[async_trait]
impl Tool for FindFileTool {
    fn name(&self) -> &'static str {
        "find_file"
    }

    fn description(&self) -> &'static str {
        "Find files matching a glob pattern. Args: \
         {\"pattern\": \"*.rs\", \"root\": \".\" (optional, default cwd), \
         \"limit\": 50 (optional)}. Returns up to `limit` matching paths."
    }

    async fn invoke(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("find_file requires {{\"pattern\":\"...\"}}"))?;
        let root_arg = args.get("root").and_then(|v| v.as_str()).unwrap_or(".");
        let root = expand_path(root_arg);
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| (n as usize).min(FIND_MAX_LIMIT))
            .unwrap_or(FIND_DEFAULT_LIMIT);

        anyhow::ensure!(
            root.exists(),
            "find_file root does not exist: {}",
            root.display()
        );

        // Prefer `fd` (fast, sane defaults — respects .gitignore, hidden
        // off by default). Fall back to portable `find` so the tool
        // still works on minimal hosts.
        let output = if on_path("fd") {
            Command::new("fd")
                .arg("--glob")
                .arg(pattern)
                .arg(root.as_os_str())
                .output()
                .await
                .map_err(|e| anyhow::anyhow!("fd: {e}"))?
        } else {
            Command::new("find")
                .arg(root.as_os_str())
                .arg("-name")
                .arg(pattern)
                .output()
                .await
                .map_err(|e| anyhow::anyhow!("find: {e}"))?
        };

        anyhow::ensure!(
            output.status.success(),
            "find_file failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        let all: Vec<String> = stdout.lines().map(|s| s.to_string()).collect();
        let total = all.len();
        let truncated = total > limit;
        let paths: Vec<String> = all.into_iter().take(limit).collect();

        let summary = if truncated {
            format!(
                "found {total} match(es) for `{pattern}` in {} (showing {limit})",
                root.display()
            )
        } else {
            format!(
                "found {total} match(es) for `{pattern}` in {}",
                root.display()
            )
        };

        Ok(ToolResult {
            tool: "find_file".into(),
            summary,
            data: json!({
                "pattern": pattern,
                "root": root.display().to_string(),
                "matches": paths,
                "total": total,
                "truncated": truncated,
            }),
        })
    }
}

// ---- read_file ----------------------------------------------------------

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a UTF-8 text file. Args: {\"path\": \"...\"}. Caps content at \
         64 KiB and refuses binary files."
    }

    async fn invoke(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("read_file requires {{\"path\":\"...\"}}"))?;
        let path = expand_path(path_str);

        anyhow::ensure!(path.exists(), "file does not exist: {}", path.display());
        anyhow::ensure!(path.is_file(), "not a regular file: {}", path.display());

        let bytes = std::fs::read(&path)?;
        let truncated = bytes.len() > READ_CAP_BYTES;
        let slice = &bytes[..bytes.len().min(READ_CAP_BYTES)];

        anyhow::ensure!(
            !slice.contains(&0u8),
            "file looks binary (contains NUL bytes): {}",
            path.display()
        );

        let content = String::from_utf8(slice.to_vec())
            .map_err(|e| anyhow::anyhow!("file is not valid UTF-8: {e}"))?;
        let lines = content.lines().count();

        let summary = if truncated {
            format!(
                "read {} ({} bytes shown of {}, {} lines)",
                path.display(),
                slice.len(),
                bytes.len(),
                lines
            )
        } else {
            format!(
                "read {} ({} bytes, {} lines)",
                path.display(),
                bytes.len(),
                lines
            )
        };

        Ok(ToolResult {
            tool: "read_file".into(),
            summary,
            data: json!({
                "path": path.display().to_string(),
                "content": content,
                "bytes": bytes.len(),
                "truncated": truncated,
            }),
        })
    }
}

// ---- edit_file ----------------------------------------------------------

/// Pure helper: validate inputs and produce the new file contents.
/// Returns the new contents on success, or an error explaining why.
fn apply_edit(original: &str, old: &str, new: &str) -> anyhow::Result<String> {
    anyhow::ensure!(!old.is_empty(), "edit_file `old` must not be empty");
    anyhow::ensure!(old != new, "edit_file `old` and `new` are identical");

    let count = original.matches(old).count();
    match count {
        0 => anyhow::bail!("`old` text not found in file"),
        1 => Ok(original.replacen(old, new, 1)),
        n => anyhow::bail!(
            "`old` text matches {n} places — make it longer/more specific so it matches once"
        ),
    }
}

/// Write `contents` to `path` atomically: write to a sibling `.tmp`,
/// then rename over the target. On crash mid-write the original survives.
fn atomic_write(path: &Path, contents: &str) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("path has no parent: {}", path.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("path has no file name: {}", path.display()))?;
    let tmp = parent.join(format!(".{}.tux.tmp", file_name.to_string_lossy()));
    std::fs::write(&tmp, contents)
        .map_err(|e| anyhow::anyhow!("write tmp {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| anyhow::anyhow!("rename {} → {}: {e}", tmp.display(), path.display()))?;
    Ok(())
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn description(&self) -> &'static str {
        "Edit a text file by exact-string search/replace. Args: \
         {\"path\": \"...\", \"old\": \"...\", \"new\": \"...\"}. The \
         `old` text must appear exactly once. The user is asked to \
         confirm on the terminal before the file is changed."
    }

    async fn invoke(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("edit_file requires `path`"))?;
        let old = args
            .get("old")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("edit_file requires `old`"))?;
        let new = args
            .get("new")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("edit_file requires `new`"))?;

        let path = expand_path(path_str);
        anyhow::ensure!(path.exists(), "file does not exist: {}", path.display());
        anyhow::ensure!(path.is_file(), "not a regular file: {}", path.display());

        let original = std::fs::read_to_string(&path)?;
        let updated = apply_edit(&original, old, new)?;

        let prompt = format!(
            "tux: edit {}\n  - {}\n  + {}\nproceed? [y/N]: ",
            path.display(),
            snippet(old, 60),
            snippet(new, 60),
        );
        if !confirm_via_tty(&prompt)? {
            return Ok(ToolResult {
                tool: "edit_file".into(),
                summary: format!("edit of {} cancelled", path.display()),
                data: json!({ "cancelled": true, "path": path.display().to_string() }),
            });
        }

        atomic_write(&path, &updated)?;
        let bytes_delta = updated.len() as i64 - original.len() as i64;

        Ok(ToolResult {
            tool: "edit_file".into(),
            summary: format!("edited {} ({:+} bytes)", path.display(), bytes_delta),
            data: json!({
                "path": path.display().to_string(),
                "bytes": updated.len(),
                "delta": bytes_delta,
            }),
        })
    }
}

// ---- write_file ---------------------------------------------------------

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Create or overwrite a file. Args: \
         {\"path\": \"...\", \"content\": \"...\", \
         \"executable\": false (optional, sets +x for scripts), \
         \"overwrite\": false (optional, must be true to replace an \
         existing file)}. Confirms on the terminal before writing."
    }

    async fn invoke(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("write_file requires `path`"))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("write_file requires `content` (string)"))?;
        let executable = args
            .get("executable")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let overwrite = args
            .get("overwrite")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let path = expand_path(path_str);

        if path.exists() {
            anyhow::ensure!(path.is_file(), "exists and is not a file: {}", path.display());
            anyhow::ensure!(
                overwrite,
                "{} already exists — pass {{\"overwrite\": true}} to replace it",
                path.display()
            );
        }
        if let Some(parent) = path.parent() {
            anyhow::ensure!(
                parent.as_os_str().is_empty() || parent.exists(),
                "parent directory does not exist: {}",
                parent.display()
            );
        }

        let action = if path.exists() { "overwrite" } else { "create" };
        let mode_label = if executable { " +x" } else { "" };
        let prompt = format!(
            "tux: {action}{mode_label} {} ({} bytes)\nproceed? [y/N]: ",
            path.display(),
            content.len(),
        );
        if !confirm_via_tty(&prompt)? {
            return Ok(ToolResult {
                tool: "write_file".into(),
                summary: format!("write of {} cancelled", path.display()),
                data: json!({ "cancelled": true, "path": path.display().to_string() }),
            });
        }

        atomic_write(&path, content)?;

        if executable {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&path, perms)
                .map_err(|e| anyhow::anyhow!("chmod +x {}: {e}", path.display()))?;
        }

        Ok(ToolResult {
            tool: "write_file".into(),
            summary: format!(
                "{action}d {} ({} bytes{})",
                path.display(),
                content.len(),
                if executable { ", executable" } else { "" }
            ),
            data: json!({
                "path": path.display().to_string(),
                "bytes": content.len(),
                "executable": executable,
                "overwritten": action == "overwrite",
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    // ---- expand_path ---------------------------------------------------

    #[test]
    fn expand_passes_absolute_through() {
        assert_eq!(expand_path("/etc/passwd"), PathBuf::from("/etc/passwd"));
    }

    #[test]
    fn expand_handles_relative() {
        assert_eq!(expand_path("foo/bar"), PathBuf::from("foo/bar"));
    }

    #[test]
    fn expand_resolves_tilde() {
        // Use a known HOME so the test is hermetic.
        std::env::set_var("HOME", "/home/test");
        assert_eq!(expand_path("~/foo"), PathBuf::from("/home/test/foo"));
        assert_eq!(expand_path("~"), PathBuf::from("/home/test"));
        // No tilde stays put.
        assert_eq!(expand_path("/abs"), PathBuf::from("/abs"));
    }

    // ---- apply_edit ----------------------------------------------------

    #[test]
    fn apply_edit_replaces_unique_occurrence() {
        let r = apply_edit("hello world", "world", "tux").unwrap();
        assert_eq!(r, "hello tux");
    }

    #[test]
    fn apply_edit_rejects_empty_old() {
        let err = apply_edit("abc", "", "x").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn apply_edit_rejects_identical_old_new() {
        let err = apply_edit("abc", "a", "a").unwrap_err();
        assert!(err.to_string().contains("identical"));
    }

    #[test]
    fn apply_edit_rejects_missing_old() {
        let err = apply_edit("hello", "world", "tux").unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn apply_edit_rejects_ambiguous_old() {
        // "fn " appears twice — model must pick a longer anchor.
        let src = "fn foo() {}\nfn bar() {}\n";
        let err = apply_edit(src, "fn ", "pub fn ").unwrap_err();
        assert!(err.to_string().contains("matches 2"));
    }

    #[test]
    fn apply_edit_handles_multiline_replacement() {
        let src = "line1\nold block\nline3\n";
        let r = apply_edit(src, "old block", "new\nblock\nspans").unwrap();
        assert_eq!(r, "line1\nnew\nblock\nspans\nline3\n");
    }

    // ---- atomic_write --------------------------------------------------

    #[test]
    fn atomic_write_round_trip() {
        let dir = tempdir();
        let path = dir.path().join("hello.txt");
        std::fs::write(&path, "before").unwrap();
        atomic_write(&path, "after").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after");
        // No leftover .tmp sibling.
        let stray: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(stray.len(), 1, "exactly one file remains");
    }

    // ---- snippet -------------------------------------------------------

    #[test]
    fn snippet_truncates_long_first_line() {
        let s = snippet(&"x".repeat(200), 10);
        assert_eq!(s, "xxxxxxxxxx…");
    }

    #[test]
    fn snippet_marks_multiline() {
        let s = snippet("first\nsecond", 80);
        assert_eq!(s, "first…");
    }

    #[test]
    fn snippet_short_single_line_unchanged() {
        let s = snippet("hello", 80);
        assert_eq!(s, "hello");
    }

    // ---- minimal tempdir without an extra crate ------------------------

    struct TmpDir(PathBuf);
    impl TmpDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir() -> TmpDir {
        let mut p = std::env::temp_dir();
        let name = format!("tux-test-{}-{}", std::process::id(), unique());
        p.push(name);
        std::fs::create_dir_all(&p).unwrap();
        TmpDir(p)
    }
    fn unique() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    // Sanity: read_file via tokio runtime + a real on-disk file.
    #[tokio::test]
    async fn read_file_round_trip() {
        let dir = tempdir();
        let path = dir.path().join("note.txt");
        std::fs::write(&path, "alpha\nbeta\n").unwrap();

        let tool = ReadFileTool;
        let res = tool
            .invoke(json!({ "path": path.display().to_string() }))
            .await
            .unwrap();
        assert_eq!(res.data["content"].as_str().unwrap(), "alpha\nbeta\n");
        assert_eq!(res.data["truncated"].as_bool(), Some(false));
    }

    #[tokio::test]
    async fn write_file_refuses_overwrite_without_flag() {
        let dir = tempdir();
        let path = dir.path().join("existing.txt");
        std::fs::write(&path, "old").unwrap();

        let tool = WriteFileTool;
        let err = tool
            .invoke(json!({
                "path": path.display().to_string(),
                "content": "new",
            }))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("already exists"),
            "got: {err}"
        );
        // Untouched on disk.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "old");
    }

    #[tokio::test]
    async fn write_file_rejects_missing_parent() {
        let dir = tempdir();
        let path = dir.path().join("nope/missing/file.txt");
        let tool = WriteFileTool;
        let err = tool
            .invoke(json!({
                "path": path.display().to_string(),
                "content": "x",
            }))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("parent directory"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn read_file_refuses_binary() {
        let dir = tempdir();
        let path = dir.path().join("blob.bin");
        std::fs::write(&path, [0u8, 1, 2, 0, 3]).unwrap();
        let tool = ReadFileTool;
        let err = tool
            .invoke(json!({ "path": path.display().to_string() }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("binary"), "got: {err}");
    }
}
