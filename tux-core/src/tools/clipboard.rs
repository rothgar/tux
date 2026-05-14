//! Clipboard tools — read/write the user's text selection.
//!
//! Picks a backend command based on the session type:
//! - wayland → `wl-paste` / `wl-copy`     (wl-clipboard package)
//! - x11     → `xclip -selection clipboard`
//!
//! Image clipboard is intentionally deferred; it only becomes useful once
//! we have a vision-capable model that can do something with the bytes.

use super::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::json;
use std::env;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

#[derive(Default)]
pub struct ClipboardReadTool;

#[derive(Default)]
pub struct ClipboardWriteTool;

fn session_is_wayland() -> bool {
    env::var("XDG_SESSION_TYPE").as_deref() == Ok("wayland")
}

#[async_trait]
impl Tool for ClipboardReadTool {
    fn name(&self) -> &'static str {
        "clipboard_read"
    }

    fn description(&self) -> &'static str {
        "Read the current text contents of the user's clipboard."
    }

    async fn invoke(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let output = if session_is_wayland() {
            Command::new("wl-paste")
                .arg("--no-newline")
                .output()
                .await
                .map_err(|e| anyhow::anyhow!("wl-paste: {e}"))?
        } else {
            Command::new("xclip")
                .args(["-selection", "clipboard", "-o"])
                .output()
                .await
                .map_err(|e| anyhow::anyhow!("xclip: {e}"))?
        };

        anyhow::ensure!(
            output.status.success(),
            "clipboard read failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );

        let text = String::from_utf8_lossy(&output.stdout).to_string();
        let preview = text.chars().take(80).collect::<String>();
        Ok(ToolResult {
            tool: "clipboard_read".into(),
            summary: format!("clipboard ({} chars): {}", text.len(), preview),
            data: json!({ "text": text }),
        })
    }
}

#[async_trait]
impl Tool for ClipboardWriteTool {
    fn name(&self) -> &'static str {
        "clipboard_write"
    }

    fn description(&self) -> &'static str {
        "Replace the user's clipboard with the given text. Args: {\"text\": \"...\"}"
    }

    async fn invoke(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("clipboard_write requires {{\"text\":\"...\"}}"))?;

        let mut child = if session_is_wayland() {
            Command::new("wl-copy")
                .stdin(Stdio::piped())
                .spawn()
                .map_err(|e| anyhow::anyhow!("wl-copy: {e}"))?
        } else {
            Command::new("xclip")
                .args(["-selection", "clipboard"])
                .stdin(Stdio::piped())
                .spawn()
                .map_err(|e| anyhow::anyhow!("xclip: {e}"))?
        };

        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(text.as_bytes()).await?;
        }
        let status = child.wait().await?;
        anyhow::ensure!(status.success(), "clipboard write failed ({status})");

        Ok(ToolResult {
            tool: "clipboard_write".into(),
            summary: format!("wrote {} chars to clipboard", text.len()),
            data: json!({ "bytes_written": text.len() }),
        })
    }
}
