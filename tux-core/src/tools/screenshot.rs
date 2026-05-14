//! Screenshot tool. Picks a backend command based on the session type:
//! - wayland → `grim`
//! - x11     → `scrot`
//!
//! Both are provided by the nix dev shell. Output is written to the user's
//! cache dir as `tux/screenshots/<timestamp>.png` and the path is returned.

use super::{Tool, ToolResult};
use async_trait::async_trait;
use directories::ProjectDirs;
use serde_json::json;
use std::env;
use std::path::PathBuf;
use tokio::process::Command;

#[derive(Default)]
pub struct ScreenshotTool;

impl ScreenshotTool {
    fn output_path() -> anyhow::Result<PathBuf> {
        let dirs = ProjectDirs::from("dev", "tux", "tux")
            .ok_or_else(|| anyhow::anyhow!("could not resolve project dirs"))?;
        let dir = dirs.cache_dir().join("screenshots");
        std::fs::create_dir_all(&dir)?;
        let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S");
        Ok(dir.join(format!("{ts}.png")))
    }
}

#[async_trait]
impl Tool for ScreenshotTool {
    fn name(&self) -> &'static str {
        "screenshot"
    }

    fn description(&self) -> &'static str {
        "Capture the current screen to a PNG and return its path."
    }

    async fn invoke(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path = Self::output_path()?;
        let session = env::var("XDG_SESSION_TYPE").unwrap_or_default();

        let status = match session.as_str() {
            "wayland" => Command::new("grim").arg(&path).status().await?,
            // default to x11 tooling otherwise
            _ => Command::new("scrot").arg(&path).status().await?,
        };

        anyhow::ensure!(
            status.success(),
            "screenshot command exited with {status} (session={session})"
        );

        Ok(ToolResult {
            tool: "screenshot".into(),
            summary: format!("captured screenshot to {}", path.display()),
            data: json!({ "path": path }),
        })
    }
}
