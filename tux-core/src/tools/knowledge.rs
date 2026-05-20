//! `set_knowledge` tool — lets the agent edit its *own* persisted
//! configuration in `$XDG_DATA_HOME/tux/system.json` (the file that
//! holds the package-manager commands, escalation tool, service
//! manager, etc. — see [`DistroKnowledge`]).
//!
//! When the user says "tux update *your* update_cmd to ..." the model
//! should emit:
//!
//!   <tool name="set_knowledge">{"field":"update_cmd","value":"..."}</tool>
//!
//! and the agent will mutate `system.json` in place. The next request
//! reloads the cache so the change takes effect immediately.

use super::{Tool, ToolResult};
use crate::knowledge;
use async_trait::async_trait;
use serde_json::json;
use std::fs;

/// Construct the tool with the host's `distro_id` so it can synthesize a
/// starting `system.json` if one doesn't exist yet (first-time write
/// after `tux init` was skipped, etc.).
pub struct SetKnowledgeTool {
    distro_id: Option<String>,
}

impl SetKnowledgeTool {
    pub fn new(distro_id: Option<String>) -> Self {
        Self { distro_id }
    }
}

impl Default for SetKnowledgeTool {
    fn default() -> Self {
        // Best-effort distro detection from /etc/os-release. Same
        // parsing as SystemContext::detect, kept inline so the tool
        // doesn't need a SystemContext handle.
        let id = fs::read_to_string("/etc/os-release").ok().and_then(|s| {
            for line in s.lines() {
                if let Some(v) = line.strip_prefix("ID=") {
                    return Some(v.trim_matches('"').to_string());
                }
            }
            None
        });
        Self { distro_id: id }
    }
}

#[async_trait]
impl Tool for SetKnowledgeTool {
    fn name(&self) -> &'static str {
        "set_knowledge"
    }

    fn description(&self) -> &'static str {
        "Update one field of tux's own persisted configuration \
         (system.json). Args: {\"field\":\"update_cmd\",\"value\":\"...\"}. \
         Valid fields: package_manager, install_cmd, remove_cmd, \
         search_cmd, update_cmd, escalation, service_manager, \
         config_paths (list), notes (list). Use this when the user asks \
         to change \"your\" config / knowledge / commands."
    }

    async fn invoke(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let field = args
            .get("field")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("set_knowledge: missing string arg `field`"))?;
        let value = args
            .get("value")
            .ok_or_else(|| anyhow::anyhow!("set_knowledge: missing arg `value`"))?;

        let (path, knowledge) =
            knowledge::update_field(self.distro_id.as_deref(), field, value)?;

        Ok(ToolResult {
            tool: "set_knowledge".into(),
            summary: format!(
                "updated {field} in {} (now: {})",
                path.display(),
                serde_json::to_string(value).unwrap_or_else(|_| "?".into())
            ),
            data: json!({
                "path": path,
                "field": field,
                "knowledge": knowledge,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_unknown_field() {
        let tool = SetKnowledgeTool::new(Some("arch".into()));
        let res = tool
            .invoke(json!({"field": "bogus", "value": "x"}))
            .await;
        assert!(res.is_err(), "unknown field must error");
    }

    #[tokio::test]
    async fn missing_args_error() {
        let tool = SetKnowledgeTool::new(Some("arch".into()));
        assert!(tool.invoke(json!({})).await.is_err());
        assert!(tool.invoke(json!({"field": "update_cmd"})).await.is_err());
    }
}
