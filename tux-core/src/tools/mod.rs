//! Tool registry. A `Tool` is something the agent can invoke on the host
//! (take a screenshot, search files, change a setting, etc.). For v0 the
//! agent loop dispatches tools by name from explicit user commands; later
//! the model will be prompted to emit tool-call JSON.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

pub mod clipboard;
pub mod files;
pub mod install;
pub mod screenshot;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool: String,
    pub summary: String,
    /// Optional structured payload (e.g. file path of a screenshot).
    pub data: serde_json::Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    async fn invoke(&self, args: serde_json::Value) -> anyhow::Result<ToolResult>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<&'static str, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register(Arc::new(screenshot::ScreenshotTool::default()));
        r.register(Arc::new(clipboard::ClipboardReadTool::default()));
        r.register(Arc::new(clipboard::ClipboardWriteTool::default()));
        r.register(Arc::new(install::InstallPackageTool::default()));
        r.register(Arc::new(files::FindFileTool::default()));
        r.register(Arc::new(files::ReadFileTool::default()));
        r.register(Arc::new(files::EditFileTool::default()));
        r.register(Arc::new(files::WriteFileTool::default()));
        r
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name(), tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list(&self) -> Vec<(&'static str, &'static str)> {
        self.tools
            .values()
            .map(|t| (t.name(), t.description()))
            .collect()
    }
}
