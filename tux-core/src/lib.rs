//! tux-core: agent loop, tools, system context, and pluggable model backends.

pub mod agent;
pub mod backend;
pub mod config;
pub mod context;
pub mod knowledge;
pub mod models;
pub mod tools;

pub use agent::{Agent, Conversation};
pub use backend::{Backend, BackendKind, ChatMessage, MockBackend, Role};
pub use config::{BackendConfig, BackendType, InferenceConfig, TuxConfig};
pub use context::SystemContext;
pub use tools::{Tool, ToolRegistry, ToolResult};
