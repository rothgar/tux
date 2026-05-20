//! Natural-language agent loop. Users type or speak free-form requests;
//! the model decides whether to answer directly or invoke a tool by
//! emitting a `<tool name="...">{...}</tool>` tag. The agent parses any
//! tool call, runs it, and feeds the result back for a final answer.
//!
//! No slash commands. No magic prefixes. Just sentences.

use crate::backend::{Backend, ChatMessage, CompletionRequest, Role};
use crate::context::SystemContext;
use crate::tools::confirm::{with_confirmer, Confirmer, TtyConfirmer};
use crate::tools::{ToolRegistry, ToolResult};
use std::sync::Arc;

const MAX_TOOL_HOPS: usize = 3;

/// Persistent chat history for a single user session.
///
/// Owned by callers (the daemon keeps a single global one; in-process
/// CLI users get a fresh one per invocation). Holds the system prompt
/// (added lazily on the first turn so it can pick up the agent's
/// fully-rendered prompt) plus every user / assistant / tool message
/// the agent has produced. Reuse is what makes the daemon faster than
/// loading a model per request.
#[derive(Default)]
pub struct Conversation {
    messages: Vec<ChatMessage>,
}

impl Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.messages.clear();
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

pub struct Agent {
    backend: Arc<dyn Backend>,
    tools: ToolRegistry,
    context: SystemContext,
}

/// What the agent ultimately produced for one user turn. `tool_calls`
/// records any tool invocations the model made along the way so the UI
/// can show them as a trace.
#[derive(Debug)]
pub struct AgentReply {
    pub text: String,
    pub tool_calls: Vec<ToolResult>,
}

impl Agent {
    pub fn new(backend: Arc<dyn Backend>, tools: ToolRegistry, context: SystemContext) -> Self {
        Self {
            backend,
            tools,
            context,
        }
    }

    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    pub fn context(&self) -> &SystemContext {
        &self.context
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Handle one user turn end-to-end, including tool invocation hops.
    /// Single-shot convenience wrapper used by tests and the in-process
    /// CLI path: spins up a fresh [`Conversation`] every call so there
    /// is no cross-turn state. Uses a [`TtyConfirmer`] so destructive
    /// tool calls still ask the human (and decline if no tty exists).
    pub async fn handle(&self, input: &str) -> anyhow::Result<AgentReply> {
        let mut conv = Conversation::new();
        self.turn(&mut conv, input, Arc::new(TtyConfirmer)).await
    }

    /// Handle one user turn against a persistent [`Conversation`].
    /// Appends the new user message (and the system prompt the very
    /// first time), runs the model + tool-dispatch loop, and pushes
    /// every assistant / tool message back onto the conversation so the
    /// next call reuses them. The `confirmer` is installed as a
    /// task-local for the lifetime of this turn so any tool that asks
    /// for confirmation routes through the right channel (tty for the
    /// CLI, socket forwarding for the daemon).
    pub async fn turn(
        &self,
        conv: &mut Conversation,
        input: &str,
        confirmer: Arc<dyn Confirmer>,
    ) -> anyhow::Result<AgentReply> {
        with_confirmer(confirmer, self.turn_inner(conv, input)).await
    }

    async fn turn_inner(
        &self,
        conv: &mut Conversation,
        input: &str,
    ) -> anyhow::Result<AgentReply> {
        if conv.messages.is_empty() {
            conv.messages.push(ChatMessage {
                role: Role::System,
                content: self.system_prompt(),
            });
        }
        conv.messages.push(ChatMessage {
            role: Role::User,
            content: input.trim().to_string(),
        });

        let mut tool_calls = Vec::new();
        let mut pending_images: Vec<std::path::PathBuf> = Vec::new();

        for _hop in 0..MAX_TOOL_HOPS {
            let req = CompletionRequest::new(conv.messages.clone())
                .with_images(std::mem::take(&mut pending_images));
            let resp = self.backend.complete(req).await?;
            let text = resp.text.trim().to_string();

            match parse_tool_call(&text) {
                Some(call) => {
                    let tool = self.tools.get(&call.name).ok_or_else(|| {
                        anyhow::anyhow!("model called unknown tool: {}", call.name)
                    })?;
                    let result = tool.invoke(call.args).await?;

                    // Tools that produce images report a path under
                    // `data.path` or `data.image_path`. Carry those into
                    // the next backend turn so the model can actually see
                    // what the tool produced (vision models only — text
                    // backends ignore the field).
                    if let Some(p) = extract_image_path(&result.data) {
                        pending_images.push(p);
                    }

                    conv.messages.push(ChatMessage {
                        role: Role::Assistant,
                        content: text,
                    });
                    conv.messages.push(ChatMessage {
                        role: Role::Tool,
                        content: serde_json::to_string(&serde_json::json!({
                            "tool": result.tool,
                            "summary": result.summary,
                            "data": result.data,
                        }))
                        .unwrap_or_else(|_| result.summary.clone()),
                    });
                    tool_calls.push(result);
                    continue;
                }
                None => {
                    conv.messages.push(ChatMessage {
                        role: Role::Assistant,
                        content: text.clone(),
                    });
                    return Ok(AgentReply { text, tool_calls });
                }
            }
        }

        // Hit the hop cap — return what we have with a note. The
        // assistant turn that exhausted the budget already lives in
        // `conv.messages` as the final tool's predecessor; we don't add
        // an extra synthetic assistant message because the model didn't
        // actually produce one.
        Ok(AgentReply {
            text: "(reached tool-call limit without a final answer)".into(),
            tool_calls,
        })
    }

    fn system_prompt(&self) -> String {
        let mut tool_lines = String::new();
        for (name, desc) in self.tools.list() {
            tool_lines.push_str(&format!("- {name}: {desc}\n"));
        }

        format!(
            "You are tux, a helpful local assistant on a Linux machine. \
Answer naturally in plain prose. Be concise. When you suggest commands, \
prefer ones appropriate for the host described below.

When the user refers to \"your\" / \"yourself\" / \"your config\" / \
\"your knowledge\" / \"your <field>\" (e.g. \"your update_cmd\", \"your \
install command\"), they mean YOUR OWN persisted configuration — the \
fields shown in the \"Package management & system commands\" block below, \
which are stored at $XDG_DATA_HOME/tux/system.json. To change one of \
those, call the `set_knowledge` tool. Do NOT refuse with \"I don't have \
access to my internal variables\" — you do, via that tool.

When you need to use a tool, output ONLY the tool call on a single line \
with no other text, in this exact format:
<tool name=\"NAME\">JSON_ARGS</tool>

Available tools:
{tool_lines}
After a tool runs you will receive its result and should then write the \
final natural-language answer for the user.

{}",
            self.context.as_prompt_block()
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedToolCall {
    name: String,
    args: serde_json::Value,
}

/// Look for an image path inside a tool's `data` payload. Recognized keys:
/// `image_path`, `path` (when the value looks like an image file by
/// extension). Returns the first match.
fn extract_image_path(data: &serde_json::Value) -> Option<std::path::PathBuf> {
    fn looks_like_image(s: &str) -> bool {
        let lower = s.to_ascii_lowercase();
        [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"]
            .iter()
            .any(|ext| lower.ends_with(ext))
    }
    if let Some(s) = data.get("image_path").and_then(|v| v.as_str()) {
        return Some(s.into());
    }
    if let Some(s) = data.get("path").and_then(|v| v.as_str()) {
        if looks_like_image(s) {
            return Some(s.into());
        }
    }
    None
}

/// Look for `<tool name="NAME">JSON</tool>` anywhere in the text. The model
/// is asked to emit ONLY this on tool turns, but small models sometimes add
/// stray whitespace or commentary, so we scan rather than match the whole
/// string.
fn parse_tool_call(text: &str) -> Option<ParsedToolCall> {
    let start = text.find("<tool")?;
    let after = &text[start..];
    let name_key = after.find("name=\"")?;
    let name_start = name_key + "name=\"".len();
    let name_end = name_start + after[name_start..].find('"')?;
    let name = after[name_start..name_end].to_string();

    let gt = name_end + after[name_end..].find('>')?;
    let body_start = gt + 1;
    let body_end = body_start + after[body_start..].find("</tool>")?;
    let body = after[body_start..body_end].trim();

    let args = if body.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_str(body).unwrap_or(serde_json::Value::String(body.into()))
    };

    Some(ParsedToolCall { name, args })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ScriptedBackend;
    use crate::tools::{Tool, ToolResult};
    use async_trait::async_trait;
    use std::sync::Arc;

    // ---- parse_tool_call ---------------------------------------------------

    #[test]
    fn parses_basic_tool_call() {
        let p = parse_tool_call(r#"<tool name="screenshot">{}</tool>"#).unwrap();
        assert_eq!(p.name, "screenshot");
        assert_eq!(p.args, serde_json::json!({}));
    }

    #[test]
    fn parses_tool_call_with_args() {
        let p = parse_tool_call(r#"<tool name="find_file">{"glob":"*.rs"}</tool>"#).unwrap();
        assert_eq!(p.name, "find_file");
        assert_eq!(p.args, serde_json::json!({ "glob": "*.rs" }));
    }

    #[test]
    fn no_tool_call_returns_none() {
        assert!(parse_tool_call("just a normal answer").is_none());
    }

    #[test]
    fn parses_tool_call_with_surrounding_prose() {
        // Small models sometimes emit a sentence before the call.
        let p =
            parse_tool_call("Sure, let me check.\n<tool name=\"echo\">{}</tool>\nDone.")
                .unwrap();
        assert_eq!(p.name, "echo");
    }

    // ---- agent loop --------------------------------------------------------

    /// Tool used in tests: records invocations, returns a deterministic
    /// payload. No I/O so tests are hermetic.
    struct EchoTool {
        invocations: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    }

    impl EchoTool {
        fn new() -> (Self, Arc<std::sync::Mutex<Vec<serde_json::Value>>>) {
            let log = Arc::new(std::sync::Mutex::new(Vec::new()));
            (
                Self {
                    invocations: log.clone(),
                },
                log,
            )
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "Echo arguments back."
        }
        async fn invoke(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
            self.invocations.lock().unwrap().push(args.clone());
            Ok(ToolResult {
                tool: "echo".into(),
                summary: "echoed".into(),
                data: args,
            })
        }
    }

    fn registry_with_echo(log: Arc<std::sync::Mutex<Vec<serde_json::Value>>>) -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(EchoTool { invocations: log }));
        r
    }

    fn agent_with(
        responses: impl IntoIterator<Item = &'static str>,
        tools: ToolRegistry,
    ) -> (Agent, Arc<ScriptedBackend>) {
        let backend = Arc::new(ScriptedBackend::new(responses));
        let agent = Agent::new(backend.clone(), tools, SystemContext::default());
        (agent, backend)
    }

    #[tokio::test]
    async fn direct_answer_passes_through() {
        // Simulates: "install vim" → model returns prose, no tool call.
        let (echo, _) = EchoTool::new();
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(echo));
        let (agent, backend) = agent_with(["sudo nix-env -iA nixos.vim"], tools);

        let reply = agent.handle("install vim").await.unwrap();

        assert_eq!(reply.text, "sudo nix-env -iA nixos.vim");
        assert!(reply.tool_calls.is_empty());
        assert_eq!(backend.calls().len(), 1, "single round-trip, no hops");
    }

    #[tokio::test]
    async fn single_tool_call_then_final_answer() {
        // Simulates: "echo hi" → model emits tool call → tool runs →
        //            model receives result → final prose.
        let (_echo, log) = EchoTool::new();
        let tools = registry_with_echo(log.clone());
        let (agent, backend) = agent_with(
            [
                r#"<tool name="echo">{"msg":"hi"}</tool>"#,
                "I echoed your message back.",
            ],
            tools,
        );

        let reply = agent.handle("echo hi for me").await.unwrap();

        assert_eq!(reply.text, "I echoed your message back.");
        assert_eq!(reply.tool_calls.len(), 1);
        assert_eq!(reply.tool_calls[0].tool, "echo");
        assert_eq!(log.lock().unwrap().len(), 1, "tool was actually invoked");
        assert_eq!(backend.calls().len(), 2, "one hop: call + final");

        // The second backend call must include the tool result as a `tool`
        // role message so the model can ground its final answer.
        let second = &backend.calls()[1];
        assert!(second
            .messages
            .iter()
            .any(|m| matches!(m.role, Role::Tool) && m.content.contains("echoed")));
    }

    #[tokio::test]
    async fn multi_hop_within_limit() {
        let (_echo, log) = EchoTool::new();
        let tools = registry_with_echo(log.clone());
        let (agent, _) = agent_with(
            [
                r#"<tool name="echo">{}</tool>"#,
                r#"<tool name="echo">{}</tool>"#,
                "done after two tool hops",
            ],
            tools,
        );

        let reply = agent.handle("do two things").await.unwrap();

        assert_eq!(reply.text, "done after two tool hops");
        assert_eq!(reply.tool_calls.len(), 2);
        assert_eq!(log.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn hop_limit_returns_note() {
        // Model never gives a final answer — keeps calling tools.
        let (_echo, _log) = EchoTool::new();
        let tools = registry_with_echo(Arc::new(std::sync::Mutex::new(Vec::new())));
        let (agent, _) = agent_with(
            std::iter::repeat(r#"<tool name="echo">{}</tool>"#).take(MAX_TOOL_HOPS + 2),
            tools,
        );

        let reply = agent.handle("loop forever").await.unwrap();

        assert!(reply.text.contains("limit"), "got: {}", reply.text);
        assert_eq!(reply.tool_calls.len(), MAX_TOOL_HOPS);
    }

    #[tokio::test]
    async fn unknown_tool_errors() {
        let (_echo, _log) = EchoTool::new();
        let tools = registry_with_echo(Arc::new(std::sync::Mutex::new(Vec::new())));
        let (agent, _) = agent_with(
            [r#"<tool name="ssh_into_my_server">{}</tool>"#],
            tools,
        );

        let err = agent.handle("do something dangerous").await.unwrap_err();
        assert!(
            err.to_string().contains("ssh_into_my_server"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn system_prompt_lists_tools_and_includes_host_facts() {
        // Common-request grounding: the model needs to know what tools
        // exist and what distro it's on. This guards against accidentally
        // dropping either from the prompt.
        let (_echo, log) = EchoTool::new();
        let tools = registry_with_echo(log);

        let backend = Arc::new(ScriptedBackend::new(["ok"]));
        let mut ctx = SystemContext::default();
        ctx.distro_pretty_name = Some("Arch Linux".into());
        ctx.desktop = Some("KDE".into());
        let agent = Agent::new(backend.clone(), tools, ctx);

        let _ = agent.handle("hi").await.unwrap();

        let req = &backend.calls()[0];
        let system = req
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .expect("system message present");
        assert!(system.content.contains("- echo:"), "tool listed");
        assert!(system.content.contains("Arch Linux"), "distro in prompt");
        assert!(system.content.contains("KDE"), "DE in prompt");
        assert!(
            system.content.contains("<tool name="),
            "tool-call format in prompt"
        );
    }
}
