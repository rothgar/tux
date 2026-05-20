//! Confirmation prompts for tools that change host state.
//!
//! Tools call [`confirm`] before doing anything destructive. *How* the
//! question is surfaced depends on where the agent is running:
//!
//! - **In-process CLI** (`tux <prompt>` falling back to a local model):
//!   the running tux process owns the user's terminal, so [`TtyConfirmer`]
//!   opens `/dev/tty` directly.
//! - **systemd-managed daemon** (`tuxd`): there is no controlling tty,
//!   so [`ChannelConfirmer`] forwards the prompt back over the unix
//!   socket; the CLI client asks the human and replies.
//!
//! The active confirmer is installed via a `tokio::task_local!` inside
//! [`with_confirmer`] for the lifetime of one agent turn. Doing it that
//! way keeps the [`crate::tools::Tool`] trait — which is `pub` and
//! implemented all over the codebase — completely unchanged.
//!
//! When no confirmer is installed (most unit tests, ad-hoc users of the
//! tool types) we fall back to [`TtyConfirmer`], which itself declines
//! when `/dev/tty` is unavailable. That preserves the long-standing
//! "no human present → don't change state" semantics.

use async_trait::async_trait;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;

#[async_trait]
pub trait Confirmer: Send + Sync {
    async fn confirm(&self, prompt: &str) -> anyhow::Result<bool>;
}

tokio::task_local! {
    static CURRENT: Arc<dyn Confirmer>;
}

/// Ask the user the question in `prompt` using whatever confirmer the
/// agent installed for this task. Falls back to a [`TtyConfirmer`] when
/// nothing is installed.
pub async fn confirm(prompt: &str) -> anyhow::Result<bool> {
    let confirmer = CURRENT
        .try_with(|c| c.clone())
        .unwrap_or_else(|_| Arc::new(TtyConfirmer) as Arc<dyn Confirmer>);
    confirmer.confirm(prompt).await
}

/// Run `fut` with `confirmer` installed as the active confirmer for any
/// nested [`confirm`] call. Cheap; just sets a task-local.
pub async fn with_confirmer<F, T>(confirmer: Arc<dyn Confirmer>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CURRENT.scope(confirmer, fut).await
}

/// Asks via `/dev/tty` (the controlling terminal of *this* process,
/// regardless of stdin/stdout redirection). Declines if no tty exists.
pub struct TtyConfirmer;

#[async_trait]
impl Confirmer for TtyConfirmer {
    async fn confirm(&self, prompt: &str) -> anyhow::Result<bool> {
        let prompt = prompt.to_string();
        tokio::task::spawn_blocking(move || tty_confirm_blocking(&prompt))
            .await
            .map_err(|e| anyhow::anyhow!("confirm task join: {e}"))?
    }
}

fn tty_confirm_blocking(prompt: &str) -> anyhow::Result<bool> {
    let mut writer = match OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
    {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("no /dev/tty for confirmation ({e}); declining");
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

/// Forwards confirmation prompts through an mpsc channel so the daemon's
/// connection task can serialize them onto the socket. The agent itself
/// stays oblivious to the wire protocol.
///
/// Pairing: each prompt comes with a oneshot reply channel. The daemon
/// writes `{"confirm":"..."}`, reads back `{"answer":...}`, and
/// fulfills the oneshot. If the connection drops mid-question, the
/// oneshot is dropped and the tool sees an error (rather than silently
/// proceeding with `false`, which would be misleading).
pub type ConfirmRequest = (String, tokio::sync::oneshot::Sender<bool>);

pub struct ChannelConfirmer {
    tx: tokio::sync::mpsc::UnboundedSender<ConfirmRequest>,
}

impl ChannelConfirmer {
    pub fn new(tx: tokio::sync::mpsc::UnboundedSender<ConfirmRequest>) -> Self {
        Self { tx }
    }
}

#[async_trait]
impl Confirmer for ChannelConfirmer {
    async fn confirm(&self, prompt: &str) -> anyhow::Result<bool> {
        let (otx, orx) = tokio::sync::oneshot::channel();
        self.tx
            .send((prompt.to_string(), otx))
            .map_err(|_| anyhow::anyhow!("confirm channel closed"))?;
        orx.await
            .map_err(|e| anyhow::anyhow!("confirm response dropped: {e}"))
    }
}
