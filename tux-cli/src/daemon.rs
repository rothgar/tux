//! `tux daemon` — run a long-lived process that holds the model in memory
//! so each `tux <prompt>` invocation skips the 2–5 s cold load.
//!
//! Wire protocol: newline-delimited JSON over a unix socket at
//! `$XDG_RUNTIME_DIR/tux.sock`. The first line on a connection is a
//! request; the daemon may then write zero or more `{"confirm":"..."}`
//! lines (each one expects a `{"answer":true|false}` line back from the
//! client) before sending the final reply line and closing.
//!
//! Requests:
//!   {"prompt": "..."}      → run the agent against the persistent
//!                            conversation; daemon may interleave confirms
//!   {"reset": true}        → clear conversation history; ack with text
//!   {"shutdown": true}     → graceful exit (no confirms)
//!   {"shutdown": false}    → noop / liveness probe
//!
//! Final responses (one per request, on the same connection):
//!   {"text": "...", "tool_calls": [...]}
//!   {"error": "..."}
//!
//! The CLI auto-detects the socket; if it's not there or doesn't answer
//! within a short timeout, the CLI falls back to an in-process model load.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, Notify};
use tux_core::tools::confirm::ChannelConfirmer;
use tux_core::tools::ToolResult;
use tux_core::{Agent, Conversation};

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Request {
    Prompt { prompt: String },
    Reset { reset: bool },
    Shutdown { shutdown: bool },
}

/// Server → client message streamed *before* the final reply when a tool
/// needs the human's OK. The CLI prints the prompt, reads y/N from
/// `/dev/tty`, and replies with `ConfirmAnswer`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ConfirmRecord {
    pub confirm: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfirmAnswer {
    pub answer: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReplyPayload {
    pub text: String,
    pub tool_calls: Vec<ToolResult>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Response {
    Ok(ReplyPayload),
    Err { error: String },
}

pub fn socket_path() -> PathBuf {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            // Avoid pulling libc just for getuid; fall back to a per-user
            // /tmp dir keyed off USER (sufficient for the no-XDG case).
            let user = std::env::var("USER").unwrap_or_else(|_| "default".into());
            PathBuf::from(format!("/tmp/tux-{user}"))
        });
    let _ = std::fs::create_dir_all(&runtime);
    runtime.join("tux.sock")
}

// ---------------------------------------------------------------------------
// server
// ---------------------------------------------------------------------------

/// Per-process conversation state shared across every connection.
/// Single-user, single-session: more elaborate routing can come later.
struct Session {
    agent: Agent,
    conv: Mutex<Conversation>,
}

pub async fn serve(agent: Agent) -> Result<()> {
    let path = socket_path();
    if path.exists() {
        // Stale socket from a previous run? Try to connect; if no one
        // answers, remove it. If someone *does* answer, refuse to start.
        if try_ping(&path).await.is_ok() {
            anyhow::bail!(
                "another tuxd is already serving at {}",
                path.display()
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    let listener =
        UnixListener::bind(&path).with_context(|| format!("bind {}", path.display()))?;

    // Restrict to user-only; UnixListener::bind respects umask but we
    // belt-and-suspenders chmod 0600.
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("chmod 600 {}", path.display()))?;

    eprintln!("tuxd listening on {}", path.display());

    let session = Arc::new(Session {
        agent,
        conv: Mutex::new(Conversation::new()),
    });
    let shutdown = Arc::new(Notify::new());

    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                eprintln!("tuxd shutting down");
                break;
            }
            accept = listener.accept() => {
                let (stream, _) = accept.context("accept")?;
                let session = session.clone();
                let shutdown = shutdown.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle(stream, session, shutdown).await {
                        eprintln!("connection error: {e:#}");
                    }
                });
            }
        }
    }

    let _ = std::fs::remove_file(&path);
    Ok(())
}

async fn handle(stream: UnixStream, session: Arc<Session>, shutdown: Arc<Notify>) -> Result<()> {
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut writer = write_half;

    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Ok(());
    }

    let req: Request = serde_json::from_str(line.trim())
        .map_err(|e| anyhow::anyhow!("bad request json: {e}"))?;

    let response = match req {
        Request::Shutdown { shutdown: true } => {
            shutdown.notify_one();
            Response::Ok(ReplyPayload {
                text: "shutting down".into(),
                tool_calls: vec![],
            })
        }
        Request::Shutdown { shutdown: false } => Response::Ok(ReplyPayload {
            text: "noop".into(),
            tool_calls: vec![],
        }),
        Request::Reset { reset: true } => {
            session.conv.lock().await.reset();
            Response::Ok(ReplyPayload {
                text: "conversation reset".into(),
                tool_calls: vec![],
            })
        }
        Request::Reset { reset: false } => Response::Ok(ReplyPayload {
            text: "noop".into(),
            tool_calls: vec![],
        }),
        Request::Prompt { prompt } => {
            run_prompt(&session, &prompt, &mut reader, &mut writer).await
        }
    };

    write_json_line(&mut writer, &response).await?;
    writer.flush().await?;
    Ok(())
}

/// Drives one prompt: spawns the agent in a worker task with a
/// [`ChannelConfirmer`] installed, ferries any confirmation prompts onto
/// the socket while the agent is running, and resolves to the final
/// `Response` once the agent task completes.
async fn run_prompt<R, W>(
    session: &Arc<Session>,
    prompt: &str,
    reader: &mut BufReader<R>,
    writer: &mut W,
) -> Response
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let confirmer = Arc::new(ChannelConfirmer::new(tx));

    // Run the agent in its own task. Hold the conversation lock for the
    // whole turn so concurrent connections serialize naturally.
    let session_for_task = session.clone();
    let prompt_owned = prompt.to_string();
    let agent_task = tokio::spawn(async move {
        let mut conv = session_for_task.conv.lock().await;
        session_for_task
            .agent
            .turn(&mut conv, &prompt_owned, confirmer)
            .await
    });

    // Pump confirmation prompts until the agent task finishes. Once the
    // agent drops its `ChannelConfirmer`, the receiver yields `None`
    // and we fall through to await the final reply.
    let mut agent_task = agent_task;
    let result = loop {
        tokio::select! {
            biased;
            // Forward a confirm prompt to the client and read back the
            // human's y/N. Network errors here become tool errors via
            // the dropped oneshot.
            maybe_confirm = rx.recv() => match maybe_confirm {
                Some((prompt, reply_tx)) => {
                    let record = ConfirmRecord { confirm: prompt };
                    if let Err(e) = write_json_line(writer, &record).await {
                        // Connection died: just drop reply_tx and wait
                        // for the agent to error out cleanly.
                        eprintln!("confirm write failed: {e:#}");
                        drop(reply_tx);
                        continue;
                    }
                    if let Err(e) = writer.flush().await {
                        eprintln!("confirm flush failed: {e:#}");
                        drop(reply_tx);
                        continue;
                    }
                    let mut buf = String::new();
                    let answer = match reader.read_line(&mut buf).await {
                        Ok(0) => false,
                        Ok(_) => match serde_json::from_str::<ConfirmAnswer>(buf.trim()) {
                            Ok(a) => a.answer,
                            Err(e) => {
                                eprintln!("confirm answer parse: {e}");
                                false
                            }
                        },
                        Err(e) => {
                            eprintln!("confirm answer read: {e}");
                            false
                        }
                    };
                    let _ = reply_tx.send(answer);
                }
                None => {
                    // Sender was dropped — agent finished. Wait for it.
                    break agent_task.await;
                }
            },
            joined = &mut agent_task => break joined,
        }
    };

    match result {
        Ok(Ok(reply)) => Response::Ok(ReplyPayload {
            text: reply.text,
            tool_calls: reply.tool_calls,
        }),
        Ok(Err(e)) => Response::Err {
            error: format!("{e:#}"),
        },
        Err(join_err) => Response::Err {
            error: format!("agent task panicked: {join_err}"),
        },
    }
}

async fn write_json_line<W: AsyncWrite + Unpin, T: Serialize>(
    writer: &mut W,
    value: &T,
) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// client
// ---------------------------------------------------------------------------

/// Generic streaming client. Sends `req`, then yields each
/// `{"confirm":...}` line the daemon emits to `on_confirm` (which must
/// return the human's y/N) until the final `Reply` / `Error` arrives.
pub async fn send_request<F>(req: Request, mut on_confirm: F) -> Result<ReplyPayload>
where
    F: FnMut(String) -> Result<bool>,
{
    let path = socket_path();
    let stream = tokio::time::timeout(Duration::from_millis(200), UnixStream::connect(&path))
        .await
        .map_err(|_| anyhow::anyhow!("daemon connect timed out"))?
        .with_context(|| format!("connect {}", path.display()))?;
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut writer = write_half;

    write_json_line(&mut writer, &req).await?;
    writer.flush().await?;

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            anyhow::bail!("daemon closed connection without a reply");
        }
        let line = line.trim();
        // Try the typed responses in order. Confirms come first because
        // they're the most specific shape; if it doesn't have a
        // `confirm` field, fall through to the response shapes.
        if let Ok(rec) = serde_json::from_str::<ConfirmRecord>(line) {
            let answer = on_confirm(rec.confirm).unwrap_or(false);
            write_json_line(&mut writer, &ConfirmAnswer { answer }).await?;
            writer.flush().await?;
            continue;
        }
        let resp: Response = serde_json::from_str(line)
            .map_err(|e| anyhow::anyhow!("bad response json: {e} (got: {line:?})"))?;
        return match resp {
            Response::Ok(payload) => Ok(payload),
            Response::Err { error } => Err(anyhow::anyhow!("daemon: {error}")),
        };
    }
}

pub async fn try_send<F>(prompt: &str, on_confirm: F) -> Result<ReplyPayload>
where
    F: FnMut(String) -> Result<bool>,
{
    send_request(
        Request::Prompt {
            prompt: prompt.to_string(),
        },
        on_confirm,
    )
    .await
}

pub async fn try_reset() -> Result<ReplyPayload> {
    send_request(Request::Reset { reset: true }, |_| Ok(false)).await
}

async fn try_ping(path: &std::path::Path) -> Result<()> {
    let mut stream = tokio::time::timeout(Duration::from_millis(100), UnixStream::connect(path))
        .await
        .map_err(|_| anyhow::anyhow!("ping timeout"))??;
    // Send a noop shutdown request (false → noop). If the other side
    // responds, it's alive.
    let req = serde_json::to_string(&Request::Shutdown { shutdown: false })?;
    stream.write_all(req.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.shutdown().await?;
    let mut buf = String::new();
    BufReader::new(stream).read_line(&mut buf).await?;
    Ok(())
}

pub async fn status() -> Result<()> {
    let path = socket_path();
    if !path.exists() {
        println!("not running (no socket at {})", path.display());
        return Ok(());
    }
    match try_ping(&path).await {
        Ok(_) => println!("running ({})", path.display()),
        Err(e) => println!("stale socket at {} ({e})", path.display()),
    }
    Ok(())
}

pub async fn stop() -> Result<()> {
    let path = socket_path();
    let mut stream = UnixStream::connect(&path)
        .await
        .with_context(|| format!("connect {}", path.display()))?;
    let req = serde_json::to_string(&Request::Shutdown { shutdown: true })?;
    stream.write_all(req.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.shutdown().await?;
    let mut buf = String::new();
    let _ = BufReader::new(stream).read_line(&mut buf).await;
    println!("sent shutdown");
    Ok(())
}
