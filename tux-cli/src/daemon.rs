//! `tux daemon` — run a long-lived process that holds the model in memory
//! so each `tux <prompt>` invocation skips the 2–5 s cold load.
//!
//! Wire protocol: newline-delimited JSON over a unix socket at
//! `$XDG_RUNTIME_DIR/tux.sock`. One request per connection.
//!
//! Requests:
//!   {"prompt": "..."}      → run the agent, return its reply
//!   {"shutdown": true}     → graceful exit
//!
//! Responses:
//!   {"text": "...", "tool_calls": [{tool, summary, data}, ...]}
//!   {"error": "..."}
//!
//! The CLI auto-detects the socket; if it's not there or doesn't answer
//! within a short timeout, the CLI falls back to an in-process model load.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Notify;
use tux_core::tools::ToolResult;
use tux_core::Agent;

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Request {
    Prompt { prompt: String },
    Shutdown { shutdown: bool },
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

    let agent = Arc::new(agent);
    let shutdown = Arc::new(Notify::new());

    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                eprintln!("tuxd shutting down");
                break;
            }
            accept = listener.accept() => {
                let (stream, _) = accept.context("accept")?;
                let agent = agent.clone();
                let shutdown = shutdown.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle(stream, agent, shutdown).await {
                        eprintln!("connection error: {e:#}");
                    }
                });
            }
        }
    }

    let _ = std::fs::remove_file(&path);
    Ok(())
}

async fn handle(stream: UnixStream, agent: Arc<Agent>, shutdown: Arc<Notify>) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half).lines();

    let Some(line) = reader.next_line().await? else {
        return Ok(());
    };

    let req: Request = serde_json::from_str(&line)
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
        Request::Prompt { prompt } => match agent.handle(&prompt).await {
            Ok(reply) => Response::Ok(ReplyPayload {
                text: reply.text,
                tool_calls: reply.tool_calls,
            }),
            Err(e) => Response::Err {
                error: format!("{e:#}"),
            },
        },
    };

    let mut bytes = serde_json::to_vec(&response)?;
    bytes.push(b'\n');
    write_half.write_all(&bytes).await?;
    write_half.flush().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// client
// ---------------------------------------------------------------------------

pub async fn try_send(prompt: &str) -> Result<ReplyPayload> {
    let path = socket_path();
    let mut stream = tokio::time::timeout(Duration::from_millis(200), UnixStream::connect(&path))
        .await
        .map_err(|_| anyhow::anyhow!("daemon connect timed out"))?
        .with_context(|| format!("connect {}", path.display()))?;

    let req = Request::Prompt {
        prompt: prompt.to_string(),
    };
    let mut bytes = serde_json::to_vec(&req)?;
    bytes.push(b'\n');
    stream.write_all(&bytes).await?;
    stream.flush().await?;
    stream.shutdown().await?;

    let mut buf = String::new();
    BufReader::new(stream).read_line(&mut buf).await?;
    let resp: Response = serde_json::from_str(buf.trim())
        .map_err(|e| anyhow::anyhow!("bad response json: {e} (got: {buf:?})"))?;

    match resp {
        Response::Ok(payload) => Ok(payload),
        Response::Err { error } => Err(anyhow::anyhow!("daemon: {error}")),
    }
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
