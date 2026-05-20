//! tux CLI — pipe-friendly, traditional Unix shape.
//!
//!   tux install neovim                       # args become the prompt
//!   echo "install neovim" | tux              # reads from stdin
//!   cat error.log | tux "what's wrong here?" # args + stdin combined
//!   tux info                                 # show host facts and tools
//!   tux init                                 # download & set up a model
//!   tux daemon serve                         # run the model-resident daemon
//!   tux daemon status | stop                 # control the daemon
//!
//! When a daemon is running, prompts go to it over a unix socket so the
//! model load happens once. Otherwise the CLI loads the model in-process.

mod daemon;
mod init;
mod spinner;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use directories::ProjectDirs;
use std::io::{IsTerminal, Read};
use std::path::PathBuf;
use std::sync::Arc;
use tux_core::backend::{Backend, MockBackend, OpenAICompatBackend};
#[cfg(feature = "llama")]
use tux_core::backend::BackendKind;
use tux_core::config::{BackendType, TuxConfig};
use tux_core::{Agent, SystemContext, ToolRegistry};

#[derive(Parser)]
#[command(name = "tux", version, about = "Local AI assistant for Linux")]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    /// Path to a GGUF model file. Overrides config.toml.
    #[arg(long, env = "TUX_MODEL", global = true)]
    model: Option<PathBuf>,

    /// Base URL of an OpenAI-compatible API (e.g. http://host:11434).
    /// Takes priority over --model and config.toml.
    #[arg(long, env = "TUX_REMOTE_URL", global = true)]
    remote_url: Option<String>,

    /// Model name to send to the remote API. Defaults to "default".
    #[arg(long, env = "TUX_REMOTE_MODEL", global = true)]
    remote_model: Option<String>,

    #[command(subcommand)]
    cmd: Option<Cmd>,

    /// Prompt words. Joined with spaces. If omitted and stdin is a pipe,
    /// the prompt is read from stdin.
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Auto-detect hardware, download a suitable model, install the systemd
    /// user daemon, and write the distro knowledge cache.
    ///
    /// Pass --remote-url to configure a remote backend instead of
    /// downloading a GGUF model.
    Init {
        /// Override the auto-pick (one of: qwen3.5-4b-q4, qwen3.5-2b-q4, qwen3.5-0.8b-q4).
        #[arg(long)]
        model: Option<String>,

        /// Also download the default vision model + mmproj so the model can
        /// look at screenshots and pasted images.
        #[arg(long)]
        with_vision: bool,

        /// Skip writing + enabling the systemd user unit for tuxd.
        #[arg(long)]
        no_daemon: bool,
    },
    /// Manage the resident daemon that holds the loaded model.
    Daemon {
        #[command(subcommand)]
        action: DaemonCmd,
    },
    /// Clear the daemon's conversation history (start a fresh session).
    Reset,
    /// Print detected system context and registered tools.
    Info,
}

#[derive(Subcommand)]
enum DaemonCmd {
    /// Run the daemon in the foreground until interrupted.
    Serve,
    /// Print whether a daemon is reachable.
    Status,
    /// Ask a running daemon to exit.
    Stop,
}

#[tokio::main]
async fn main() -> Result<()> {
    // `TUX_DEBUG=1` surfaces llama.cpp / ggml chatter and our own debug
    // traces. Otherwise we keep stderr quiet and only show warnings.
    let tux_debug = std::env::var_os("TUX_DEBUG").is_some_and(|v| !v.is_empty());
    let default_filter = if tux_debug { "debug" } else { "warn" };
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_filter)),
        )
        .init();

    let cli = Cli::parse();

    let remote_url = cli.remote_url.as_deref();
    let remote_model = cli.remote_model.as_deref();

    match cli.cmd {
        Some(Cmd::Init {
            model,
            with_vision,
            no_daemon,
        }) => {
            return init::run(init::InitOptions {
                model,
                install_daemon: !no_daemon,
                with_vision,
                remote_url: cli.remote_url,
                remote_model: cli.remote_model,
            })
            .await
        }
        Some(Cmd::Daemon { action: DaemonCmd::Status }) => return daemon::status().await,
        Some(Cmd::Daemon { action: DaemonCmd::Stop }) => return daemon::stop().await,
        Some(Cmd::Daemon { action: DaemonCmd::Serve }) => {
            let backend = build_backend(cli.model.as_ref(), remote_url, remote_model)?;
            let agent = Agent::new(backend, ToolRegistry::with_defaults(), SystemContext::detect());
            return daemon::serve(agent).await;
        }
        Some(Cmd::Reset) => {
            match daemon::try_reset().await {
                Ok(payload) => {
                    println!("{}", payload.text);
                    return Ok(());
                }
                Err(e) => {
                    return Err(anyhow!(
                        "could not reach daemon to reset: {e}\n\
                         (no daemon running? in-process sessions are reset every invocation.)"
                    ))
                }
            }
        }
        Some(Cmd::Info) => {
            let backend = build_backend(cli.model.as_ref(), remote_url, remote_model)?;
            let agent = Agent::new(backend, ToolRegistry::with_defaults(), SystemContext::detect());
            print_info(&agent);
            return Ok(());
        }
        None => {}
    }

    // Default action: answer a prompt from args/stdin.
    let prompt = read_prompt(cli.prompt)?;

    // Try the daemon first; if it's not running, fall back to in-process.
    // The daemon may interleave confirmation prompts (`{"confirm":...}`)
    // before the final reply — `ask_via_tty` ferries those to /dev/tty
    // and reads the human's answer.
    let spin = spinner::Spinner::start("thinking…");
    let confirm_spin = std::cell::RefCell::new(Some(spin));
    match daemon::try_send(&prompt, |prompt_text| {
        // Pause the spinner so it doesn't trample the y/N prompt.
        if let Some(s) = confirm_spin.borrow_mut().take() {
            // Best-effort sync stop on the drop path; restart is too
            // costly for a brief tool confirmation.
            drop(s);
        }
        ask_via_tty(&prompt_text)
    })
    .await
    {
        Ok(payload) => {
            if let Some(s) = confirm_spin.borrow_mut().take() {
                s.stop().await;
            }
            for tc in &payload.tool_calls {
                eprintln!("· {} → {}", tc.tool, tc.summary);
            }
            println!("{}", payload.text);
            return Ok(());
        }
        Err(e) => {
            if let Some(s) = confirm_spin.borrow_mut().take() {
                s.stop().await;
            }
            tracing::debug!("no daemon ({e}); loading model in-process");
        }
    }

    let load_spin = spinner::Spinner::start("loading model…");
    let backend = build_backend(cli.model.as_ref(), remote_url, remote_model)?;
    let agent = Agent::new(backend, ToolRegistry::with_defaults(), SystemContext::detect());
    load_spin.stop().await;

    let spin = spinner::Spinner::start("thinking…");
    let reply = agent.handle(&prompt).await?;
    spin.stop().await;

    for tc in &reply.tool_calls {
        eprintln!("· {} → {}", tc.tool, tc.summary);
    }
    println!("{}", reply.text);
    Ok(())
}

/// Build the prompt from positional args and/or stdin.
fn read_prompt(args: Vec<String>) -> Result<String> {
    let from_args = args.join(" ").trim().to_string();
    let from_stdin = if !std::io::stdin().is_terminal() {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf.trim().to_string()
    } else {
        String::new()
    };

    match (from_args.is_empty(), from_stdin.is_empty()) {
        (true, true) => Err(anyhow!(
            "no prompt — pass words as args or pipe text via stdin\n\
             try: tux install neovim\n\
             or:  echo 'why is text blurry?' | tux"
        )),
        (false, true) => Ok(from_args),
        (true, false) => Ok(from_stdin),
        (false, false) => Ok(format!("{from_args}\n\n{from_stdin}")),
    }
}

/// Forward a daemon-originated confirmation question to the user via
/// `/dev/tty`, returning their y/N. Mirrors the in-process
/// `TtyConfirmer` so the UX is identical whether the agent ran here or
/// inside `tuxd`. Declines if no controlling terminal exists.
fn ask_via_tty(prompt: &str) -> Result<bool> {
    use std::fs::OpenOptions;
    use std::io::{BufRead, BufReader, Write};
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

fn print_info(agent: &Agent) {
    println!("backend: {}", agent.backend_name());
    println!("\n{}", agent.context().as_prompt_block());
    println!("tools:");
    for (name, desc) in agent.tools().list() {
        println!("  {:<14} {}", name, desc);
    }
}

fn default_model_path() -> Option<PathBuf> {
    let dirs = ProjectDirs::from("dev", "tux", "tux")?;
    let p = dirs.data_dir().join("models").join("default.gguf");
    p.exists().then_some(p)
}

/// Sibling vision projector: `<models>/default.mmproj` if it exists.
#[cfg_attr(not(feature = "llama"), allow(dead_code))]
fn default_mmproj_path() -> Option<PathBuf> {
    let dirs = ProjectDirs::from("dev", "tux", "tux")?;
    let p = dirs.data_dir().join("models").join("default.mmproj");
    p.exists().then_some(p)
}

/// Construct a backend using the priority chain:
///   1. `--remote-url` flag → OpenAI-compat
///   2. `--model` flag      → local GGUF (inference params from config)
///   3. config.toml remote  → OpenAI-compat
///   4. config.toml local   → local GGUF
///   5. default.gguf exists → local GGUF
///   6. mock (warns)
fn build_backend(
    model: Option<&PathBuf>,
    remote_url: Option<&str>,
    remote_model: Option<&str>,
) -> Result<Arc<dyn Backend>> {
    // 1. Remote flag
    if let Some(url) = remote_url {
        let name = remote_model.unwrap_or("default").to_string();
        return Ok(Arc::new(OpenAICompatBackend::new(url.to_string(), name, None)));
    }

    let cfg = TuxConfig::load();

    // 2. Explicit --model flag
    if let Some(path) = model {
        return build_local(path, default_mmproj_path().as_ref(), &cfg)
            .with_context(|| format!("failed to load model {}", path.display()));
    }

    // 3 & 4. Config file
    match cfg.backend.kind {
        BackendType::Remote => {
            let url = cfg
                .backend
                .url
                .ok_or_else(|| anyhow!("config: remote backend requires a url"))?;
            let name = cfg.backend.model.unwrap_or_else(|| "default".to_string());
            return Ok(Arc::new(OpenAICompatBackend::new(url, name, cfg.backend.api_key)));
        }
        BackendType::Local => {
            if let Some(path) = cfg.backend.model_path.as_ref().filter(|p| p.exists()) {
                return build_local(path, cfg.backend.mmproj_path.as_ref(), &cfg)
                    .with_context(|| format!("failed to load model {}", path.display()));
            }
        }
    }

    // 5. default.gguf fallback
    if let Some(path) = default_model_path() {
        return build_local(&path, default_mmproj_path().as_ref(), &cfg)
            .with_context(|| format!("failed to load model {}", path.display()));
    }

    // 6. Mock
    tracing::warn!("no model configured; using mock backend. run `tux init` to set one up.");
    Ok(Arc::new(MockBackend))
}

fn build_local(
    model_path: &PathBuf,
    mmproj_path: Option<&PathBuf>,
    cfg: &TuxConfig,
) -> Result<Arc<dyn Backend>> {
    #[cfg(feature = "llama")]
    {
        let inf = &cfg.inference;
        return tux_core::backend::from_kind(BackendKind::LlamaCpp {
            model_path: model_path.clone(),
            mmproj_path: mmproj_path.cloned(),
            n_threads: inf.n_threads,
            ctx_size: inf.ctx_size,
            n_gpu_layers: inf.n_gpu_layers,
            batch_size: inf.batch_size,
        });
    }
    #[cfg(not(feature = "llama"))]
    {
        let _ = (model_path, mmproj_path, cfg);
        tracing::warn!("--model passed but tux was built without the `llama` feature; using mock");
        Ok(Arc::new(MockBackend))
    }
}
