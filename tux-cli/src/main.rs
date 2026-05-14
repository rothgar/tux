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

use anyhow::{anyhow, Result};
#[cfg(feature = "llama")]
use anyhow::Context;
use clap::{Parser, Subcommand};
use directories::ProjectDirs;
use std::io::{IsTerminal, Read};
use std::path::PathBuf;
use std::sync::Arc;
use tux_core::backend::{Backend, BackendKind, MockBackend};
use tux_core::{Agent, SystemContext, ToolRegistry};

#[derive(Parser)]
#[command(name = "tux", version, about = "Local AI assistant for Linux")]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    /// Path to a GGUF model file. Defaults to
    /// `$XDG_DATA_HOME/tux/models/default.gguf` if it exists.
    #[arg(long, env = "TUX_MODEL", global = true)]
    model: Option<PathBuf>,

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
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

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
            })
            .await
        }
        Some(Cmd::Daemon { action: DaemonCmd::Status }) => return daemon::status().await,
        Some(Cmd::Daemon { action: DaemonCmd::Stop }) => return daemon::stop().await,
        Some(Cmd::Daemon { action: DaemonCmd::Serve }) => {
            let model_path = cli.model.or_else(default_model_path);
            let backend = build_backend(model_path.as_ref())?;
            let agent = Agent::new(backend, ToolRegistry::with_defaults(), SystemContext::detect());
            return daemon::serve(agent).await;
        }
        Some(Cmd::Info) => {
            // Info is cheap and shouldn't pay for a daemon round-trip.
            let model_path = cli.model.or_else(default_model_path);
            let backend = build_backend(model_path.as_ref())?;
            let agent = Agent::new(backend, ToolRegistry::with_defaults(), SystemContext::detect());
            print_info(&agent);
            return Ok(());
        }
        None => {}
    }

    // Default action: answer a prompt from args/stdin.
    let prompt = read_prompt(cli.prompt)?;

    // Try the daemon first; if it's not running, fall back to in-process.
    match daemon::try_send(&prompt).await {
        Ok(payload) => {
            for tc in &payload.tool_calls {
                eprintln!("· {} → {}", tc.tool, tc.summary);
            }
            println!("{}", payload.text);
            return Ok(());
        }
        Err(e) => {
            tracing::debug!("no daemon ({e}); loading model in-process");
        }
    }

    let model_path = cli.model.or_else(default_model_path);
    let backend = build_backend(model_path.as_ref())?;
    let agent = Agent::new(backend, ToolRegistry::with_defaults(), SystemContext::detect());
    let reply = agent.handle(&prompt).await?;
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

fn build_backend(model: Option<&PathBuf>) -> Result<Arc<dyn Backend>> {
    match model {
        #[cfg(feature = "llama")]
        Some(path) => tux_core::backend::llama::from_kind(&BackendKind::LlamaCpp {
            model_path: path.clone(),
            mmproj_path: default_mmproj_path(),
        })
        .with_context(|| format!("failed to load model {}", path.display())),
        #[cfg(not(feature = "llama"))]
        Some(_) => {
            tracing::warn!(
                "--model passed but tux was built without the `llama` feature; using mock"
            );
            let _ = BackendKind::Mock;
            Ok(Arc::new(MockBackend))
        }
        None => {
            tracing::warn!(
                "no model configured; using mock backend. \
                 run `tux init` to download one."
            );
            Ok(Arc::new(MockBackend))
        }
    }
}
