//! Eval suite — runs *real* prompts through the actual llama.cpp backend
//! and checks that the model behaves sensibly for common Linux requests.
//!
//! Skipped automatically unless:
//!   - the `llama` feature is enabled  (`cargo test --features llama`)
//!   - `TUX_MODEL` points at a readable GGUF file
//!
//! Recommended:
//!   nix develop
//!   export TUX_MODEL=$XDG_DATA_HOME/tux/models/default.gguf
//!   cargo test --features llama --release --test eval -- --nocapture --test-threads=1
//!
//! `--release` makes inference roughly 5–10× faster; `--test-threads=1`
//! prevents two tests from hammering the same llama context concurrently.

#![cfg(feature = "llama")]

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tux_core::backend::{llama::from_kind, Backend, BackendKind};
use tux_core::{Agent, SystemContext, ToolRegistry};

fn model_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("TUX_MODEL")?);
    p.exists().then_some(p)
}

fn shared_backend() -> Option<Arc<dyn Backend>> {
    static BACKEND: OnceLock<Result<Arc<dyn Backend>, String>> = OnceLock::new();
    let path = model_path()?;
    let result = BACKEND.get_or_init(|| {
        from_kind(&BackendKind::LlamaCpp {
            model_path: path,
            mmproj_path: None,
        })
        .map_err(|e| format!("{e:#}"))
    });
    match result {
        Ok(b) => Some(b.clone()),
        Err(e) => panic!("failed to load TUX_MODEL: {e}"),
    }
}

fn agent() -> Option<Agent> {
    let backend = shared_backend()?;
    Some(Agent::new(
        backend,
        ToolRegistry::with_defaults(),
        SystemContext::detect(),
    ))
}

/// Asserts that `text` (case-insensitive) contains *any* of the candidate
/// substrings. The model has freedom in wording — we only fail if it's
/// clearly off-topic.
fn assert_contains_any(text: &str, candidates: &[&str], context: &str) {
    let lower = text.to_lowercase();
    let hit = candidates.iter().any(|c| lower.contains(&c.to_lowercase()));
    assert!(
        hit,
        "{context}\nexpected one of {candidates:?}\ngot: {text}"
    );
}

macro_rules! eval {
    ($name:ident, $body:expr) => {
        #[tokio::test(flavor = "current_thread")]
        async fn $name() {
            let Some(agent) = agent() else {
                eprintln!(
                    "[skip] {}: set TUX_MODEL to a GGUF path to run",
                    stringify!($name)
                );
                return;
            };
            let body: fn(Agent) -> _ = $body;
            body(agent).await
        }
    };
}

// ---------------------------------------------------------------------------
// common-request behavior checks
// ---------------------------------------------------------------------------

eval!(answers_install_request_with_a_command, |agent: Agent| async move {
    let r = agent.handle("how do I install neovim?").await.unwrap();
    eprintln!("→ {}", r.text);
    // Should mention *some* package-management verb. We don't pin a distro:
    // the SystemContext on the test host could be anything.
    assert_contains_any(
        &r.text,
        &["install", "pacman", "apt", "dnf", "nix", "zypper", "package"],
        "install request should mention installation or a package manager",
    );
});

eval!(answers_distro_question_using_host_context, |agent: Agent| async move {
    // Only meaningful if SystemContext detected a distro.
    let distro = agent.context().distro_pretty_name.clone();
    let Some(distro) = distro else {
        eprintln!("[skip] no distro detected on host");
        return;
    };
    let r = agent.handle("what linux distribution am I running?").await.unwrap();
    eprintln!("→ {}", r.text);
    // Match on the first word of the pretty name (e.g. "NixOS" from "NixOS 25.11 (...)").
    let key = distro.split_whitespace().next().unwrap_or(&distro);
    assert_contains_any(
        &r.text,
        &[key],
        "distro question should mention the actual distro from host facts",
    );
});

eval!(invokes_screenshot_tool_when_asked, |agent: Agent| async move {
    let r = agent
        .handle("take a screenshot of my screen so we can look at it together")
        .await;
    // The screenshot tool may fail to actually capture in CI/headless, but
    // we *only* care that the model decided to call it. Accept either:
    //   - a successful tool_calls entry, or
    //   - an error mentioning the screenshot tool / grim / scrot.
    match r {
        Ok(reply) => {
            eprintln!("→ tool_calls: {:?}", reply.tool_calls.iter().map(|t| &t.tool).collect::<Vec<_>>());
            eprintln!("→ {}", reply.text);
            assert!(
                reply.tool_calls.iter().any(|t| t.tool == "screenshot"),
                "expected the model to invoke the screenshot tool; got reply: {}",
                reply.text
            );
        }
        Err(e) => {
            let msg = format!("{e:#}");
            eprintln!("→ err: {msg}");
            assert!(
                msg.contains("screenshot") || msg.contains("grim") || msg.contains("scrot"),
                "expected screenshot-related error, got: {msg}"
            );
        }
    }
});

eval!(stays_on_topic_for_a_diagnostic_question, |agent: Agent| async move {
    let r = agent
        .handle("why might fonts look blurry on an external monitor under wayland?")
        .await
        .unwrap();
    eprintln!("→ {}", r.text);
    assert_contains_any(
        &r.text,
        &[
            "scaling",
            "scale",
            "dpi",
            "fractional",
            "hinting",
            "subpixel",
            "wayland",
            "xwayland",
            "display",
        ],
        "diagnostic question should touch on a relevant rendering / scaling concept",
    );
});

eval!(does_not_leak_chain_of_thought, |agent: Agent| async move {
    // We pre-fill <think></think> in the chat template; verify it actually
    // suppresses reasoning leakage on Qwen3 / Qwen3.5.
    let r = agent.handle("what time is it on the moon?").await.unwrap();
    eprintln!("→ {}", r.text);
    let lower = r.text.to_lowercase();
    assert!(
        !lower.contains("<think>") && !lower.contains("</think>"),
        "raw think tags leaked into output: {}",
        r.text
    );
});
