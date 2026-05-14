//! Minimal GTK4 shell: a single window with a transcript view and an entry.
//! Submitting an entry runs the agent on a tokio runtime and appends the
//! reply (plus any tool-call trace) to the transcript.

use anyhow::Result;
use directories::ProjectDirs;
use gtk4::glib;
use gtk4::prelude::*;
use gtk4::{
    Application, ApplicationWindow, Box as GtkBox, Entry, Orientation, ScrolledWindow, TextView,
    WrapMode,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use tux_core::agent::AgentReply;
use tux_core::backend::{Backend, BackendKind, MockBackend};
use tux_core::{Agent, SystemContext, ToolRegistry};

const APP_ID: &str = "dev.tux.Tux";

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let app = Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_ui);
    app.run();
    Ok(())
}

fn default_model_path() -> Option<PathBuf> {
    let dirs = ProjectDirs::from("dev", "tux", "tux")?;
    let p = dirs.data_dir().join("models").join("default.gguf");
    p.exists().then_some(p)
}

fn build_backend() -> Arc<dyn Backend> {
    let model = std::env::var_os("TUX_MODEL")
        .map(PathBuf::from)
        .or_else(default_model_path);

    #[cfg(feature = "llama")]
    if let Some(path) = model {
        let mmproj = ProjectDirs::from("dev", "tux", "tux")
            .map(|d| d.data_dir().join("models").join("default.mmproj"))
            .filter(|p| p.exists());
        match tux_core::backend::llama::from_kind(&BackendKind::LlamaCpp {
            model_path: path,
            mmproj_path: mmproj,
        }) {
            Ok(b) => return b,
            Err(e) => tracing::warn!("falling back to mock: {e:#}"),
        }
    }
    #[cfg(not(feature = "llama"))]
    let _ = (model, BackendKind::Mock);

    Arc::new(MockBackend)
}

fn build_ui(app: &Application) {
    let runtime = Rc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime"),
    );

    let agent = Arc::new(Agent::new(
        build_backend(),
        ToolRegistry::with_defaults(),
        SystemContext::detect(),
    ));

    let window = ApplicationWindow::builder()
        .application(app)
        .title("tux")
        .default_width(640)
        .default_height(480)
        .build();

    let vbox = GtkBox::new(Orientation::Vertical, 6);
    vbox.set_margin_top(8);
    vbox.set_margin_bottom(8);
    vbox.set_margin_start(8);
    vbox.set_margin_end(8);

    let transcript = TextView::builder()
        .editable(false)
        .wrap_mode(WrapMode::WordChar)
        .monospace(true)
        .build();
    let buffer = transcript.buffer();
    let scroll = ScrolledWindow::builder()
        .vexpand(true)
        .child(&transcript)
        .build();
    vbox.append(&scroll);

    let entry = Entry::builder()
        .placeholder_text("ask tux anything…")
        .build();
    vbox.append(&entry);

    window.set_child(Some(&vbox));

    let buffer = Rc::new(buffer);
    let busy = Rc::new(RefCell::new(false));

    {
        let buffer = buffer.clone();
        let agent = agent.clone();
        let runtime = runtime.clone();
        let busy = busy.clone();
        entry.connect_activate(move |entry| {
            if *busy.borrow() {
                return;
            }
            let text = entry.text().to_string();
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return;
            }
            append_line(&buffer, &format!("» {trimmed}"));
            entry.set_text("");

            let (tx, rx) = std::sync::mpsc::channel::<Result<AgentReply>>();
            let agent_for_task = agent.clone();
            let prompt = trimmed.to_string();
            *busy.borrow_mut() = true;
            runtime.spawn(async move {
                let reply = agent_for_task.handle(&prompt).await;
                let _ = tx.send(reply);
            });

            let buffer = buffer.clone();
            let busy = busy.clone();
            glib::idle_add_local(move || match rx.try_recv() {
                Ok(Ok(reply)) => {
                    for tc in &reply.tool_calls {
                        append_line(&buffer, &format!("· {} → {}", tc.tool, tc.summary));
                    }
                    append_line(&buffer, &reply.text);
                    *busy.borrow_mut() = false;
                    glib::ControlFlow::Break
                }
                Ok(Err(e)) => {
                    append_line(&buffer, &format!("error: {e:#}"));
                    *busy.borrow_mut() = false;
                    glib::ControlFlow::Break
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    *busy.borrow_mut() = false;
                    glib::ControlFlow::Break
                }
            });
        });
    }

    window.present();
}

fn append_line(buffer: &gtk4::TextBuffer, line: &str) {
    let mut end = buffer.end_iter();
    let prefix = if buffer.char_count() == 0 { "" } else { "\n" };
    buffer.insert(&mut end, &format!("{prefix}{line}"));
}
