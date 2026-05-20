//! Minimal stderr spinner shown while the model is thinking.
//!
//! - Active only when stderr is a TTY — silent under pipes / journald so
//!   `tux ... | tee` and similar stay byte-clean.
//! - Single-line, redrawn in place with `\r`; cleared on drop with the
//!   ANSI "erase to end of line" sequence so no junk is left behind.
//! - Self-contained: no extra crate dependency.

use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::task::JoinHandle;

const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    enabled: bool,
}

impl Spinner {
    /// Start a spinner with the given label. If stderr is not a TTY this
    /// is a cheap no-op so callers don't need to gate it themselves.
    pub fn start(label: &str) -> Self {
        let enabled = std::io::stderr().is_terminal();
        if !enabled {
            return Self {
                stop: Arc::new(AtomicBool::new(true)),
                handle: None,
                enabled: false,
            };
        }

        let stop = Arc::new(AtomicBool::new(false));
        let stop_task = stop.clone();
        let label = label.to_string();

        let handle = tokio::spawn(async move {
            let mut i = 0usize;
            // Hide the terminal cursor while we animate; restore on stop.
            let _ = write!(std::io::stderr(), "\x1b[?25l");
            while !stop_task.load(Ordering::Relaxed) {
                let _ = write!(
                    std::io::stderr(),
                    "\r{} {}",
                    FRAMES[i % FRAMES.len()],
                    label
                );
                let _ = std::io::stderr().flush();
                i += 1;
                tokio::time::sleep(Duration::from_millis(90)).await;
            }
            // Erase the spinner line and restore the cursor.
            let _ = write!(std::io::stderr(), "\r\x1b[2K\x1b[?25h");
            let _ = std::io::stderr().flush();
        });

        Self {
            stop,
            handle: Some(handle),
            enabled: true,
        }
    }

    /// Stop the spinner and wait for the worker task to clean up.
    pub async fn stop(mut self) {
        if !self.enabled {
            return;
        }
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.await;
        }
    }
}

impl Drop for Spinner {
    /// Safety net for the panic / early-return path: tell the task to
    /// exit and try to clear the line synchronously. The async `stop`
    /// above is preferred when the caller can await it.
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        self.stop.store(true, Ordering::Relaxed);
        let _ = write!(std::io::stderr(), "\r\x1b[2K\x1b[?25h");
        let _ = std::io::stderr().flush();
    }
}
