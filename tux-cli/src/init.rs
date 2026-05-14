//! `tux init` — detect hardware, pick a model from the registry, download
//! it, validate it loads, and symlink it as the default. Non-interactive
//! by default; pass `--model <id>` to override the auto-pick.

use anyhow::{Context, Result};
use directories::ProjectDirs;
use futures_util::StreamExt;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};
use tux_core::models::{self, ModelEntry, ModelKind};
use tux_core::{knowledge, SystemContext};

/// Best-effort total system RAM in MiB. Reads /proc/meminfo; returns 0 on
/// platforms without it (forces fallback to the smallest model).
fn total_ram_mib() -> u32 {
    let Ok(content) = fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            if let Some(kib) = rest.split_whitespace().next().and_then(|s| s.parse::<u64>().ok())
            {
                return (kib / 1024) as u32;
            }
        }
    }
    0
}

fn cpu_cores() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

fn models_dir() -> Result<PathBuf> {
    let dirs = ProjectDirs::from("dev", "tux", "tux")
        .ok_or_else(|| anyhow::anyhow!("could not resolve project dirs"))?;
    let dir = dirs.data_dir().join("models");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub struct InitOptions {
    pub model: Option<String>,
    pub install_daemon: bool,
    pub with_vision: bool,
}

pub async fn run(opts: InitOptions) -> Result<()> {
    let ram = total_ram_mib();
    let cores = cpu_cores();
    eprintln!("detected: {cores} cores, {} MiB RAM", ram);

    let entry = match opts.model.as_deref() {
        Some(id) => models::lookup(id)
            .ok_or_else(|| anyhow::anyhow!("unknown model id: {id}"))?,
        None => models::pick_for_host(ram, cores),
    };

    eprintln!("selected: {} (~{} MiB)", entry.name, entry.size_mib);

    let dir = models_dir()?;
    let dest = dir.join(format!("{}.gguf", entry.id));
    let default_link = dir.join("default.gguf");

    if dest.exists() {
        let len_mib = fs::metadata(&dest)?.len() / 1024 / 1024;
        eprintln!("already downloaded: {} ({} MiB)", dest.display(), len_mib);
    } else {
        download(entry, &dest).await?;
    }

    update_default_symlink(&dest, &default_link)?;

    #[cfg(feature = "llama")]
    {
        eprintln!("validating model loads…");
        let _ = tux_core::backend::llama::from_kind(&tux_core::backend::BackendKind::LlamaCpp {
            model_path: dest.clone(),
            mmproj_path: None,
        })
        .with_context(|| format!("model failed to load: {}", dest.display()))?;
        eprintln!("ok");
    }
    #[cfg(not(feature = "llama"))]
    {
        eprintln!(
            "skipping load validation (built without --features llama); \
             rerun in the nix dev shell to verify"
        );
    }

    if opts.with_vision {
        let v = models::lookup(models::DEFAULT_VISION_MODEL)
            .ok_or_else(|| anyhow::anyhow!("no default vision model in registry"))?;
        download_vision(v).await?;
    }

    persist_system_knowledge()?;

    if opts.install_daemon {
        install_daemon_unit()?;
    } else {
        eprintln!("\nskipping daemon install (--no-daemon)");
    }

    eprintln!("\ndone. default model: {}", default_link.display());
    eprintln!("try: tux \"how do I check disk usage?\"");
    Ok(())
}

/// Download a vision model + its mmproj alongside, and symlink them as
/// `default.mmproj` (sibling of `default.gguf`) so the runtime picks them
/// up automatically.
async fn download_vision(entry: &ModelEntry) -> Result<()> {
    let ModelKind::Vision { mmproj_url, .. } = entry.kind else {
        anyhow::bail!("registry entry `{}` is not a vision model", entry.id);
    };
    let dir = models_dir()?;

    let model_dest = dir.join(format!("{}.gguf", entry.id));
    if !model_dest.exists() {
        download(entry, &model_dest).await?;
    } else {
        eprintln!("vision model already downloaded: {}", model_dest.display());
    }

    let mmproj_dest = dir.join(format!("{}.mmproj", entry.id));
    if !mmproj_dest.exists() {
        let temp = ModelEntry {
            id: "vision-mmproj",
            name: "mmproj projector",
            url: mmproj_url,
            size_mib: 0,
            min_ram_mib: 0,
            quality: 0,
            kind: ModelKind::Text, // doesn't matter for the download path
        };
        download(&temp, &mmproj_dest).await?;
    } else {
        eprintln!("mmproj already downloaded: {}", mmproj_dest.display());
    }

    update_default_symlink(&mmproj_dest, &dir.join("default.mmproj"))?;
    eprintln!("vision ready — model can now inspect images via the screenshot tool");
    Ok(())
}

/// Detect the host's distro knowledge and write it to system.json so the
/// user can inspect / edit it. Loaded back at runtime by `SystemContext`.
fn persist_system_knowledge() -> Result<()> {
    let ctx = SystemContext::detect();
    match (&ctx.distro_id, &ctx.knowledge) {
        (Some(id), Some(k)) => {
            let path = knowledge::save(k)?;
            eprintln!("wrote distro knowledge for `{id}` → {}", path.display());
        }
        (Some(id), None) => {
            eprintln!(
                "no curated knowledge for distro `{id}` — model will rely on general Linux knowledge.\n\
                 add an entry to tux-core/src/knowledge.rs (or write system.json by hand) to teach it."
            );
        }
        _ => eprintln!("could not detect distro from /etc/os-release"),
    }
    Ok(())
}

/// Install (and start) a systemd user unit for `tuxd`. Skips gracefully on
/// hosts without systemd-user.
fn install_daemon_unit() -> Result<()> {
    if Command::new("systemctl")
        .args(["--user", "--version"])
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!(
            "\nsystemctl --user not available; skipping daemon auto-install.\n\
             to run the daemon manually: `tux daemon serve &`"
        );
        return Ok(());
    }

    let bin = std::env::current_exe()
        .with_context(|| "could not resolve current executable path")?;

    let unit_dir = ProjectDirs::from("", "", "systemd")
        .map(|d| d.config_dir().to_path_buf())
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".config/systemd/user")
        });
    // ProjectDirs gives us config_dir for "systemd" → ~/.config/systemd, but
    // the user-units folder is the `user/` subdir of that:
    let unit_dir = if unit_dir.ends_with("user") {
        unit_dir
    } else {
        unit_dir.join("user")
    };
    fs::create_dir_all(&unit_dir)?;

    let unit_path = unit_dir.join("tuxd.service");
    let unit = format!(
        "[Unit]
Description=tux local AI daemon
After=default.target

[Service]
ExecStart={bin}\u{0020}daemon serve
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
",
        bin = bin.display()
    );
    fs::write(&unit_path, unit)
        .with_context(|| format!("write {}", unit_path.display()))?;
    eprintln!("\ninstalled systemd user unit: {}", unit_path.display());

    let reload = Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .status();
    if !matches!(reload, Ok(s) if s.success()) {
        eprintln!("warning: `systemctl --user daemon-reload` failed; skipping enable");
        return Ok(());
    }

    let enable = Command::new("systemctl")
        .args(["--user", "enable", "--now", "tuxd.service"])
        .status();
    match enable {
        Ok(s) if s.success() => eprintln!("enabled and started tuxd.service"),
        Ok(s) => eprintln!("warning: `systemctl --user enable --now tuxd.service` exited {s}"),
        Err(e) => eprintln!("warning: failed to enable tuxd.service: {e}"),
    }
    Ok(())
}

async fn download(entry: &ModelEntry, dest: &std::path::Path) -> Result<()> {
    let partial = dest.with_extension("gguf.partial");
    eprintln!("downloading {} → {}", entry.url, partial.display());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60 * 60))
        .build()?;
    let resp = client
        .get(entry.url)
        .send()
        .await
        .with_context(|| format!("GET {}", entry.url))?
        .error_for_status()?;

    let total = resp.content_length();
    let mut file = fs::File::create(&partial)
        .with_context(|| format!("create {}", partial.display()))?;

    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let started = Instant::now();
    let mut last_print = Instant::now();
    let mut stderr = std::io::stderr();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| "download stream error")?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;

        if last_print.elapsed() >= Duration::from_millis(500) {
            print_progress(&mut stderr, downloaded, total, started)?;
            last_print = Instant::now();
        }
    }
    print_progress(&mut stderr, downloaded, total, started)?;
    eprintln!();

    file.sync_all()?;
    drop(file);

    fs::rename(&partial, dest)
        .with_context(|| format!("rename {} → {}", partial.display(), dest.display()))?;
    Ok(())
}

fn print_progress(
    out: &mut impl Write,
    bytes: u64,
    total: Option<u64>,
    started: Instant,
) -> std::io::Result<()> {
    let mib = bytes as f64 / 1024.0 / 1024.0;
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    let mibps = mib / elapsed;
    match total {
        Some(t) if t > 0 => {
            let pct = (bytes as f64 / t as f64) * 100.0;
            let total_mib = t as f64 / 1024.0 / 1024.0;
            write!(
                out,
                "\r  {:>6.1} / {:>6.1} MiB  ({:>5.1}%)  {:>5.1} MiB/s",
                mib, total_mib, pct, mibps
            )?;
        }
        _ => write!(out, "\r  {:>6.1} MiB  {:>5.1} MiB/s", mib, mibps)?,
    }
    out.flush()
}

fn update_default_symlink(target: &std::path::Path, link: &std::path::Path) -> Result<()> {
    if link.is_symlink() || link.exists() {
        let _ = fs::remove_file(link);
    }
    std::os::unix::fs::symlink(target, link)
        .with_context(|| format!("symlink {} → {}", link.display(), target.display()))?;
    Ok(())
}
