//! Runtime configuration. Written by `tux init`, loaded at startup.
//! Lives at `~/.config/tux/config.toml` — edit freely.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TuxConfig {
    #[serde(default)]
    pub backend: BackendConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
}

/// Selects local GGUF inference vs. a remote OpenAI-compat API.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    #[default]
    Local,
    Remote,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendConfig {
    #[serde(default)]
    pub kind: BackendType,
    /// GGUF model file path (local backend).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,
    /// mmproj path for vision models (local backend, optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mmproj_path: Option<PathBuf>,
    /// Base URL for an OpenAI-compat API (remote backend).
    /// Example: `http://aperture.gerbil-dragon.ts.net`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Model name sent to the remote API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// API key (remote backend, optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// Hardware-tuned inference parameters written by `tux init`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// CPU threads for inference. 0 = auto-detect physical cores at startup.
    pub n_threads: u32,
    /// Context window in tokens. Larger = more RAM/VRAM.
    pub ctx_size: u32,
    /// GPU layers to offload. 0 = CPU-only; 99 = all layers on GPU.
    pub n_gpu_layers: u32,
    /// Prompt-evaluation batch size. Larger = faster prefill, more RAM.
    pub batch_size: u32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            n_threads: 0,
            ctx_size: 4096,
            n_gpu_layers: 0,
            batch_size: 512,
        }
    }
}

impl TuxConfig {
    /// Load from `~/.config/tux/config.toml`. Returns defaults when the
    /// file is absent or unparseable.
    pub fn load() -> Self {
        let path = Self::path();
        let Ok(s) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&s).unwrap_or_else(|e| {
            tracing::warn!("config parse error ({}): {e}; using defaults", path.display());
            Self::default()
        })
    }

    /// Write to `~/.config/tux/config.toml`, creating parent dirs as needed.
    pub fn save(&self) -> anyhow::Result<PathBuf> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(path)
    }

    pub fn path() -> PathBuf {
        directories::ProjectDirs::from("dev", "tux", "tux")
            .map(|d| d.config_dir().join("config.toml"))
            .unwrap_or_else(|| PathBuf::from("config.toml"))
    }
}
