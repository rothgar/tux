//! Built-in model registry. Used by `tux init` to pick a sensible default
//! based on detected hardware. Kept as plain const data so we don't need
//! a config file or network call for the catalog itself.
//!
//! All entries are ChatML-template Qwen3.5 dense models — what the agent
//! and chat template are tuned for. Adding other families later means
//! also teaching `render_chatml` / the agent system prompt about them.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelKind {
    /// Pure text completion.
    Text,
    /// Vision-language model. Needs a separate multimodal projection file
    /// (`mmproj`) loaded alongside the GGUF.
    Vision {
        mmproj_url: &'static str,
        mmproj_size_mib: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Short identifier used on the CLI (`--model qwen3.5-4b-q4`).
    pub id: &'static str,
    /// Human-readable name shown during init.
    pub name: &'static str,
    /// HTTPS URL to the GGUF file. Resolves directly (no HF Hub API).
    pub url: &'static str,
    /// Approximate file size in MiB — used for both download display and
    /// hardware fit decisions.
    pub size_mib: u32,
    /// Minimum RAM (MiB) we want the host to have before recommending
    /// this model. Roughly model-on-disk × 1.3 to leave headroom.
    pub min_ram_mib: u32,
    /// Quality tier — purely for ordering when multiple models fit.
    /// Higher = better answers, all else equal.
    pub quality: u8,
    /// Whether this is a text-only or vision model.
    #[serde(default = "default_kind")]
    pub kind: ModelKind,
}

fn default_kind() -> ModelKind {
    ModelKind::Text
}

pub const REGISTRY: &[ModelEntry] = &[
    ModelEntry {
        id: "qwen3.5-4b-q4",
        name: "Qwen3.5-4B (Q4_K_M)",
        url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
        size_mib: 2500,
        min_ram_mib: 6 * 1024,
        quality: 80,
        kind: ModelKind::Text,
    },
    ModelEntry {
        id: "qwen3.5-2b-q4",
        name: "Qwen3.5-2B (Q4_K_M)",
        url: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
        size_mib: 1300,
        min_ram_mib: 3 * 1024,
        quality: 60,
        kind: ModelKind::Text,
    },
    ModelEntry {
        id: "qwen3.5-0.8b-q4",
        name: "Qwen3.5-0.8B (Q4_K_M)",
        url: "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
        size_mib: 600,
        min_ram_mib: 1500,
        quality: 30,
        kind: ModelKind::Text,
    },
    // Vision model: text + image inputs via mtmd + mmproj.
    // URLs assumed to follow the same `unsloth/<name>-GGUF` convention; if
    // the file isn't there yet at first run, override with --vision-model
    // pointing at any compatible GGUF + mmproj pair.
    ModelEntry {
        id: "qwen3-vl-2b",
        name: "Qwen3-VL-2B (Q4_K_M + mmproj)",
        url: "https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3-VL-2B-Instruct-Q4_K_M.gguf",
        size_mib: 1500,
        min_ram_mib: 4 * 1024,
        quality: 50,
        kind: ModelKind::Vision {
            mmproj_url: "https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3-VL-2B-Instruct-F16.gguf",
            mmproj_size_mib: 700,
        },
    },
];

/// Default vision model id chosen by `tux init --with-vision`.
pub const DEFAULT_VISION_MODEL: &str = "qwen3-vl-2b";

pub fn lookup(id: &str) -> Option<&'static ModelEntry> {
    REGISTRY.iter().find(|m| m.id == id)
}

fn is_text(m: &ModelEntry) -> bool {
    matches!(m.kind, ModelKind::Text)
}

/// Pick the best text entry that fits the host.
pub fn pick_for_host(total_ram_mib: u32, cpu_cores: u32) -> &'static ModelEntry {
    let max_quality_for_cores: u8 = match cpu_cores {
        0..=3 => 30, // 0.8B only
        4..=5 => 60, // up to 2B
        _ => 80,     // 4B
    };

    let best = REGISTRY
        .iter()
        .filter(|m| {
            is_text(m) && m.min_ram_mib <= total_ram_mib && m.quality <= max_quality_for_cores
        })
        .max_by_key(|m| m.quality);

    match best {
        Some(m) => m,
        None => REGISTRY
            .iter()
            .filter(|m| is_text(m))
            .min_by_key(|m| m.size_mib)
            .expect("registry must contain at least one text model"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_4b_on_a_capable_box() {
        let m = pick_for_host(16 * 1024, 12);
        assert_eq!(m.id, "qwen3.5-4b-q4");
    }

    #[test]
    fn picks_2b_on_modest_laptop() {
        let m = pick_for_host(8 * 1024, 4);
        assert_eq!(m.id, "qwen3.5-2b-q4");
    }

    #[test]
    fn picks_smallest_on_tiny_machine() {
        let m = pick_for_host(2 * 1024, 2);
        assert_eq!(m.id, "qwen3.5-0.8b-q4");
    }

    #[test]
    fn picks_smallest_when_nothing_fits() {
        let m = pick_for_host(512, 8);
        assert_eq!(m.id, "qwen3.5-0.8b-q4");
    }

    #[test]
    fn lookup_known_id() {
        assert!(lookup("qwen3.5-4b-q4").is_some());
        assert!(lookup("nope").is_none());
    }
}
