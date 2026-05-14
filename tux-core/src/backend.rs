//! Model backends. The `Backend` trait abstracts over local llama.cpp
//! inference so the agent loop can be exercised without a real model.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// One message in a chat-style conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: usize,
    /// Image files to attach to the *last user message*. Vision-capable
    /// backends inspect them; text-only backends ignore them silently.
    pub images: Vec<std::path::PathBuf>,
}

impl CompletionRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            max_tokens: 512,
            images: vec![],
        }
    }

    pub fn with_images(mut self, images: Vec<std::path::PathBuf>) -> Self {
        self.images = images;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub text: String,
}

#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse>;
}

/// Identifies which backend implementation to construct. Kept as data so the
/// CLI / GUI can make the choice at runtime.
#[derive(Debug, Clone)]
pub enum BackendKind {
    Mock,
    #[cfg(feature = "llama")]
    LlamaCpp {
        model_path: std::path::PathBuf,
        /// Optional multimodal projection file. If provided, the backend
        /// loads an `MtmdContext` and can answer requests with attached
        /// images. Must be matched to the same model family.
        mmproj_path: Option<std::path::PathBuf>,
    },
}

/// Minimal stand-in backend for development: echoes the last user message
/// with a marker. Lets the agent loop run without a model file.
pub struct MockBackend;

/// Test-only backend that returns a pre-programmed sequence of responses
/// and records every request it received. Useful for exercising the agent
/// loop, tool dispatch, and multi-hop scenarios without invoking a real
/// model.
pub struct ScriptedBackend {
    responses: std::sync::Mutex<std::collections::VecDeque<String>>,
    calls: std::sync::Mutex<Vec<CompletionRequest>>,
}

impl ScriptedBackend {
    pub fn new<I, S>(responses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            responses: std::sync::Mutex::new(
                responses.into_iter().map(Into::into).collect(),
            ),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<CompletionRequest> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl Backend for ScriptedBackend {
    fn name(&self) -> &'static str {
        "scripted"
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        self.calls.lock().unwrap().push(req);
        let text = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("ScriptedBackend ran out of responses"))?;
        Ok(CompletionResponse { text })
    }
}

#[async_trait]
impl Backend for MockBackend {
    fn name(&self) -> &'static str {
        "mock"
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();
        Ok(CompletionResponse {
            text: format!("[mock] you asked: {}", last_user.trim()),
        })
    }
}

#[cfg(feature = "llama")]
pub mod llama {
    //! llama.cpp backend. Compiled only with `--features llama`. Requires
    //! cmake + clang + libclang (provided by the nix dev shell).
    //!
    //! Uses the Qwen2.5 / ChatML prompt template by default since that's
    //! what we recommend as the small purpose-built model. Other instruct
    //! models with the same template (Qwen2, Hermes, etc.) work too.

    use super::*;
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel};
    use llama_cpp_2::mtmd::{
        mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
    };
    use llama_cpp_2::sampling::LlamaSampler;
    use std::ffi::CString;
    use std::num::NonZeroU32;
    use std::path::PathBuf;
    use std::pin::pin;
    use std::sync::{Arc, Mutex, OnceLock};

    /// The llama backend may only be initialized once per process.
    fn shared_backend() -> anyhow::Result<&'static LlamaBackend> {
        static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
        if let Some(b) = BACKEND.get() {
            return Ok(b);
        }
        let b = LlamaBackend::init().map_err(|e| anyhow::anyhow!("llama backend init: {e}"))?;
        Ok(BACKEND.get_or_init(|| b))
    }

    /// Heavy, non-clonable state. Wrapped in `Arc` so async callers can
    /// move it into `spawn_blocking` without lifetime gymnastics.
    struct Inner {
        model: LlamaModel,
        /// Optional multimodal context. Present iff the backend was
        /// constructed with an `mmproj_path`. When set, requests with
        /// images use the mtmd path.
        mtmd: Option<MtmdContext>,
        ctx_size: NonZeroU32,
        // Serialize generation: `LlamaContext` isn't safe to share and we
        // only want one inflight inference per process for now.
        gate: Mutex<()>,
    }

    pub struct LlamaCppBackend {
        inner: Arc<Inner>,
    }

    impl LlamaCppBackend {
        pub fn new(model_path: PathBuf) -> anyhow::Result<Self> {
            Self::new_with_mmproj(model_path, None)
        }

        pub fn new_with_mmproj(
            model_path: PathBuf,
            mmproj_path: Option<PathBuf>,
        ) -> anyhow::Result<Self> {
            anyhow::ensure!(
                model_path.exists(),
                "model file does not exist: {}",
                model_path.display()
            );
            let backend = shared_backend()?;
            let params = pin!(LlamaModelParams::default());
            let model = LlamaModel::load_from_file(backend, &model_path, &params)
                .map_err(|e| anyhow::anyhow!("load model {}: {e}", model_path.display()))?;

            let mtmd = match mmproj_path {
                Some(path) => {
                    anyhow::ensure!(
                        path.exists(),
                        "mmproj file does not exist: {}",
                        path.display()
                    );
                    let mut params = MtmdContextParams::default();
                    params.media_marker = CString::new(mtmd_default_marker())
                        .expect("default marker contains no nul");
                    let ctx = MtmdContext::init_from_file(
                        path.to_str().ok_or_else(|| {
                            anyhow::anyhow!("non-UTF8 mmproj path: {}", path.display())
                        })?,
                        &model,
                        &params,
                    )
                    .map_err(|e| anyhow::anyhow!("load mmproj {}: {e:?}", path.display()))?;
                    anyhow::ensure!(
                        ctx.support_vision(),
                        "mmproj does not support vision: {}",
                        path.display()
                    );
                    Some(ctx)
                }
                None => None,
            };

            Ok(Self {
                inner: Arc::new(Inner {
                    model,
                    mtmd,
                    ctx_size: NonZeroU32::new(4096).unwrap(),
                    gate: Mutex::new(()),
                }),
            })
        }

        /// Render the conversation in ChatML (Qwen2.5 / Qwen3 / Qwen3.5).
        /// We append an empty `<think></think>` block right after the
        /// assistant opener — this is the documented trick to skip the
        /// chain-of-thought block on Qwen3 / Qwen3.5. Without it, Qwen3.5
        /// thinks by default and the reasoning leaks into stdout.
        fn render_chatml(messages: &[ChatMessage]) -> String {
            let mut s = String::new();
            for m in messages {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                s.push_str(&format!(
                    "<|im_start|>{role}\n{}<|im_end|>\n",
                    m.content.trim_end()
                ));
            }
            // Open the assistant turn and pre-fill an empty think block so
            // the model emits a direct answer (Qwen3 / Qwen3.5 friendly;
            // harmless for Qwen2.5 which simply ignores it as prose).
            s.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
            s
        }

        /// Vision-aware variant: appends `n_images` `<__media__>` markers
        /// to the *last user message* so mtmd's tokenizer knows where to
        /// splice the image embeddings.
        fn render_chatml_with_images(messages: &[ChatMessage], n_images: usize) -> String {
            let marker = mtmd_default_marker();
            let mut s = String::new();
            // Find the index of the last user message.
            let last_user_idx = messages
                .iter()
                .rposition(|m| matches!(m.role, Role::User));
            for (i, m) in messages.iter().enumerate() {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                let mut content = m.content.trim_end().to_string();
                if Some(i) == last_user_idx {
                    for _ in 0..n_images {
                        content.push('\n');
                        content.push_str(marker);
                    }
                }
                s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            s.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
            s
        }
    }

    impl Inner {
        fn generate(&self, req: CompletionRequest) -> anyhow::Result<String> {
            let _g = self.gate.lock().unwrap();
            let backend = shared_backend()?;

            let has_images = !req.images.is_empty();
            anyhow::ensure!(
                !has_images || self.mtmd.is_some(),
                "request has {} image(s) but the model was loaded without an mmproj — \
                 download the vision model with `tux init --with-vision`",
                req.images.len()
            );

            let ctx_params = LlamaContextParams::default().with_n_ctx(Some(self.ctx_size));
            let mut ctx = self
                .model
                .new_context(backend, ctx_params)
                .map_err(|e| anyhow::anyhow!("new context: {e}"))?;

            // -- prime the context: prompt → tokens (text) or chunks (vision)
            let n_max = (req.max_tokens as i32).max(16);
            let n_past_after_prompt;

            if has_images {
                let mtmd = self.mtmd.as_ref().expect("checked above");
                let prompt = LlamaCppBackend::render_chatml_with_images(
                    &req.messages,
                    req.images.len(),
                );

                let mut bitmaps = Vec::with_capacity(req.images.len());
                for path in &req.images {
                    let s = path.to_str().ok_or_else(|| {
                        anyhow::anyhow!("non-UTF8 image path: {}", path.display())
                    })?;
                    bitmaps.push(
                        MtmdBitmap::from_file(mtmd, s)
                            .map_err(|e| anyhow::anyhow!("load image {s}: {e:?}"))?,
                    );
                }
                let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

                let chunks = mtmd
                    .tokenize(
                        MtmdInputText {
                            text: prompt,
                            add_special: true,
                            parse_special: true,
                        },
                        &bitmap_refs,
                    )
                    .map_err(|e| anyhow::anyhow!("mtmd tokenize: {e:?}"))?;

                // Evaluate the chunks (text + image embeddings) into the
                // llama context. logits_last=true so we can sample next.
                let new_n_past = chunks
                    .eval_chunks(mtmd, &ctx, 0, 0, 512, true)
                    .map_err(|e| anyhow::anyhow!("mtmd eval_chunks: {e:?}"))?;
                n_past_after_prompt = new_n_past;
            } else {
                let prompt = LlamaCppBackend::render_chatml(&req.messages);
                let tokens = self
                    .model
                    .str_to_token(&prompt, AddBos::Always)
                    .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

                let n_ctx = ctx.n_ctx() as i32;
                anyhow::ensure!(
                    tokens.len() as i32 + n_max <= n_ctx,
                    "prompt + max_tokens ({}) exceeds context window ({})",
                    tokens.len() as i32 + n_max,
                    n_ctx
                );

                let mut batch = LlamaBatch::new(512, 1);
                let last_index = tokens.len() as i32 - 1;
                for (i, token) in tokens.iter().enumerate() {
                    let i = i as i32;
                    batch
                        .add(*token, i, &[0], i == last_index)
                        .map_err(|e| anyhow::anyhow!("batch add: {e}"))?;
                }
                ctx.decode(&mut batch)
                    .map_err(|e| anyhow::anyhow!("decode prompt: {e}"))?;
                n_past_after_prompt = tokens.len() as i32;
            }

            // -- sample loop (identical for both paths)
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::temp(0.7),
                LlamaSampler::top_p(0.9, 1),
                LlamaSampler::dist(1234),
            ]);

            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let mut out = String::new();
            let mut n_cur = n_past_after_prompt;
            let n_end = n_past_after_prompt + n_max;
            let mut batch = LlamaBatch::new(512, 1);

            while n_cur <= n_end {
                let last_logit_idx = if has_images && n_cur == n_past_after_prompt {
                    // After eval_chunks, the *next* token's logits live at
                    // the last position; we just call sample() at index 0
                    // because the per-step batch only contains 1 token.
                    0
                } else {
                    batch.n_tokens().saturating_sub(1)
                };
                let token = sampler.sample(&ctx, last_logit_idx);
                sampler.accept(token);

                if self.model.is_eog_token(token) {
                    break;
                }

                let piece = self
                    .model
                    .token_to_piece(token, &mut decoder, true, None)
                    .map_err(|e| anyhow::anyhow!("token_to_piece: {e}"))?;
                out.push_str(&piece);

                if out.contains("<|im_end|>") {
                    if let Some(idx) = out.find("<|im_end|>") {
                        out.truncate(idx);
                    }
                    break;
                }

                batch.clear();
                batch
                    .add(token, n_cur, &[0], true)
                    .map_err(|e| anyhow::anyhow!("batch add gen: {e}"))?;
                ctx.decode(&mut batch)
                    .map_err(|e| anyhow::anyhow!("decode gen: {e}"))?;
                n_cur += 1;
            }

            Ok(out.trim().to_string())
        }
    }

    #[async_trait]
    impl Backend for LlamaCppBackend {
        fn name(&self) -> &'static str {
            "llama.cpp"
        }

        async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
            // llama_context isn't Send across awaits; clone the Arc and
            // do the blocking work on a dedicated thread.
            let inner = self.inner.clone();
            let text = tokio::task::spawn_blocking(move || inner.generate(req)).await??;
            Ok(CompletionResponse { text })
        }
    }

    /// Construct from a `BackendKind` without leaking llama-cpp-2 types
    /// across crate boundaries.
    pub fn from_kind(kind: &BackendKind) -> anyhow::Result<Arc<dyn Backend>> {
        match kind {
            BackendKind::Mock => Ok(Arc::new(MockBackend)),
            BackendKind::LlamaCpp {
                model_path,
                mmproj_path,
            } => Ok(Arc::new(LlamaCppBackend::new_with_mmproj(
                model_path.clone(),
                mmproj_path.clone(),
            )?)),
        }
    }
}
