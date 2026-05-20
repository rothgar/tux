//! Model backends. The `Backend` trait abstracts over local llama.cpp
//! inference and remote OpenAI-compat APIs so the agent loop can be
//! exercised without a real model.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
        /// Optional multimodal projection file for vision requests.
        mmproj_path: Option<std::path::PathBuf>,
        /// CPU threads. 0 = auto-detect physical cores at load time.
        n_threads: u32,
        /// Context window in tokens.
        ctx_size: u32,
        /// GPU layers to offload. 0 = CPU-only; 99 = all layers.
        n_gpu_layers: u32,
        /// Prompt evaluation batch size.
        batch_size: u32,
    },
    OpenAICompat {
        url: String,
        model: String,
        api_key: Option<String>,
    },
}

/// Construct a backend from a `BackendKind` descriptor.
pub fn from_kind(kind: BackendKind) -> anyhow::Result<Arc<dyn Backend>> {
    match kind {
        BackendKind::Mock => Ok(Arc::new(MockBackend)),
        BackendKind::OpenAICompat { url, model, api_key } => {
            Ok(Arc::new(OpenAICompatBackend::new(url, model, api_key)))
        }
        #[cfg(feature = "llama")]
        BackendKind::LlamaCpp {
            model_path,
            mmproj_path,
            n_threads,
            ctx_size,
            n_gpu_layers,
            batch_size,
        } => llama::LlamaCppBackend::new_with_mmproj(
            model_path,
            mmproj_path,
            n_threads,
            ctx_size,
            n_gpu_layers,
            batch_size as usize,
        )
        .map(|b| Arc::new(b) as Arc<dyn Backend>),
    }
}

/// Strip `<think>...</think>` blocks from model output. Handles Qwen3's
/// chain-of-thought tokens and orphaned `</think>` closers that leak when
/// the client doesn't pre-fill the thinking block.
pub(crate) fn strip_thinking(s: &str) -> String {
    let mut result = s.to_string();
    loop {
        match (result.find("<think>"), result.find("</think>")) {
            (Some(start), Some(end)) if start <= end => {
                let after = end + "</think>".len();
                result = format!("{}{}", &result[..start], &result[after..]);
            }
            _ => break,
        }
    }
    // Strip any orphaned </think> that survived (e.g. model output leak
    // when the client didn't pre-fill an opening <think> block).
    result = result.replace("</think>", "");
    result.trim().to_string()
}

// ---------------------------------------------------------------------------
// MockBackend / ScriptedBackend
// ---------------------------------------------------------------------------

/// Minimal stand-in backend for development: echoes the last user message
/// with a marker. Lets the agent loop run without a model file.
pub struct MockBackend;

/// Test-only backend that returns a pre-programmed sequence of responses
/// and records every request it received.
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

// ---------------------------------------------------------------------------
// OpenAICompatBackend
// ---------------------------------------------------------------------------

/// Backend that delegates to any OpenAI-compatible `/v1/chat/completions`
/// endpoint. Works with Ollama, vLLM, llama-server, and hosted providers.
/// Strips Qwen3 `<think>...</think>` blocks automatically.
pub struct OpenAICompatBackend {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl OpenAICompatBackend {
    pub fn new(base_url: String, model: String, api_key: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            api_key,
        }
    }
}

#[async_trait]
impl Backend for OpenAICompatBackend {
    fn name(&self) -> &'static str {
        "openai-compat"
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let messages: Vec<serde_json::Value> = req
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        Role::System    => "system",
                        Role::User      => "user",
                        Role::Assistant => "assistant",
                        Role::Tool      => "tool",
                    },
                    "content": m.content,
                })
            })
            .collect();

        let body = serde_json::json!({
            "model":      self.model,
            "messages":   messages,
            "max_tokens": req.max_tokens,
            "stream":     false,
        });

        let mut builder = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&body);

        if let Some(key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {key}"));
        }

        let resp = builder
            .send()
            .await?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("remote API: {e}"))?;

        let json: serde_json::Value = resp.json().await?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("unexpected API response shape: {json}"))?
            .to_string();

        Ok(CompletionResponse {
            text: strip_thinking(&text),
        })
    }
}

// ---------------------------------------------------------------------------
// LlamaCppBackend (feature = "llama")
// ---------------------------------------------------------------------------

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
    use llama_cpp_2::context::LlamaContext;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel};
    use llama_cpp_2::mtmd::{
        mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
    };
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::token::LlamaToken;
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
        let debug = std::env::var_os("TUX_DEBUG").is_some_and(|v| !v.is_empty());
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(debug),
        );
        let b = LlamaBackend::init().map_err(|e| anyhow::anyhow!("llama backend init: {e}"))?;
        Ok(BACKEND.get_or_init(|| b))
    }

    /// KV-warm context cached between requests.
    struct CacheState {
        ctx: LlamaContext<'static>,
        tokens: Vec<LlamaToken>,
    }

    // SAFETY: see comment on Inner below.
    unsafe impl Send for CacheState {}

    /// Heavy, non-clonable state. Wrapped in `Arc` so async callers can
    /// move it into `spawn_blocking` without lifetime gymnastics.
    ///
    /// FIELD ORDER MATTERS: `cache` must be declared before `model` and
    /// `mtmd` so Rust drops it first — `LlamaContext` borrows `LlamaModel`
    /// and must be freed before it.
    struct Inner {
        cache: Mutex<Option<CacheState>>,
        model: LlamaModel,
        mtmd: Option<MtmdContext>,
        ctx_size: NonZeroU32,
        /// Resolved thread count (already converted from 0 = auto).
        n_threads: i32,
        batch_size: usize,
    }

    pub struct LlamaCppBackend {
        inner: Arc<Inner>,
    }

    impl LlamaCppBackend {
        pub fn new(model_path: PathBuf) -> anyhow::Result<Self> {
            Self::new_with_mmproj(model_path, None, 0, 4096, 0, 512)
        }

        /// `n_threads` = 0 → auto-detect physical cores.
        /// `ctx_size`   = 0 → fallback to 4096.
        /// `n_gpu_layers` = 0 → CPU-only.
        pub fn new_with_mmproj(
            model_path: PathBuf,
            mmproj_path: Option<PathBuf>,
            n_threads: u32,
            ctx_size: u32,
            n_gpu_layers: u32,
            batch_size: usize,
        ) -> anyhow::Result<Self> {
            anyhow::ensure!(
                model_path.exists(),
                "model file does not exist: {}",
                model_path.display()
            );
            let backend = shared_backend()?;

            let resolved_threads = if n_threads == 0 {
                crate::context::physical_cores() as i32
            } else {
                n_threads as i32
            };
            let resolved_ctx =
                NonZeroU32::new(ctx_size.max(1)).unwrap_or(NonZeroU32::new(4096).unwrap());

            let params = pin!(LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers));
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
                    cache: Mutex::new(None),
                    model,
                    mtmd,
                    ctx_size: resolved_ctx,
                    n_threads: resolved_threads,
                    batch_size: batch_size.max(1),
                }),
            })
        }

        /// Render the conversation in ChatML (Qwen2.5 / Qwen3 / Qwen3.5).
        /// Pre-fills an empty `<think></think>` block after the assistant
        /// opener to suppress chain-of-thought on Qwen3 / Qwen3.5.
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
            s.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
            s
        }

        fn render_chatml_with_images(messages: &[ChatMessage], n_images: usize) -> String {
            let marker = mtmd_default_marker();
            let mut s = String::new();
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
        fn ensure_cache<'g>(
            &'g self,
            guard: &'g mut std::sync::MutexGuard<'_, Option<CacheState>>,
        ) -> anyhow::Result<&'g mut CacheState> {
            if guard.is_none() {
                let backend = shared_backend()?;
                let ctx_params = LlamaContextParams::default()
                    .with_n_ctx(Some(self.ctx_size))
                    .with_n_threads(self.n_threads)
                    .with_n_threads_batch(self.n_threads);
                let ctx = self
                    .model
                    .new_context(backend, ctx_params)
                    .map_err(|e| anyhow::anyhow!("new context: {e}"))?;
                // SAFETY: `ctx` borrows `self.model`. Both live in the same
                // `Arc<Inner>` (heap, stable address). `Inner.cache` is
                // declared before `Inner.model` so it drops first.
                let ctx_static: LlamaContext<'static> =
                    unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(ctx) };
                **guard = Some(CacheState {
                    ctx: ctx_static,
                    tokens: Vec::new(),
                });
            }
            Ok(guard.as_mut().expect("just initialized"))
        }

        fn generate(&self, req: CompletionRequest) -> anyhow::Result<String> {
            let mut cache_guard = self.cache.lock().unwrap();

            let has_images = !req.images.is_empty();
            anyhow::ensure!(
                !has_images || self.mtmd.is_some(),
                "request has {} image(s) but the model was loaded without an mmproj — \
                 download the vision model with `tux init --with-vision`",
                req.images.len()
            );

            if has_images {
                if let Some(state) = cache_guard.as_mut() {
                    state.ctx.clear_kv_cache();
                    state.tokens.clear();
                }
            }

            let cache = self.ensure_cache(&mut cache_guard)?;
            let ctx = &mut cache.ctx;

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

                let new_n_past = chunks
                    .eval_chunks(mtmd, ctx, 0, 0, 512, true)
                    .map_err(|e| anyhow::anyhow!("mtmd eval_chunks: {e:?}"))?;
                n_past_after_prompt = new_n_past;
                cache.tokens.clear();
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

                let mut prefix = cache
                    .tokens
                    .iter()
                    .zip(tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                if prefix == tokens.len() && prefix > 0 {
                    prefix -= 1;
                }

                if cache.tokens.len() > prefix {
                    ctx.clear_kv_cache_seq(Some(0), Some(prefix as u32), None)
                        .map_err(|e| anyhow::anyhow!("clear kv tail: {e:?}"))?;
                    cache.tokens.truncate(prefix);
                }

                let suffix = &tokens[prefix..];
                tracing::debug!(
                    "kv reuse: prefix={prefix} suffix={} total={}",
                    suffix.len(),
                    tokens.len()
                );

                let batch_cap = suffix.len().max(self.batch_size);
                let mut batch = LlamaBatch::new(batch_cap, 1);
                let last_index = suffix.len() as i32 - 1;
                for (i, token) in suffix.iter().enumerate() {
                    let pos = (prefix + i) as i32;
                    let i = i as i32;
                    batch
                        .add(*token, pos, &[0], i == last_index)
                        .map_err(|e| anyhow::anyhow!("batch add: {e}"))?;
                }
                ctx.decode(&mut batch)
                    .map_err(|e| anyhow::anyhow!("decode prompt: {e}"))?;
                cache.tokens.extend_from_slice(suffix);
                n_past_after_prompt = tokens.len() as i32;
            }

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
                let token = sampler.sample(ctx, 0);
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
                cache.tokens.push(token);
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
            let inner = self.inner.clone();
            let text = tokio::task::spawn_blocking(move || inner.generate(req)).await??;
            Ok(CompletionResponse { text })
        }
    }
}
