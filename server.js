
// server.js — OpenAI-compatible proxy for NVIDIA NIM
// Express 5 compatible.

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = 180000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}
validateConfig();

// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'gemini-turbo?': 'abacusai/dracarys-llama-3.1-70b-instruct',
  'gpt-3.5o': 'nvidia/nemotron-mini-4b-instruct',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash',
  'glm-5.2': 'z-ai/glm-5.2',
  'mistral': 'mistralai/mistral-large-3-675b-instruct-2512',
  'mistral-turbo': 'mistralai/mistral-medium-3.5-128b',
  'mistral-pro': 'mistralai/mistral-small-4-119b-2603',
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'mistralai/ministral-14b-instruct-2512',
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'google/gemma-2-2b-it',
  'google-lighter': 'google/gemma-3n-e4b-it',
  'm2.7': 'minimaxai/minimax-m2.7',
  'm3': 'minimaxai/minimax-m3',
  'step-3.5-flash': 'stepfun-ai/step-3.5-flash',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash'
};

// Default model used when an unrecognized alias is requested.
const DEFAULT_MODEL = 'nvidia/llama-3.3-nemotron-super-49b-v1.5';

const FALLBACK_MODELS = [
  'mistralai/mistral-medium-3.5-128b',
  'mistralai/mistral-small-4-119b-2603',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'google/gemma-4-31b-it'
];

// ─── Reasoning subsystem notes ─────────────────────────────────────────────
// Reasoning/thinking parameters vary by backend model and aren't part of the
// OpenAI schema, so they can't just be forwarded as-is — getReasoningPayload()
// below maps each backend model to its own request shape.
//
// Everything getReasoningPayload() returns is spread directly into the
// top-level JSON body sent to NIM via axios. Do NOT wrap it in an
// `extra_body` key — that's an openai-SDK-only convention, unwrapped
// client-side by the official SDKs and merged into the outgoing JSON as
// top-level fields. This proxy talks to NIM's raw REST endpoint via axios
// directly, so a literal "extra_body" field is just silently ignored.
// Confirmed against NVIDIA's own curl docs, which send chat_template_kwargs
// directly at the top level.
//
// GLM models think by default. `reasoning_effort` only controls thinking
// *intensity* once thinking is already happening — it does NOT turn thinking
// off. The actual on/off switch is the top-level `thinking: { type: "enabled"
// | "disabled" }` field (per z.ai's docs). GLM-5.2 accepts only "high" or
// "max" for reasoning_effort, with "max" as the default.
//
// nemotron-3-ultra's `force_nonempty_content` flag is NOT a confirmed NVIDIA
// parameter — left in as opt-in/best-effort since unrecognized
// chat_template_kwargs are typically ignored by the backend rather than
// causing a hard failure.
//
// nemotron-3-super and nemotron-3-ultra also expose a `low_effort: true`
// chat_template_kwargs flag — a middle ground between full reasoning and
// off, but a fixed tier, not a self-deciding mode. Reachable by sending
// reasoning_effort: "low" on a request.
//
// MiniMax-M3 controls reasoning via chat_template_kwargs.thinking_mode:
// "enabled" | "disabled" | "adaptive" (confirmed against NVIDIA's own NIM
// API reference for this model). "adaptive" is the only genuinely
// self-deciding reasoning mode in this proxy — the model chooses whether to
// think per-turn — and is reachable by sending reasoning_effort: "adaptive".
// M3 also emits its reasoning inline in content wrapped in
// <mm:think>...</mm:think> — a different tag than the generic <think> used
// by qwen/nemotron-super — so it needs its own entry in
// CONTENT_DELIMITER_TAGS or the tags leak straight into content unparsed.
//
// google/gemma-4-31b-it needs TWO separate flags to actually see reasoning
// output: chat_template_kwargs.enable_thinking turns thinking on, but the
// `reasoning` field is only populated in the response if include_reasoning
// is ALSO sent as true at the top level (confirmed against NVIDIA's VLM
// NIM docs). Sending enable_thinking alone makes the model reason
// internally with nothing to show for it.
//
// Reasoning output format: by default, reasoning is kept out of `content`
// and returned in a structured `reasoning`/`reasoning_content` field.
// Clients that expect legacy inline <thinking> tags baked into content can
// opt in by sending the `x-reasoning-format: inline` header.

// Backend models that embed reasoning inline in `content` via delimiter tags,
// rather than returning it as a separate structured field. Mapped to their
// specific tag pair so DelimiterParser knows what to look for.
const CONTENT_DELIMITER_TAGS = {
  'qwen/qwen3.5-397b-a17b': ['<think>', '</think>'],
  'nvidia/llama-3.3-nemotron-super-49b-v1.5': ['<think>', '</think>'],
  // MiniMax-M3 uses its own namespaced tag, not the generic <think> one.
  'minimaxai/minimax-m3': ['<mm:think>', '</mm:think>']
};

// Pure, stateful string parser for extracting reasoning blocks across chunks.
class DelimiterParser {
  constructor(openTag, closeTag) {
    this.openTag = openTag;
    this.closeTag = closeTag;
    this.inThinking = false;
    this.buffer = '';
  }

  processChunk(chunk) {
    this.buffer += chunk;
    let content = '';
    let reasoning = '';

    while (true) {
      const targetTag = this.inThinking ? this.closeTag : this.openTag;
      const tagIndex = this.buffer.indexOf(targetTag);

      if (tagIndex !== -1) {
        const textBefore = this.buffer.substring(0, tagIndex);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.inThinking = !this.inThinking;
        this.buffer = this.buffer.substring(tagIndex + targetTag.length);
      } else {
        // Check for partial tag at the end
        let partialLen = 0;
        const maxLen = Math.min(this.buffer.length, targetTag.length - 1);
        for (let i = maxLen; i > 0; i--) {
          if (targetTag.startsWith(this.buffer.substring(this.buffer.length - i))) {
            partialLen = i;
            break;
          }
        }
        const textBefore = this.buffer.substring(0, this.buffer.length - partialLen);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.buffer = this.buffer.substring(this.buffer.length - partialLen);
        break;
      }
    }

    return { content, reasoning };
  }

  flush() {
    let content = '';
    let reasoning = '';
    if (this.buffer) {
      if (this.inThinking) {
        reasoning += this.buffer;
      } else {
        content += this.buffer;
      }
      this.buffer = '';
    }
    return { content, reasoning };
  }
}

// Normalizes structured reasoning fields and extracts content delimiters.
class StreamNormalizer {
  constructor(model) {
    this.model = model;
    this.parser = null;
    // ONLY use content delimiters for models that embed reasoning in content
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) {
      this.parser = new DelimiterParser(tags[0], tags[1]);
    }
    // Models like Gemma 4, DeepSeek, GPT-OSS use structured fields and are NOT parsed here.
  }

  processDelta(delta) {
    const normalizedDelta = { ...delta };
    let reasoning = normalizedDelta.reasoning || normalizedDelta.reasoning_content || '';
    let content = normalizedDelta.content || '';

    // Priority: Structured reasoning > Content delimiters
    if (!reasoning && content && this.parser) {
      const parsed = this.parser.processChunk(content);
      reasoning = parsed.reasoning;
      content = parsed.content;
    }

    if (content) normalizedDelta.content = content;
    else delete normalizedDelta.content;

    if (reasoning) normalizedDelta.reasoning = reasoning;
    else delete normalizedDelta.reasoning;

    delete normalizedDelta.reasoning_content;
    return normalizedDelta;
  }

  flush() {
    if (!this.parser) return { content: '', reasoning: '' };
    return this.parser.flush();
  }
}

function normalizeNonStreamChoice(choice, model) {
  if (!choice) return choice;
  const message = choice.message || {};
  let reasoning = message.reasoning || message.reasoning_content || '';
  let content = message.content || '';

  if (!reasoning && content) {
    let parser = null;
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) {
      parser = new DelimiterParser(tags[0], tags[1]);
    }
    if (parser) {
      const parsed = parser.processChunk(content);
      const flushed = parser.flush();
      content = (parsed.content || '') + (flushed.content || '');
      reasoning = (parsed.reasoning || '') + (flushed.reasoning || '');
    }
  }

  const newMessage = { ...message };
  if (content) newMessage.content = content;
  if (reasoning) newMessage.reasoning = reasoning;
  delete newMessage.reasoning_content;

  return { ...choice, message: newMessage };
}

// Valid reasoning_effort values per backend model, where the backend enforces
// an enum. Anything outside this set is dropped rather than forwarded, so a
// bad client value fails fast in proxy logs instead of as an opaque upstream
// 400.
const REASONING_EFFORT_ENUMS = {
  'openai/gpt-oss-120b': ['low', 'medium', 'high'],
  'openai/gpt-oss-20b': ['low', 'medium', 'high'],
  'mistralai/mistral-medium-3.5-128b': ['high', 'none'],
  'mistralai/mistral-small-4-119b-2603': ['high', 'none'],
  'z-ai/glm-5.2': ['high', 'max'],
  // Confirmed via NVIDIA's own build.nvidia.com curl examples and vLLM's
  // official recipe docs: DeepSeek V4 accepts only these two once thinking
  // is on. This was previously missing here, meaning literally any string a
  // client sent was forwarded to NIM unchecked in the deepseek-v4 case below.
  'deepseek-ai/deepseek-v4-pro': ['high', 'max'],
  'deepseek-ai/deepseek-v4-flash': ['high', 'max'],
  // Not a true adaptive/effort scale — these two only expose a single extra
  // "low_effort" middle tier between full reasoning and off.
  'nvidia/nemotron-3-super-120b-a12b': ['low'],
  'nvidia/nemotron-3-ultra-550b-a55b': ['low'],
  // MiniMax-M3's only non-binary option: let the model decide per-turn.
  'minimaxai/minimax-m3': ['adaptive']
};

function validReasoningEffort(model, effort) {
  const allowed = REASONING_EFFORT_ENUMS[model];
  if (!allowed) return effort; // no enum enforced for this model, pass through
  if (allowed.includes(effort)) return effort;
  if (effort) {
    console.warn(`[REASONING] Dropping invalid reasoning_effort "${effort}" for ${model} (allowed: ${allowed.join(', ')})`);
  }
  return undefined;
}

// Pure function returning model-specific reasoning request payloads.
// IMPORTANT: everything returned here gets spread DIRECTLY into the top-level
// JSON body sent to NIM via axios. Do NOT wrap anything in an `extra_body` key —
// see the reasoning subsystem notes above.
function getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools) {
  const effort = validReasoningEffort(model, clientReasoningEffort);

  switch (model) {
    case 'nvidia/nemotron-3-super-120b-a12b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      if (effort === 'low') payload.chat_template_kwargs.low_effort = true;
      return payload;
    }

    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      if (effort === 'low') payload.chat_template_kwargs.low_effort = true;
      // Unverified param — see header comment. Left as opt-in best-effort.
      if (hasTools) payload.chat_template_kwargs.force_nonempty_content = true;
      return payload;
    }

    case 'qwen/qwen3.5-397b-a17b': {
      // Model appears to default to thinking-on in its chat template. Only send
      // a field when the caller explicitly wants thinking OFF; otherwise let the
      // <think> delimiter parser handle whatever the model does natively.
      if (enableThinking) return {};
      return { chat_template_kwargs: { enable_thinking: false } };
    }

    case 'deepseek-ai/deepseek-v4-pro':
    case 'deepseek-ai/deepseek-v4-flash': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { thinking: true } };
      if (effort) payload.chat_template_kwargs.reasoning_effort = effort;
      return payload;
    }

    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b': {
      if (effort) return { reasoning_effort: effort };
      if (enableThinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'mistralai/mistral-medium-3.5-128b':
    case 'mistralai/mistral-small-4-119b-2603': {
      if (effort) return { reasoning_effort: effort };
      if (enableThinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'z-ai/glm-5.2': {
      // GLM-5.2 thinks by default. `reasoning_effort` only controls
      // intensity (max vs high) once thinking is already happening — it does
      // NOT turn thinking off. The actual on/off switch is `thinking.type`.
      const payload = {
        thinking: { type: enableThinking ? 'enabled' : 'disabled' }
      };
      if (enableThinking && effort) payload.reasoning_effort = effort;
      return payload;
    }

    case 'google/gemma-4-31b-it': {
      if (!enableThinking) return {};
      // enable_thinking only makes the model reason internally — it does NOT
      // by itself put that reasoning in the response. NVIDIA's own VLM docs
      // require a separate top-level include_reasoning flag to actually
      // return the `reasoning` field; without it we may be paying the
      // latency/token cost of thinking and never seeing the output. Match it
      // to SHOW_REASONING so behavior is explicit instead of relying on
      // whatever include_reasoning defaults to upstream.
      return {
        chat_template_kwargs: { enable_thinking: true },
        include_reasoning: SHOW_REASONING
      };
    }

    case 'stepfun-ai/step-3.7-flash': {
      if (enableThinking) return {};
      return { chat_template_kwargs: { thinking: false } };
    }

    case 'minimaxai/minimax-m3': {
      // Per NVIDIA's own NIM API reference, MiniMax-M3 controls reasoning via
      // chat_template_kwargs.thinking_mode: "enabled" | "disabled" | "adaptive".
      // "adaptive" lets the model decide per-turn whether to think — the only
      // genuinely self-deciding reasoning mode across every model in this
      // proxy. Send reasoning_effort: "adaptive" on a request to use it;
      // otherwise this falls back to the standard on/off toggle like every
      // other model here.
      const thinkingMode = effort === 'adaptive'
        ? 'adaptive'
        : (enableThinking ? 'enabled' : 'disabled');
      return { chat_template_kwargs: { thinking_mode: thinkingMode } };
    }

    default:
      // Default reasoning models (Kimi, MiniMax, etc.) or non-reasoning models
      return {};
  }
}

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Catch malformed JSON bodies so clients get a clean OpenAI-style error
// instead of Express's default HTML error page. Without this, a broken
// request body never reaches the route handler's try/catch at all — Express
// throws during body parsing, before routing happens.
app.use((err, req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({
      error: {
        message: 'Invalid JSON in request body',
        type: 'invalid_request_error',
        code: 400
      }
    });
  }
  next(err);
});

// Extract token AFTER "Bearer " prefix, compare only the token
function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);
  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ─── Validation ─────────────────────────────────────────────────────────────

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');
  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: VALIDATION_TIMEOUT_MS
    });

    const availableModels = new Set(
      (response.data.data || []).map(m => m.id)
    );

    const invalid = [];
    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(invalid);
    } else {
      console.log('[VALIDATION] All models valid.');
    }
  } catch (err) {
    console.warn(`[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`);
    console.warn('[VALIDATION] Consider setting SKIP_VALIDATION=true if your NIM provider lacks a model listing endpoint.');
  }
}

async function sendDiscordAlert(invalidModels) {
  if (!DISCORD_WEBHOOK_URL) return;

  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, {
      embeds: [embed],
      username: 'NIM Proxy Monitor'
    }, { timeout: 5000 });
    console.log('[DISCORD] Alert sent.');
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────

function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Helper: Fallback Chain ─────────────────────────────────────────────────

async function callWithFallback(baseRequest, models, enableThinking, clientReasoningEffort, hasTools) {
  let lastError = null;

  for (const model of models) {
    try {
      const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools);
      const res = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        { ...baseRequest, model, ...reasoningPayload },
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: baseRequest.stream ? 'stream' : 'json',
          timeout: REQUEST_TIMEOUT_MS
        }
      );
      return { response: res, model };
    } catch (err) {
      lastError = err;
      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        err.response?.status,
        err.response?.data?.error?.message || err.message
      );
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.3.0' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      // OpenAI's spec documents this field in Unix seconds, not milliseconds
      // — Date.now() alone is 1000x too large and inconsistent with the
      // correctly-converted timestamp used below in the chat completions
      // response.
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream,
      tools,
      tool_choice
    } = req.body;

    let primaryModel = MODEL_MAPPING[model];
    if (!primaryModel) {
      console.warn(`[PROXY] Unknown model alias "${model}", falling back to default: ${DEFAULT_MODEL}`);
      primaryModel = DEFAULT_MODEL;
    }

    // De-dupe in case the requested alias resolves to a model that's also in
    // the fallback chain — otherwise a failure retries the identical model
    // twice before actually diversifying.
    const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])];

    const baseRequest = {
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false,
      // Forward tool-calling fields as-is. Without this, clients using
      // function/tool calling silently get a plain chat completion back —
      // NIM never sees the tool definitions, so it never returns tool_calls.
      ...(tools && { tools }),
      ...(tool_choice && { tool_choice })
    };

    const { response, model: usedModel } = await callWithFallback(
      baseRequest,
      modelChain,
      ENABLE_THINKING_MODE,
      req.body.reasoning_effort,
      !!tools
    );

    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    // Determine if the client wants legacy inline <thinking> tags in the content stream
    const inlineReasoning = req.headers['x-reasoning-format'] === 'inline';

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;
      const normalizer = new StreamNormalizer(usedModel);

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        // Exact match only. .includes() would false-positive on any legitimate
        // model output that happens to contain the literal substring "[DONE]"
        // in its generated content (e.g. a coding/task-tracking response),
        // silently truncating the reply right there.
        if (line.trim() === 'data: [DONE]') {
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            const normalizedDelta = normalizer.processDelta(delta);
            let clientContent = '';

            if (SHOW_REASONING && inlineReasoning) {
              // Inline reasoning format: bake <thinking> tags into content
              // for clients that don't parse structured reasoning fields.
              if (normalizedDelta.reasoning && !reasoningOpen) {
                clientContent += `<thinking>\n${normalizedDelta.reasoning}`;
                reasoningOpen = true;
              } else if (normalizedDelta.reasoning) {
                clientContent += normalizedDelta.reasoning;
              }

              if (normalizedDelta.content && reasoningOpen) {
                clientContent += `\n</thinking>\n\n${normalizedDelta.content}`;
                reasoningOpen = false;
              } else if (normalizedDelta.content) {
                clientContent += normalizedDelta.content;
              }
            } else {
              // Default behavior: clean content, no inline tags
              clientContent = normalizedDelta.content || '';
            }

            delta.content = clientContent;

            // Keep a structured reasoning field alongside inline tags in
            // content. Some clients parse the inline <thinking> tags;
            // others (OpenRouter-style apps) look for a separate
            // `reasoning`/`reasoning_content` field to render their own
            // collapsible thinking UI. Send both so either style works.
            if (SHOW_REASONING && normalizedDelta.reasoning) {
              delta.reasoning = normalizedDelta.reasoning;
              delta.reasoning_content = normalizedDelta.reasoning;
            } else {
              delete delta.reasoning;
              delete delta.reasoning_content;
            }
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
        } catch {
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Upstream sent malformed chunk',
              type: 'stream_parse_error',
              details: line.slice(0, 100)
            }
          })}\n\n`);
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream buffer overflow',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();
        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        const flushedDelta = normalizer.flush();
        if (flushedDelta.content || flushedDelta.reasoning) {
          let clientContent = '';

          if (SHOW_REASONING && inlineReasoning) {
            // Inline reasoning format: bake <thinking> tags into content.
            if (flushedDelta.reasoning && !reasoningOpen) {
              clientContent += `<thinking>\n${flushedDelta.reasoning}`;
              reasoningOpen = true;
            } else if (flushedDelta.reasoning) {
              clientContent += flushedDelta.reasoning;
            }

            if (flushedDelta.content && reasoningOpen) {
              clientContent += `\n</thinking>\n\n${flushedDelta.content}`;
              reasoningOpen = false;
            } else if (flushedDelta.content) {
              clientContent += flushedDelta.content;
            }
          } else {
            // Default behavior: clean content, no inline tags
            clientContent = flushedDelta.content || '';
          }

          const finalChunk = { choices: [{ delta: {} }] };
          if (clientContent) finalChunk.choices[0].delta.content = clientContent;

          // Mirror the per-chunk handling above: leftover reasoning text
          // must also reach structured-format clients, not just get folded
          // into inline tags. Previously this was dropped entirely whenever
          // SHOW_REASONING was on but inlineReasoning was off.
          if (SHOW_REASONING && !inlineReasoning && flushedDelta.reasoning) {
            finalChunk.choices[0].delta.reasoning = flushedDelta.reasoning;
            finalChunk.choices[0].delta.reasoning_content = flushedDelta.reasoning;
          }

          if (Object.keys(finalChunk.choices[0].delta).length > 0) {
            safeWrite(res, `data: ${JSON.stringify(finalChunk)}\n\n`);
          }
        }

        // A model can get cut off mid-reasoning (e.g. hits max_tokens while
        // still inside a <think> block, never emitting the closing tag).
        // If we opened an inline <thinking> tag earlier and nothing above
        // closed it, close it now so inline-format clients aren't left with
        // an unterminated tag.
        if (SHOW_REASONING && inlineReasoning && reasoningOpen) {
          safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: { content: '\n</thinking>\n' } }] })}\n\n`);
          reasoningOpen = false;
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }
        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream interrupted by upstream error',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;
        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
        }
        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });
    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        // Report the model that actually answered (which may differ from the
        // requested alias if a fallback kicked in), not the raw client input.
        model: usedModel,
        created: Math.floor(Date.now() / 1000),
        choices: (response.data.choices || []).map((choice, i) => {
          const normalizedChoice = normalizeNonStreamChoice(choice, usedModel);
          let content = normalizedChoice.message?.content || '';
          const reasoning = normalizedChoice.message?.reasoning || '';

          if (SHOW_REASONING && inlineReasoning && reasoning) {
            // Inline reasoning format: bake <thinking> tags into content.
            content = `<thinking>\n${reasoning}\n</thinking>\n\n${content}`;
          }

          const finalMessage = { ...normalizedChoice.message, content };

          // Same as the streaming path: keep the structured field alongside
          // the inline tags so structured-reasoning clients can render their
          // own UI.
          if (SHOW_REASONING && reasoning) {
            finalMessage.reasoning = reasoning;
            finalMessage.reasoning_content = reasoning;
          } else {
            delete finalMessage.reasoning;
            delete finalMessage.reasoning_content;
          }

          const finalChoice = {
            ...normalizedChoice,
            index: i,
            message: finalMessage
          };
          return finalChoice;
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }
  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    console.error('[PROXY] NIM response:', error.response?.data);

    if (!res.headersSent) {
      // If this fires after the streaming branch already called
      // res.setHeader('Content-Type', 'text/event-stream') but before any
      // actual write, res.json() below won't override it — Express's
      // res.json() only sets Content-Type when it isn't already set. Force
      // it back to JSON explicitly so the error body's declared type
      // actually matches its content.
      res.set('Content-Type', 'application/json');
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error'
        }
      })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Startup ───────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});

 
