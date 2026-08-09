### THIS IS A TEST BRANCH WHERE I WILL TEST NEW FEATURES TO ADD, DO NOT FORK THIS AS IT IS EXPERIMENTAL AND MAY HAVE BUGS.


### Announcement 
Due to personal health reasons, the old owner will no longer be able to maintain this project. However, for the time being, I’m gonna try to maintain this!! The original repository I forked from is https://github.com/Jontte6/nim-to-openai-proxy 


### If you forked before June 7, 2026, please pull the latest version — previous versions had an auth bypass and startup DDoS vulnerability.

### Guys. Just don't use reasoning at this point. There's not a single standard that nim follows that I could make use of. The reasoning on one model works, then I fix it for the other and the first breaks. I’ll try to see if I can fix it but don’t count on it.


### NVIDIA NIM to OpenAI Proxy
Hello, this is my first ever project on Github that I am making public. This is essentially just a translation layer between the API format that NVIDIA NIM uses to the format OpenAI uses. Jontte made this originally by building on a script from a Reddit guide. Over the time of a month he had iterated on it, fixed problems, added auth, more models, and removed/replaced deprecated models.
These are the current available models for usage, and the use cases for all of them. (Note: The Google models are mostly for troubleshooting issues with latency and timeouts.)

### Why use this proxy?

A lot of frontends such as JanitorAI requires an OpenAI-compatible proxy to use NVIDIA NIM. SillyTavern can connect to NIM directly, but if you use **Lorebary** for prompts, lorebooks, or plugins, this proxy is necessary — Lorebary does not support NIM natively. Various other frontends for coding and other things also may bug with NIM.

### Legality

Yes, it's legal. It's just HTTP requests routed through your own proxy. You still need a valid NVIDIA API key and are subject to their rate limits. This is no different from using any other API gateway or reverse proxy.


### Requirements

Node.js 24+, a NVAPI/Nim API key, a deployment platform (though if you follow the guide below none of those should be a problem).

### Model Mapping

| Alias | Backend Model | Best For | Speed | Filters |
|---|---|---|---|---|
| `gpt-4-turbo` | `moonshotai/kimi-k2.6` | Deep, immersive RP | Medium | Medium-High |
| `gpt-4` | `nvidia/nemotron-3-ultra-550b-a55b` | Immersive RP | Fast | Low |
| `gpt-4o` | `deepseek-ai/deepseek-v4-pro` |
| `gpt-4-flash` | `deepseek-ai/deepseek-v4-flash` | Fast, non-edgy RP | Fast | High |
| `gpt-3.5o` | `nvidia/nemotron-mini-4b-instruct` | Lightweight RP, fast responses | Very Fast | Low |
| `gemini-pro` | `nvidia/llama-3.3-nemotron-super-49b-v1.5` | Daily driver, low latency | Fast | Low |
| `gemini-turbo` | `meta/llama-3.3-70b-instruct` | Fast general purpose | Fast | Low-Medium |
| `gemini-turbo?` | `abacusai/dracarys-llama-3.1-70b-instruct` | Fine-tuned variant of above | Fast | Low-Medium |
| `mistral` | `mistralai/mistral-large-3-675b-instruct-2512` | Best quality, unfiltered | Very Slow | Low |
| `mistral-turbo` | `mistralai/mistral-medium-3.5-128b` | Fast fallback | Fast | Low |
| `mistral-pro` | `mistralai/mistral-small-4-119b-2603` | Lightweight scenes | Very Fast | Low |
| `mistral-fast` | `mistralai/ministral-14b-instruct-2512` | Fast, compact Mistral | Very Fast | Low |
| `mistral-nemo` | `mistralai/mistral-nemotron` | Casual/anime RP | Fast | Low |
| `claude-3-opus` | `openai/gpt-oss-120b` | Alternative to Chinese models | Medium | Low-Medium |
| `claude-3-sonnet` | `openai/gpt-oss-20b` | Fast, distinct voice | Fast | Low-Medium |
| `glm-5.2` | `z-ai/glm-5.2` | General purpose & Coding | Medium | Medium |
| `gpt-3.5-turbo` | `nvidia/nemotron-3-super-120b-a12b` | Lightweight tasks | Fast | Low |
| `gpt-3.5` | `qwen/qwen3.5-397b-a17b` | Qwen fallback | Medium | Medium |
| `google-light` | `google/gemma-4-31b-it` | Short scenes, fast | Fast | Low-Medium |
| `google-lighter` | `google/gemma-3n-e4b-it` | Mostly testing only | Very Fast | Low-Medium |
| `google-lightest` | `google/gemma-2-2b-it` | Testing only | Extremely fast | Low |
| `m2.7` | `minimaxai/minimax-m2.7` | Experimental | Medium | Unknown (to me) |
| `m3` | `minimaxai/minimax-m3` | Experimental | fast | Unknown (to me) |
| `step-3.5-flash` | `stepfun-ai/step-3.5-flash` | Chinese creative model | Fast | Medium |
| `step-3.7-flash` | `stepfun-ai/step-3.7-flash` | Chinese creative model | Fast | Medium |

### Filter Guide

| If your use case involves... | Avoid | Use instead |
|---|---|---|
| Dark themes, violence, mature content | `gpt-4o`, `gpt-4-flash`, `gpt-4-turbo` (They have high filters due to being based in China) | `mistral`, `gemini-pro`, `claude-3-opus` |
| Fast responses needed | `mistral` (675B) | `gemini-pro`, `mistral-turbo`, `gpt-3.5o` |
| Long context / memory | Anything under 30B | `gpt-4-turbo`, `mistral`, `gpt-4`, `glm-5.2` |
| Testing / very fast replies | — | `google-lightest`, `gpt-3.5o` |
| Coding / Long horizon work | — | `glm-5.2` |

### Fallback Chain

If your requested model fails, the proxy automatically tries:
1. Requested model
2. `mistralai/mistral-medium-3.5-128b`
3. `mistralai/mistral-small-4-119b-2603`
4. `nvidia/llama-3.3-nemotron-super-49b-v1.5`
5. `google/gemma-4-31b-it`

All fallbacks are non-Chinese-hosted to avoid filter interruption mid-scene. These can be changed, but i found that these four work best as fallbacks.

### Auth Guide
Jontte added auth middleware that wasn't present in the code he built upon. It uses an env var in your deployment. Use any secure string of 32+ characters, or generate one by hashing your NVAPI key. I recommend using an online hash tool or command to make a hash of your NVAPI key since the key is already complex as is, and a hash makes it more secure as it cannot be realistically reversed back to the NVAPI key. The first 32 characters of the hash are enough.
You can easily generate the hash with an online SHA-256 generator or any hash tool. Then make an env variable called "CLIENT_AUTH_KEY" and enter the first 32 characters of your hash into the variable (or any custom length over 16, or a custom key). Enter the hash into the API Key field in JanitorAI/SillyTavern.

### Proxy Setup Guide

Firstly head to https://build.nvidia.com/ and login/create an account. Then click your profile icon and navigate to "API keys". There you can generate an API key, and label it whatever you want. Save it immediately — you'll need to regenerate it if lost.

You *can* use basically any service that allows cloud deployments/VMs with a static IP, but I recommend Railway, Render, and Vercel (which I personally use). Possibly Oracle if you are comfortable with SSH and value the freedom it gives, but Railway is the easiest to setup.
You need to login to Railway with your Github. **Fork the repo before deploying. I cannot see your env vars, but forking ensures your deployment is fully isolated!** This prevents me (or anyone) from seeing your deployment in Railway's dashboard or through github. I also recommend making sure deployments aren't visible on the frontpage.
After you have made a deployment, you need to wait around 3 minutes for it to finish deploying. Then go into the "variables" tab, and create an env var with the name "NIM_API_KEY", and enter your NVAPI key into the variable. Next in your deployment go to the settings page, and there the networking section. Generate a public URL for your deployment. This is necessary to access it. Now your proxy is ready.

### Important Information
You can check the status of your proxy with the "/health" endpoint, and a list of models with "/v1/models". These endpoints intentionally do not require the auth, so clients can verify connectivity before configuring auth.
Your actual chat endpoint is in "/v1/chat/completions", and is the one you use in whatever platform
The client never sees your NVAPI key, which is why we don't use it as the auth, since the whole point of the auth configuration is so that your NVAPI key is not stored on your client.

### Optional Environment Variables

After deploying, you can set these in Railway's **Variables** tab (reasoning does not work reliably):

| Variable | Value | Effect |
|---|---|---|
| `SHOW_REASONING` | `true` | Shows model reasoning in `<thinking>` tags |
| `ENABLE_THINKING_MODE` | `true` | Sends thinking parameters to supported models |
| `DISCORD_WEBHOOK_URL` | Webhook URL | Alerts you when models fail validation |
| `SKIP_VALIDATION` | `true` | Disables startup model checks |


Set to `false` or remove to disable. Changes apply without redeploying.

### Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| "All models failed" error | NIM API key invalid or expired | Regenerate key at build.nvidia.com |
| Very slow responses | Using `mistral` (675B) or Chinese models during peak hours | Switch to `gemini-pro`, `mistral-turbo`, or `gpt-3.5o` |
| Filter interrupts RP | Using Chinese-hosted model for mature content | Use `mistral`, `gemini-pro`, or `claude-3-opus` |
| 404 on `/v1/chat/completions` | Auth mismatch | Verify `CLIENT_AUTH_KEY` matches between Railway and client |
| "Failed to fetch (unk)" / "A network error occurred" | JanitorAI cached old proxy config after changing URL or model | **Reload the page** — changes don't apply until refresh |


## Contributing

This is a project I am maintaining for my own use, but I'm happy if it helps others. If you spot a bug, want to suggest a model mapping, or have a small improvement, feel free to open an issue or PR. I will try to be as fast as possible as responses, but again, don’t count on it since I maintain this in my free time. I have many other projects i’m working on at the same time, but I'll do my best.

### What I'm open to
- Model mapping updates (NIM deprecates things constantly)
- Bug fixes
- Small feature additions that don't complicate the core flow
- Documentation improvements

### What I'm less likely to merge
- Major architectural changes (I want to keep this simple as to honour Jontte’s wishes)
- Features I don't personally use (harder for me to maintain)
- Anything that adds complexity without clear benefit

## Issues

Before opening an issue, check if it's already covered in the [Troubleshooting](#troubleshooting) section. If a model stopped working, it's probably deprecated by NVIDIA — check the [NIM catalog](https://build.nvidia.com/) first and please contact me on Discord (my username is provided below) to inform me.

When reporting bugs, include:
- Which model alias you were using
- Whether streaming was enabled
- The error message (or "All models failed" if that's what you got)
- Your deployment platform (Railway, Render, etc.)

## Contact
Need to reach out faster? Add me on Discord (Username - `Skywalker_1401`)

## Disclaimer

I am not a professional developer, this whole project was made by Jontte, so all credits to him! I’m going to maintain this for the time being. Jontte made this project via the help of AI tools and community guides as well.
