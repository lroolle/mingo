# Stream fixtures

Each `<name>.sse` is fed through `assemble` and compared with `<name>.want.json`.

- `deepseek-v4-flash-*.sse`: recorded live on 2026-09-13 with curl against
  api.deepseek.com (`stream_options.include_usage`, thinking on for the
  tool call, off for the text). Real wire bytes; ids are the server's.
- `llama-server-usage-only.sse`: synthetic, in the shape llama-server sends
  (finish_reason on its own chunk, a trailing usage-only chunk with
  `prompt_tokens_details.cached_tokens`), as verified on 2026-09-12.
- `cut-mid-answer.sse`: synthetic, a connection dropped mid-answer.
  `"error": true` says it must be rejected.

Add a fixture whenever a provider changes shape; the test fails loudly if
the assembler stops understanding one.
- `responses-gpt5-mini-*.sse`: the Responses dialect, recorded live on
  2026-09-13 with curl through openrouter.ai/api/v1/responses against
  `openai/gpt-5-mini` (`store: false`, `include: reasoning.encrypted_content`,
  `reasoning.summary: auto`). The tool-call stream ends in a
  `response.completed` whose output holds a `reasoning` item with
  `encrypted_content` and a `function_call` item; the text stream is the
  next round, made by replaying those items plus a `function_call_output`.
  Files with the `responses-` prefix go through `assembleResponses`.
