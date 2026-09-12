# spark.go

A general agentic AI agent in one Go file: 2,500 lines, standard library
only, six tools, a SOUL file for character, a sandbox that says what it
enforces. Talks to DeepSeek, OpenAI, or a llama-server on your own machine,
where a 4B model on a CPU fixed the bug below 8 runs out of 8.

    go build -o spark . && DEEPSEEK_API_KEY=... ./spark

```
spark: root=/work/app model=deepseek:deepseek-v4-flash think=high sandbox=workspace net=open session=20260909-010947 skills=1
> Add has a bug. Fix it, verify, and tell me in two sentences.
> exec: ls -la && find . -type f -name '*.go' | head -50
> read: add.go
> edit: add.go
> skill: gotest
> exec: go vet ./... && go test ./...
Fixed: Add returned a - b; it now returns a + b. Verified with go vet (clean) and go test (pass).
spark: wrote add.go
```

## What is in the file

| Piece | What it is |
| --- | --- |
| system prompt | short rules, the soul, the project's `AGENTS.md`, environment, skills catalog. `-show-prompt` prints it |
| SOUL | voice and working values in markdown, layered: built-in default, then `~/.spark/SOUL.md`, then `./.spark/SOUL.md`. Later layers extend, never erase |
| skills | markdown with `name` and `description` front matter under `.spark/skills/<name>/SKILL.md`, `.spark/skills/<name>.md`, or `~/.spark/skills/`. The catalog sits in the prompt; the body loads through the `skill` tool when a task matches |
| read | numbered lines, paged. Reading is what makes a file editable |
| write | creates files and parents. Overwriting needs a prior read of the current content |
| edit | replaces one unique block, with a trailing-whitespace-tolerant retry. Refuses if the file changed since it was read |
| exec | `/bin/sh -c` in the root: own process group, hard timeout kills the tree, secrets scrubbed from the environment, stdin closed, output capped head and tail |
| agent | a sub-agent with fresh context and the same tools minus `agent` and `monitor`, sandbox as strict or stricter, round-bounded. Only its report comes back; its transcript is saved beside the parent's |
| monitor | poll a command until a regexp matches or it exits 0, or a plain timer. Blocking, or in the background so it wakes the agent later with a new user turn, headless included |

After every turn that used tools the console prints a receipt: `wrote a.go, b.go`
or `no files written`. A model that says "fixed" while the receipt says nothing
was written is caught on the spot. Small models do this.

Sessions are append-only JSONL under `.spark/sessions/`, mode 0600, ignored by
a `.gitignore` the agent writes itself. `-resume last` continues. Compaction
runs on a context-overflow error or when the last prompt passed 85% of
`-context`; an unanswered user message survives it verbatim. `-max-requests`
caps model calls for the whole run, sub-agents and monitors included. Usage is
counted in tokens; no prices are hardcoded because prices move.

## Sandbox

Three modes, `-sandbox read-only | workspace | full` (`-yolo` = full).

| | read-only | workspace (default) | full |
| --- | --- | --- | --- |
| read | yes | yes | yes |
| write, edit | no | inside root | inside root |
| exec, read-only command | yes | yes | yes |
| exec, other | no | asks | yes |
| background monitor cmd | read-only only | read-only only | any |

What it is:

- a jail for `read`, `write`, `edit` and `cwd`: every open goes through
  `os.Root`, so a path or symlink that leaves the root fails at the kernel,
  not at a string check. `.spark/` is off limits except `skills/`.
  Credential-shaped files (`.env*`, `*.pem`, `id_*`, `.ssh/`, `.aws/`, ...)
  are refused anywhere;
- a policy for `exec`: a conservative read-only classifier (`ls`, `cat`,
  `grep`, `git status|log|diff`, `go build|vet`; no redirection, no
  substitution, no `sudo|xargs|eval`) runs at once. Anything else is refused,
  asked, or allowed by mode. Ask without a console is Deny: an unattended run
  never gets consent it did not have;
- a process fence: own process group, timeout, closed stdin, and an
  environment with anything named like key, token, secret or password
  removed, the agent's own `SPARK_*` included;
- optional network isolation: `-no-net` wraps commands in `unshare -rn` when
  Linux allows it. The banner says `net=isolated` or `net=open`.

What it is not: `exec` is not path-jailed. A shell command can name any path
the user can. The gate on it is the mode, the classifier, the confirmation and
the timeout, not string inspection of the command. Parsing commands for paths
is theater and this code does not pretend otherwise.

## Providers

| | deepseek (default) | openai | local |
| --- | --- | --- | --- |
| key | `DEEPSEEK_API_KEY` | `OPENAI_API_KEY` | none |
| endpoint | api.deepseek.com | api.openai.com/v1 | 127.0.0.1:8080/v1 |
| default model | `deepseek-v4-flash` | `gpt-6-astra` | whatever is loaded |
| thinking | `thinking.type` + `reasoning_effort` | `reasoning_effort` | `chat_template_kwargs.enable_thinking` |
| reasoning replay | every assistant message carries `reasoning_content` when tools are present, even empty, or the API answers 400 | stripped | stripped |

`SPARK_BASE_URL` overrides the endpoint; https is required except on loopback.
Streaming usage is read from whichever chunk carries it: DeepSeek moved it
from a usage-only chunk to the last content chunk in August 2026, llama-server
still sends the usage-only chunk, and both shapes are pinned in the tests.

## Local model

Any OpenAI-compatible server on loopback works. The one this was tuned on is
[Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) through the
vendor's llama.cpp fork, on a 10-core arm64 CPU with no GPU:

    git clone https://github.com/XHToken/llama.cpp.git && cd llama.cpp
    cmake -B build -DLLAMA_CURL=ON && cmake --build build -j --target llama-server
    ./build/bin/llama-server -hf XHToken/Spark-X2.5-4B-GGUF:Q4_K_M \
      -c 32768 -fa on --jinja --cache-reuse 256 --parallel 1 --reasoning-format deepseek
    ./spark -provider local

`make local` runs the same server from a downloaded GGUF and `make bench`
reproduces the table. Measured on the fix-a-bug task above, 4-bit quant,
80 tok/s prompt and 7 tok/s generation:

| | fixed the bug | wall time |
| --- | --- | --- |
| `-think high` (default) | 8 of 8 | 75 to 100 s, median 84 s |
| `-think off` | 2 of 6 | 29 to 85 s |
| `--spec-type ngram-simple` on the server | 4 of 4 | no change |

The thinking-off failures are the interesting ones: the model read the file,
ran `go vet`, then wrote "Changed the return to a + b" without ever calling
edit. It finished fastest because it skipped the work. The receipt line
catches it; keep thinking on for a 4B model.

What makes it usable on a CPU is the prompt cache. The system prompt and tool
schemas are a stable prefix, so llama-server reprocesses only the new tokens
each round: 78 to 90% of prompt tokens came from cache across these runs, and
the first round of a fresh server is the only one that pays for the full
prefix (about 1,400 tokens, 18 s here). The tool schemas are the bulk of that
prefix, which is why they are written tight.

## When to skip it

- You want the host protected from the model, not the model from surprises.
  Run tools in a container instead; this sandbox asks a human.
- You need Windows. exec relies on process groups.
- You want a framework with plugins, a TUI, and MCP. This is one file on
  purpose; fork it and change it.

## Flags

```
-provider    deepseek | openai | local     -sandbox      read-only | workspace | full
-model       model id                      -yolo         full and never ask
-think       off | low | high | max        -no-net       isolate exec from the network
-max-tokens  completion cap                -p            one prompt, headless
-max-requests model calls per run          -resume       last | session id
-context     window for compaction         -max-rounds   tool rounds per turn
-cwd         root                          -quiet        no reasoning or tool chatter
-show-prompt print the system prompt
```

Slash commands: `/new /compact /cost /monitors /cancel ID /skills /soul
/sandbox [mode] /quit`. Ctrl-C cancels the running turn.

`go install github.com/lroolle/spark.go@latest` works but names the binary
`spark.go`; `go build -o spark .` or a release binary is nicer.

## Verify

    go test ./...

The suite drives the whole loop against a fake streaming provider: tool
rounds, reasoning replay, overflow compaction, 429 retry, the request ceiling,
truncated tool calls, sub-agent bounds, round budget, headless deny, the write
receipt, stale-read refusal, monitor firing and retiring, the jail, the
classifier, the process-tree kill. What it cannot prove is a live wire. Those
were checked by hand: DeepSeek on 2026-09-09 and Spark-X2.5-4B through
llama-server on 2026-09-12.

## Layout of spark.go

```
config     flags, env, providers
wire       messages, streaming client, tool-call assembly, usage
soul       SOUL layers, skills, system prompt
sandbox    os.Root jail, secret gate, exec policy and classifier
tools      read, write, edit, exec, skill
scheduler  monitors that wake the agent
agent      the loop, compaction, sub-agent, sessions
console    REPL, slash commands, confirmations, signals, receipts
```

Only `wire` knows HTTP. Only `sandbox` decides permission. Only `console`
talks to a terminal. Tools never print; they return text.

## License

MIT
