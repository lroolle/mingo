# min.go

A small agent runtime in one Go file.

Small enough to understand. Built to do real work.

Minimize machinery, not capability: min is a tool loop over any
OpenAI-compatible endpoint (a llama-server on this machine, DeepSeek,
OpenRouter, OpenAI's Responses API), a sandbox that says exactly what it
enforces, and a session log that survives interruption. It is built so
that a 4B model on a laptop CPU can deliver, not demo: the harness checks
what the model claims, refuses what it cannot verify, closes every loop
the model leaves open, and keeps the transcript small.

    go build -o min . && DEEPSEEK_API_KEY=... ./min

```
min: root=/work/app model=deepseek:deepseek-v4-flash think=high sandbox=workspace fence=bwrap context=128000 session=20260913-081502-9c1a2f skills=1
> Add has a bug. Fix it, verify, and tell me in two sentences.
> exec: ls -la && find . -type f -name '*.go' | head -50
> read: add.go
> edit: add.go
> exec: go vet ./... && go test ./...
Fixed: Add returned a - b; it now returns a + b. Verified with go vet (clean) and go test (pass).
min: wrote add.go
```

Standard library only, POSIX, one core file (`min.go`), thick tests
around it.

## What is in the file

| Piece | What it is |
| --- | --- |
| system prompt | short rules, the soul, the project's `AGENTS.md`, environment, skills catalog. A stable prefix, so a local server serves most of it from cache. `-show-prompt` prints it |
| SOUL | voice and working values in markdown, layered: built-in default, then `~/.min/SOUL.md`, then `./.min/SOUL.md`. Later layers extend, never erase |
| skills | markdown with `name` and `description` front matter under `.min/skills/<name>/SKILL.md`, `.min/skills/<name>.md`, or `~/.min/skills/`. The catalog sits in the prompt; the body loads through the `skill` tool when a task matches |
| read | numbered lines, paged. Reading is what makes a file editable |
| write | creates files and parents, atomically: a private temp file (random name, created exclusively) and a rename. Overwriting needs a prior read of the current content |
| edit | replaces one unique block, with a trailing-whitespace-tolerant retry. Refuses if the file changed since it was read |
| exec | `/bin/sh -c` inside the fence: own process group, hard timeout kills the tree, secrets scrubbed from the environment, stdin closed, output bounded in memory while it runs |
| agent | a sub-agent with fresh context and the same tools minus `agent`, sandbox as strict or stricter, round-bounded. Only its report comes back; its transcript is saved beside the parent's |

After every turn that used tools the console prints a receipt: `wrote a.go, b.go`
or `no files written`. A model that says "fixed" while the receipt says nothing
was written is caught on the spot. Small models do this.

## What the loop guarantees

These are properties of the program, not of the model's manners. Each has
a test that tries to break it.

- **No tool runs from an incomplete reply.** A stream that ends without a
  terminal event is rejected; a reply cut by the token cap has its calls
  answered "not executed" and the model told why.
- **A batch is judged whole before any of it runs.** Every call needs a
  name and an id, no two calls share an id; a batch that fails this is an
  error, not retried. Cancellation is checked at the moment of dispatch,
  and again by `write` and `edit` right before they commit.
- **Budgets are terminal.** At `-max-rounds` the open batch is closed
  unexecuted, the model gets one tool-less request for a final report, and
  the turn ends with exit 3. `-max-requests` counts every HTTP attempt by
  every agent in the process, sub-agents included.
- **Every batch is closed.** A cancelled turn answers the calls it did not
  run, so the next message never earns a 400 for a missing tool result.
- **A repeated failing call is not run again.** The same name and arguments
  right after an error get "change the approach" instead of a round.
- **Missing arguments are not empty ones.** `write` without `content` or
  `edit` without `new` is refused; it cannot empty a file.
- **Compaction is a transaction and it shrinks, checked.** The summary is
  obtained first, refused if the token cap cut it, if it is empty, or if
  it is not smaller than what it replaces. The new log is staged and
  synced in full before the session switches to it and before memory
  changes; a failure anywhere leaves the old transcript in memory and the
  old file as the session. If the summary request itself overflows, the
  oldest half is dropped and it tries again. An unanswered user message is
  carried over verbatim, never summarized.
- **A retry never prints twice.** A failed request is retried only while
  nothing has reached the console.

## Sandbox

Three modes, `-sandbox read-only | workspace | full` (`-yolo` = full).

| | read-only | workspace (default) | full |
| --- | --- | --- | --- |
| read | yes | yes | yes |
| write, edit | no | inside root | inside root |
| exec, allowlisted command | yes | yes | yes |
| exec, other | no | asks | yes |

Three layers, each honest about what it covers:

- **the jail**, for `read`, `write`, `edit` and `cwd`: every open goes
  through `os.Root`, so a path or symlink that leaves the root fails in
  the kernel, not in a string check. `.min/` is off limits except
  `skills/`. Credential-shaped files (`.env*`, `*.pem`, `id_*`, `.ssh/`,
  `.aws/`, ...) are refused anywhere;
- **the policy**, for `exec`: an allowlist of read-only tools (`ls`, `cat`,
  `grep`, `find`, `git status|log|diff`, `go vet|build`, ...) runs without
  asking. The command is split the way `sh` splits it (quotes and
  backslashes understood), so `'-delete'` is `-delete`; writing flags are
  denied attached or clustered too (`sort -oout`, `sort -ro out`); a second
  operand to `uniq`, an operand to `date`, an expansion (`$X`, `${X}`), a
  glob in a flag, any redirection except to stderr or `/dev/null`, any
  substitution, `sudo|xargs|eval|exec`, a path-qualified verb, a `PATH=`
  or `LD_*` assignment: each fails the whole command. Anything else is
  refused, asked, or allowed by mode. Ask without a console is Deny: an
  unattended run never gets consent it did not have. The console only
  collects the answer; the sandbox that asked applies it, so "always" at a
  sub-agent's prompt never widens the parent's authority. This is a
  convenience gate, not a security boundary;
- **the fence**, the OS boundary `exec` runs inside. It is detected once by
  running `/bin/true` through the exact arguments `exec` will use, and the
  banner names it.

| fence | where | what exec gets |
| --- | --- | --- |
| `bwrap` | Linux with bubblewrap | the whole filesystem read-only, the root bound writable (read-only in read-only mode), a fresh tmpfs on `/tmp`, build caches writable (`~/.cache`, `~/go/pkg`, `~/.cargo/registry`, `$GOCACHE`, `$GOMODCACHE`; never a whole `$GOPATH` or `$CARGO_HOME`), `~/.ssh ~/.aws ~/.gnupg ~/.kube ~/.docker ~/.config/gh ~/.netrc ~/.git-credentials ~/.npmrc ~/.pypirc` masked, own pid namespace, network shared unless `-no-net` |
| `seatbelt` | macOS, `sandbox-exec` | writes denied except the root (not in read-only mode), tmp and the same caches; the same credential paths unreadable; network denied with `-no-net` |
| `none` | anything else | process group, timeout, scrubbed environment, closed stdin. `exec` can then name any path the user can; the policy is the only gate |

With a fence, "read-only" is a kernel property for `exec` too: the tests
run `rm`, `echo >` and `touch` through it and check the disk afterwards. CI
installs bubblewrap on Linux and uses sandbox-exec on macOS, and a test
fails the job if either machine ends up with `fence=none`, so the fence
cannot silently degrade into a green build.

`-no-net` and `-sandbox read-only` ask the OS for a boundary. On a machine
with no fence they fail at startup instead of running weaker than asked;
`-unfenced` accepts the policy gate alone, and the banner says so.

What the sandbox is not: a `.env` inside the project is readable by `cat`
through `exec`, as it is by any tool you run yourself; the file tools refuse
it, the fence masks the well-known credential paths under `$HOME`. Parsing
commands for paths is theater and this code does not pretend otherwise.

## Sessions

Every message is appended to `.min/sessions/<id>.jsonl` and fsynced as it
happens, mode 0600, ignored by a `.gitignore` the runtime writes itself. The
id is a UTC timestamp plus a random suffix, created with `O_EXCL`, and the
file is locked for as long as it is open, so two agents in one directory
never share a log and a second process resuming the same session is told
so. The first line is a header; every other line is one message, readable
with `jq`.

- `-resume last` or `-resume <id>` continues. `last` is the session most
  recently written to. A batch whose results never arrived (the run was
  killed between a call and its result) is closed with "outcome unknown,
  check before redoing", and that repair is written to the log, so a
  second resume finds nothing to repair. A torn last line is cut at the
  last good boundary on disk, and a complete last record that lost its
  newline gets one, before anything is appended; corruption anywhere else
  is an error and the file is left alone.
- `-sessions` lists them: id, message count, size, first user message.
- `-show last` or `-show <id>` replays a transcript through the console
  without a model: the same text, tool lines and results the run showed.
- Compaction switches to a new file that starts with the summary; the old
  file stays. `/new` does the same without a summary.
- A log that cannot be written fails the turn before the model is called.
  A step that is not logged is a step that cannot be resumed.

## Headless

`-p "prompt"` runs one turn and exits. The answer goes to stdout, everything
else to stderr, and the exit code is the outcome:

```
0 done   1 failed   2 usage   3 budget exhausted   4 reply truncated   130 cancelled
```

`-json` prints one object on stdout instead of the answer:

```json
{"outcome":"done","answer":"Fixed ...","written":["add.go"],"session":"20260913-081502-9c1a2f","requests":4,"usage":{"prompt":5200,"cached":4100,"completion":310}}
```

A job runner reads `outcome` and `written`, not the prose. `done` means
the model answered within its budgets; it is the runtime's outcome, not a
verification of the work. Verification is the job of whatever runs min:
the graders under `eval/` are one example. `-quiet` silences reasoning
and tool lines on stderr; notes and the receipt stay.

## Providers

| | deepseek (default) | openai | openrouter | local |
| --- | --- | --- | --- | --- |
| key | `DEEPSEEK_API_KEY` | `OPENAI_API_KEY` | `OPENROUTER_API_KEY` | none |
| endpoint | api.deepseek.com | api.openai.com/v1/responses | openrouter.ai/api/v1 | 127.0.0.1:8080/v1 |
| wire | chat completions | Responses API | chat completions | chat completions |
| default model | `deepseek-v4-flash` | `gpt-6-astra` | `deepseek/deepseek-v4-flash` | whatever is loaded |
| context | 128000 | 922000 | 128000 | read from the server's `/props` |
| thinking | `thinking.type` + `reasoning_effort` | `reasoning.effort` (low, high, max pass through) | `reasoning.effort` / `reasoning.enabled` | `chat_template_kwargs.enable_thinking` |
| reasoning replay | every assistant message carries `reasoning_content` when tools are present, even empty, or the API answers 400 | the encrypted reasoning items come back and are replayed verbatim; nothing is stored server-side | stripped | stripped |
| completion cap | the API's | the API's | the API's | 4096 (`-max-tokens` overrides) |

The OpenAI provider speaks the Responses API: instructions at the top
level, typed `function_call` and `function_call_output` items, `store`
off, `include: reasoning.encrypted_content` so the model's own reasoning
travels with the transcript, and typed stream events folded by the
terminal one. OpenRouter is one key for a few hundred models; `-model`
picks any id from `openrouter.ai/api/v1/models`, reasoning streams back in
the delta's `reasoning` field. `MIN_BASE_URL` overrides any endpoint; https
is required except on loopback, and loopback means `localhost` or a loopback
IP literal, not a hostname that starts with `127.`.

Streaming usage is read from whichever chunk carries it: DeepSeek moved it
from a usage-only chunk to the last content chunk in August 2026,
llama-server still sends the usage-only chunk. Both shapes, and both
Responses streams, are pinned by recorded streams under `testdata/streams/`.

## Local model

Any OpenAI-compatible server on loopback works; min reads the context
window from `/props` so compaction fits the model actually loaded, and caps
a reply at 4096 tokens because a llama-server has no cap of its own and a
4B model that starts thinking in circles will otherwise fill its window in
one request (measured: 15,864 tokens in 45 minutes at 6 tok/s). The one
this was tuned on is [Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B)
through the vendor's llama.cpp fork, on a 10-core arm64 CPU with no GPU:

    git clone https://github.com/XHToken/llama.cpp.git && cd llama.cpp
    cmake -B build -DLLAMA_CURL=ON && cmake --build build -j --target llama-server
    ./build/bin/llama-server -hf XHToken/Spark-X2.5-4B-GGUF:Q4_K_M \
      -c 32768 -fa on --jinja --cache-reuse 256 --parallel 1 --reasoning-format deepseek
    ./min -provider local

`make local` runs the same server from a downloaded GGUF and `make eval`
runs the live tasks under `eval/`.

LOCAL_EVAL_TABLE

What the harness does for a small model, beyond the guarantees above: the
system prompt and tool schemas are a stable prefix, so the server reprocesses
only the new tokens each round; double-encoded tool arguments are unwrapped;
a call that repeats a failure is answered without a round; the receipt tells
the person what actually changed.

## When to skip it

- You need Windows. exec relies on process groups.
- You want background monitors, plugins, a TUI, or MCP. This is one file on
  purpose; fork it and change it. A shell loop with a timeout through `exec`
  covers "wait until the build passes".

## Flags

```
-provider     deepseek | openai | openrouter | local    -sandbox      read-only | workspace | full
-model        model id                                  -yolo         full and never ask
-think        off | low | high | max                    -no-net       cut exec off the network (needs a fence)
-max-tokens   completion cap (local: 4096)              -unfenced     accept no fence for read-only or no-net
-max-requests model calls per run                       -p            one prompt, headless
-max-rounds   tool rounds per turn                      -json         one JSON result on stdout
-context      window for compaction                     -resume       last | session id
-cwd          root                                      -sessions     list sessions
-quiet        no reasoning or tool lines                -show         last | id: replay a transcript
-no-skills    do not load skills                        -show-prompt  print the system prompt
-version
```

Slash commands: `/new /compact /cost /skills /soul /sandbox [mode] /quit`.
Ctrl-C cancels the running turn.

`go install github.com/lroolle/min.go@latest` works but names the binary
`min.go`; `go build -o min .` or a release binary is nicer.

## Verify

    make race      # go test -race: 85 tests
    make bench     # go benchmarks of the paths the loop runs every round
    make eval      # live tasks against a real model, spends requests

The suite drives the whole loop against a fake streaming provider and the
tools against a real sandbox: tool rounds, reasoning replay, recorded
DeepSeek, llama-server and Responses streams, cut streams, malformed
batches, overflow compaction and its failure, refusal and shrink paths,
429 retry, the request ceiling, truncated calls and truncated answers, the
terminal round budget, cancel mid-batch and cancel at dispatch, sub-agent
bounds, budgets and consent, headless deny, the write receipt, stale-read
refusal, required arguments, atomic writes and their private temp files,
the jail, the allowlist and the holes two reviews found in it, the output
cap, the process-tree kill, the fence's argv and, where a fence exists,
its guarantees on disk; unique and exclusive session logs, resume repair
on disk across resume cycles, the last-active rule, staged compaction
commits, the exit codes and the JSON result end to end.

What it cannot prove is a live wire. The five tasks under `eval/` were run
on 2026-09-13 against DeepSeek (5 of 5), gpt-6-astra over the Responses
API and deepseek-v4-flash over OpenRouter (fix-add, both passed); their
graders call the code, mutate it, and compare the tree, so a comment that
says "fixed" or a test with no assertions does not pass. See
`eval/README.md`.

## Layout of min.go

```
config     flags, env, providers, outcomes
wire       messages, streaming client for both dialects, tool-call assembly, usage
soul       SOUL layers, skills, system prompt
sandbox    os.Root jail, secret gate, exec policy, the fence
tools      read, write, edit, exec, agent, skill
agent      the loop, budgets, compaction, sub-agent
session    append-only JSONL, repair on resume, listing, replay
console    REPL, slash commands, confirmations, signals, receipts
```

Only `wire` knows HTTP. Only `sandbox` decides permission. Only `console`
talks to a terminal. Tools never print; they return text.

## License

MIT
