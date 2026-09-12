package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// --- wire

func sse(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestAssembleToolCallsAcrossChunks(t *testing.T) {
	stream := sse(
		`{"choices":[{"delta":{"reasoning_content":"think"}}]}`,
		`{"choices":[{"delta":{"content":"hi "}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{\"pa"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a\"}"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"exec","arguments":"{}"}}]}}]}`,
		// usage riding the last content chunk, the shape DeepSeek switched to on 2026-08-28
		`{"choices":[{"delta":{"content":"there"},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_cache_hit_tokens":8}}`,
	)
	r, err := assemble(strings.NewReader(stream), nopSink{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Content != "hi there" || r.Reasoning != "think" || r.Finish != "tool_calls" {
		t.Fatalf("bad reply %+v", r)
	}
	if len(r.ToolCalls) != 2 || r.ToolCalls[0].ID != "c1" || r.ToolCalls[0].Function.Arguments != `{"path":"a"}` || r.ToolCalls[1].Function.Name != "exec" {
		t.Fatalf("bad tool calls %+v", r.ToolCalls)
	}
	if r.Usage != (Usage{Prompt: 10, Cached: 8, Completion: 4}) {
		t.Fatalf("bad usage %+v", r.Usage)
	}
}

func TestAssembleUsageOnlyChunkAndOpenAICached(t *testing.T) {
	stream := sse(
		`{"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":3}}}`,
	)
	r, err := assemble(strings.NewReader(stream), nopSink{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Usage != (Usage{Prompt: 5, Cached: 3, Completion: 1}) {
		t.Fatalf("bad usage %+v", r.Usage)
	}
}

func TestAssembleStreamError(t *testing.T) {
	_, err := assemble(strings.NewReader(sse(`{"error":{"message":"boom"}}`)), nopSink{})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("want stream error, got %v", err)
	}
}

func TestRequestReasoningReplayPerProvider(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}}
	tools := []ToolSpec{{Name: "x", Parameters: schema(nil)}}

	ds := &Client{cfg: &Config{Provider: providers["deepseek"], Model: "m", Think: "high"}}
	raw, _ := json.Marshal(ds.request(msgs, tools))
	if !strings.Contains(string(raw), `"reasoning_content":""`) {
		t.Fatalf("deepseek must replay reasoning_content even when empty: %s", raw)
	}
	if !strings.Contains(string(raw), `"thinking":{"type":"enabled"}`) || !strings.Contains(string(raw), `"reasoning_effort":"high"`) {
		t.Fatalf("deepseek thinking fields missing: %s", raw)
	}
	raw, _ = json.Marshal(ds.request(msgs, nil))
	if strings.Contains(string(raw), "reasoning_content") {
		t.Fatalf("without tools reasoning_content is noise: %s", raw)
	}

	oa := &Client{cfg: &Config{Provider: providers["openai"], Model: "m", Think: "max", MaxTokens: 9}}
	r := "r"
	msgs[1].ReasoningContent = &r
	raw, _ = json.Marshal(oa.request(msgs, tools))
	if strings.Contains(string(raw), "reasoning_content") || strings.Contains(string(raw), `"thinking"`) {
		t.Fatalf("openai must not see deepseek fields: %s", raw)
	}
	if !strings.Contains(string(raw), `"reasoning_effort":"high"`) || !strings.Contains(string(raw), `"max_completion_tokens":9`) {
		t.Fatalf("openai fields wrong: %s", raw)
	}
}

func TestOverflowDetection(t *testing.T) {
	if !isOverflow(&apiError{400, "This model's maximum context length is 131072 tokens"}) {
		t.Fatal("overflow not detected")
	}
	if isOverflow(&apiError{429, "rate limit"}) || isOverflow(&apiError{400, "bad json"}) {
		t.Fatal("false overflow")
	}
}

// --- soul + skills

func TestParseSkill(t *testing.T) {
	raw := []byte("---\r\nname: go-test\r\ndescription: \"Run go tests\"\r\n---\r\nbody here\r\n")
	s, body, ok := parseSkill("/x/go-test/SKILL.md", raw)
	if !ok || s.Name != "go-test" || s.Description != "Run go tests" || strings.TrimSpace(body) != "body here" {
		t.Fatalf("crlf skill: %+v ok=%v body=%q", s, ok, body)
	}
	if _, _, ok := parseSkill("/x/a.md", []byte("\n---\nname: a\ndescription: d\n---\n")); ok {
		t.Fatal("fence must be on line 1")
	}
	if _, _, ok := parseSkill("/x/a.md", []byte("---\nname: a\n---\n")); ok {
		t.Fatal("description is required")
	}
	s, _, _ = parseSkill("/x/dirname/SKILL.md", []byte("---\ndescription: d\n---\n"))
	if s.Name != "dirname" {
		t.Fatalf("name should fall back to the directory, got %q", s.Name)
	}
}

func TestLoadSkillsAndSoulLayers(t *testing.T) {
	root := t.TempDir()
	home := t.TempDir()
	must(t, os.MkdirAll(filepath.Join(root, ".spark", "skills", "deploy"), 0o755))
	must(t, os.WriteFile(filepath.Join(root, ".spark", "skills", "deploy", "SKILL.md"), []byte("---\nname: deploy\ndescription: ship it\n---\nsteps\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(root, ".spark", "skills", "deploy", "notes.md"), []byte("---\nname: notes\ndescription: not a skill\n---\n"), 0o644))
	must(t, os.MkdirAll(filepath.Join(home, "skills"), 0o755))
	must(t, os.WriteFile(filepath.Join(home, "skills", "review.md"), []byte("---\ndescription: review code\n---\nlook\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(home, "skills", "deploy.md"), []byte("---\ndescription: shadowed by project\n---\n"), 0o644))
	skills := loadSkills(filepath.Join(root, ".spark", "skills"), filepath.Join(home, "skills"))
	if len(skills) != 2 || skills[0].Name != "deploy" || skills[0].Description != "ship it" || skills[1].Name != "review" {
		t.Fatalf("skills: %+v", skills)
	}
	must(t, os.WriteFile(filepath.Join(home, "SOUL.md"), []byte("user layer"), 0o644))
	must(t, os.WriteFile(filepath.Join(root, ".spark", "SOUL.md"), []byte("project layer"), 0o644))
	soul := readSoul(home, root)
	if !strings.HasPrefix(soul, defaultSoul) || !strings.Contains(soul, "user layer\n\nproject layer") {
		t.Fatalf("soul layers wrong:\n%s", soul)
	}
	cfg := &Config{Root: root, Mode: ModeWorkspace}
	sys := buildSystemPrompt(cfg, soul, skills, false)
	for _, want := range []string{"# Soul", "- deploy: ship it", "- review: review code", "sandbox: workspace"} {
		if !strings.Contains(sys, want) {
			t.Fatalf("system prompt lacks %q", want)
		}
	}
}

// --- sandbox

func newTestSandbox(t *testing.T, mode Mode) (*Sandbox, string) {
	root, err := filepath.EvalSymlinks(t.TempDir())
	must(t, err)
	sb, err := newSandbox(&Config{Root: root, Mode: mode})
	must(t, err)
	t.Cleanup(func() { sb.fs.Close() })
	return sb, root
}

func TestResolveJail(t *testing.T) {
	sb, root := newTestSandbox(t, ModeWorkspace)
	outside := t.TempDir()
	must(t, os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("x"), 0o644))
	must(t, os.Symlink(filepath.Join(outside, "secret.txt"), filepath.Join(root, "link")))
	must(t, os.MkdirAll(filepath.Join(root, ".spark", "skills", "s"), 0o755))
	must(t, os.MkdirAll(filepath.Join(root, ".spark", "sessions"), 0o755))

	ok := []string{"a.go", "sub/dir/new.txt", root + "/b.go", ".spark/skills/s/SKILL.md", "./x/../y.txt"}
	for _, p := range ok {
		if _, err := sb.Resolve(p); err != nil {
			t.Errorf("%s should resolve: %v", p, err)
		}
	}
	bad := map[string]string{
		"../escape.txt":           "outside",
		"/etc/passwd":             "outside",
		"link":                    "outside",
		".spark/sessions/x.jsonl": "off limits",
		".spark/SOUL.md":          "off limits",
		".env":                    "credential",
		".env.local":              "credential",
		"deploy/server.pem":       "credential",
		"keys/id_ed25519":         "credential",
		".ssh/config":             "credential",
		"config/credentials.json": "credential",
		"":                        "empty",
		filepath.Join(root, "..", filepath.Base(root)+"x", "f"): "outside",
	}
	for p, want := range bad {
		_, err := sb.Resolve(p)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%q: want error containing %q, got %v", p, want, err)
		}
	}
}

func TestReadOnlyClassifier(t *testing.T) {
	yes := []string{
		"ls -la", "cat a.go | grep foo", "git status && git diff --stat", "go build ./... ; go vet ./...",
		"find . -name '*.go' | wc -l", "FOO=1 grep -r x .", "/bin/ls", "git log --oneline -5", "docker ps",
	}
	no := []string{
		"", "rm -rf x", "cat a > b", "git commit -m x", "go test ./...", "ls $(echo x)", "echo `id`",
		"find . | xargs rm", "sudo ls", "git push", "npm install", "python3 x.py", "go run .", "sed -i s/a/b/ f",
		"ls; make", "git status | tee out", "eval ls", "exec ls",
	}
	for _, c := range yes {
		if !readOnlyCommand(c) {
			t.Errorf("%q should be read-only", c)
		}
	}
	for _, c := range no {
		if readOnlyCommand(c) {
			t.Errorf("%q should NOT be read-only", c)
		}
	}
}

func TestExecPolicy(t *testing.T) {
	sb, _ := newTestSandbox(t, ModeReadOnly)
	if err := sb.Exec("ls"); err != nil {
		t.Fatal(err)
	}
	if err := sb.Exec("rm x"); err == nil {
		t.Fatal("read-only must refuse")
	}
	sb.Mode = ModeWorkspace
	if err := sb.Exec("rm x"); err == nil || !strings.Contains(err.Error(), "nobody is at the console") {
		t.Fatalf("headless Ask must deny, got %v", err)
	}
	asked := 0
	sb.Confirm = func(string) bool { asked++; return asked > 1 }
	if err := sb.Exec("rm x"); err == nil {
		t.Fatal("declined must refuse")
	}
	if err := sb.Exec("rm x"); err != nil {
		t.Fatal("approved must run")
	}
	sb.Always("exec")
	if err := sb.Exec("rm y"); err != nil || asked != 2 {
		t.Fatalf("always must skip the prompt: err=%v asked=%d", err, asked)
	}
	sb.Mode = ModeFull
	sb.Confirm = func(string) bool { t.Fatal("full mode never asks"); return false }
	if err := sb.Exec("rm z"); err != nil {
		t.Fatal(err)
	}
	child := sb.child(true)
	if child.Mode != ModeReadOnly {
		t.Fatal("readonly child must be read-only")
	}
	if sb.child(false).Mode != ModeFull {
		t.Fatal("child inherits the parent's mode")
	}
}

func TestRunScrubsSecretsAndKillsTree(t *testing.T) {
	t.Setenv("MY_API_KEY", "hunter2")
	t.Setenv("SPARK_MODEL", "x")
	sb, root := newTestSandbox(t, ModeFull)
	out, code, err := sb.Run(context.Background(), "env; echo exit-test; exit 3", root, 5*time.Second)
	if err != nil || code != 3 {
		t.Fatalf("code=%d err=%v", code, err)
	}
	if strings.Contains(out, "hunter2") || strings.Contains(out, "SPARK_MODEL") || !strings.Contains(out, "SPARK=1") {
		t.Fatalf("env not scrubbed:\n%s", out)
	}
	start := time.Now()
	_, code, err = sb.Run(context.Background(), "sh -c 'sleep 30' & sleep 30", root, 300*time.Millisecond)
	if err != nil || code != 124 {
		t.Fatalf("timeout: code=%d err=%v", code, err)
	}
	if time.Since(start) > 5*time.Second {
		t.Fatal("timeout did not kill the process tree promptly")
	}
	if _, _, err := sb.Run(context.Background(), "cat", root, time.Second); err != nil {
		t.Fatalf("closed stdin must not block: %v", err)
	}
}

// --- tools

func TestReplaceUnique(t *testing.T) {
	text := "a\nfoo(1)\nb\nfoo(2)\n"
	if _, _, err := replaceUnique(text, "foo(", "bar("); err == nil || !strings.Contains(err.Error(), "2 times") {
		t.Fatalf("ambiguous must fail: %v", err)
	}
	out, line, err := replaceUnique(text, "foo(2)", "bar(2)")
	if err != nil || line != 4 || out != "a\nfoo(1)\nb\nbar(2)\n" {
		t.Fatalf("exact: %q line=%d err=%v", out, line, err)
	}
	out, line, err = replaceUnique("x  \ny\t\nz\n", "x\ny\n", "w\n")
	if err != nil || line != 1 || out != "w\nz\n" {
		t.Fatalf("whitespace-tolerant: %q line=%d err=%v", out, line, err)
	}
	out, _, err = replaceUnique("a\nb\nc\n", "b\n", "")
	if err != nil || out != "a\nc\n" {
		t.Fatalf("delete: %q err=%v", out, err)
	}
	if _, _, err := replaceUnique("a\n", "zzz", "y"); err == nil {
		t.Fatal("missing must fail")
	}
}

func TestReadWriteEditRules(t *testing.T) {
	sb, root := newTestSandbox(t, ModeWorkspace)
	tb := newToolbox(sb)
	tb.add(tb.readTool())
	tb.add(tb.writeTool())
	tb.add(tb.editTool())
	call := func(name, args string) string {
		return tb.Call(context.Background(), ToolCall{Function: FuncCall{Name: name, Arguments: args}}, tb.Specs())
	}
	if out := call("edit", `{"path":"a.txt","old":"x","new":"y"}`); !strings.Contains(out, "does not exist") {
		t.Fatalf("edit of a missing file: %s", out)
	}
	must(t, os.WriteFile(filepath.Join(root, "a.txt"), []byte("x\n"), 0o644))
	if out := call("edit", `{"path":"a.txt","old":"x","new":"y"}`); !strings.Contains(out, "not read") {
		t.Fatalf("edit before read must refuse: %s", out)
	}
	if out := call("write", `{"path":"deep/a.txt","content":"one\ntwo\n"}`); !strings.HasPrefix(out, "wrote") {
		t.Fatalf("write: %s", out)
	}
	if out := call("write", `{"path":"deep/a.txt","content":"again"}`); !strings.HasPrefix(out, "wrote") {
		t.Fatalf("write after write (counts as read): %s", out)
	}
	// a hand edit between read and edit must be caught, and a fresh read must clear it
	must(t, os.WriteFile(filepath.Join(root, "deep", "a.txt"), []byte("changed by hand"), 0o644))
	if out := call("edit", `{"path":"deep/a.txt","old":"again","new":"x"}`); !strings.Contains(out, "changed since you read it") {
		t.Fatalf("stale read must refuse: %s", out)
	}
	if out := call("write", `{"path":"deep/a.txt","content":"x"}`); !strings.Contains(out, "changed since you read it") {
		t.Fatalf("stale overwrite must refuse: %s", out)
	}
	call("read", `{"path":"deep/a.txt"}`)
	if out := call("edit", `{"path":"deep/a.txt","old":"by hand","new":"by edit"}`); !strings.Contains(out, "edited") {
		t.Fatalf("edit after re-read: %s", out)
	}
	must(t, os.WriteFile(filepath.Join(root, "b.txt"), []byte("l1\nl2\nl3\n"), 0o644))
	if out := call("write", `{"path":"b.txt","content":"x"}`); !strings.Contains(out, "was not read") {
		t.Fatalf("overwrite unread must refuse: %s", out)
	}
	out := call("read", `{"path":"b.txt","offset":2,"limit":1}`)
	if out != "     2| l2\n... truncated; continue with offset 3\n" {
		t.Fatalf("read paging: %q", out)
	}
	if out := call("edit", `{"path":"b.txt","old":"l2","new":"L2"}`); !strings.Contains(out, "line 2") {
		t.Fatalf("edit: %s", out)
	}
	got, _ := os.ReadFile(filepath.Join(root, "b.txt"))
	if string(got) != "l1\nL2\nl3\n" {
		t.Fatalf("file after edit: %q", got)
	}
	sb.Mode = ModeReadOnly
	if out := call("edit", `{"path":"b.txt","old":"l1","new":"x"}`); !strings.Contains(out, "read-only") {
		t.Fatalf("read-only must refuse edits: %s", out)
	}
	if out := call("read", `{"path":"../etc"}`); !strings.Contains(out, "outside") {
		t.Fatalf("jail: %s", out)
	}
	if out := call("nope", `{}`); !strings.Contains(out, "no tool") {
		t.Fatalf("unknown tool: %s", out)
	}
}

func TestClip(t *testing.T) {
	s := strings.Repeat("a", 50) + strings.Repeat("b", 50)
	out := clip(s, 30)
	if !strings.HasPrefix(out, "aaaaaaaaaa\n") || !strings.HasSuffix(out, strings.Repeat("b", 20)) || !strings.Contains(out, "[70 bytes cut]") {
		t.Fatalf("clip: %q", out)
	}
	if clip("short", 30) != "short" {
		t.Fatal("clip must not touch short output")
	}
}

// --- scheduler

func TestMonitorFiresAndRetires(t *testing.T) {
	sb, root := newTestSandbox(t, ModeWorkspace)
	sched := newScheduler(sb)
	flag := filepath.Join(root, "flag")
	if _, err := sched.Add(&Monitor{Cmd: "touch x", Every: time.Second, Timeout: time.Minute, Runs: 1}); err == nil {
		t.Fatal("background monitors must be read-only in workspace mode")
	}
	id, err := sched.Add(&Monitor{Cmd: "cat flag", Until: regexp.MustCompile("ready"), Prompt: "go on", Every: 50 * time.Millisecond, Timeout: 5 * time.Second, Runs: 1})
	must(t, err)
	time.Sleep(120 * time.Millisecond)
	select {
	case w := <-sched.wake:
		t.Fatalf("fired before the condition held: %+v", w)
	default:
	}
	must(t, os.WriteFile(flag, []byte("ready\n"), 0o644))
	select {
	case w := <-sched.wake:
		if w.ID != id || !strings.Contains(w.Text, "go on") || !strings.Contains(w.Text, "ready") {
			t.Fatalf("wake: %+v", w)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("monitor never fired")
	}
	if sched.Active() != 0 {
		t.Fatal("monitor should retire after its run")
	}
	// a timer with no cmd fires each tick until cancelled
	id, err = sched.Add(&Monitor{Prompt: "tick", Every: 30 * time.Millisecond, Timeout: 5 * time.Second, Runs: 0})
	must(t, err)
	for i := 0; i < 2; i++ {
		select {
		case <-sched.wake:
		case <-time.After(time.Second):
			t.Fatal("timer did not tick")
		}
	}
	if !sched.Cancel(id) || sched.Active() != 0 {
		t.Fatal("cancel")
	}
	// timeout wakes with a report
	_, err = sched.Add(&Monitor{Cmd: "false", Every: 20 * time.Millisecond, Timeout: 60 * time.Millisecond, Runs: 1})
	must(t, err)
	select {
	case w := <-sched.wake:
		if !strings.Contains(w.Text, "timed out") {
			t.Fatalf("want timeout report, got %+v", w)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no timeout report")
	}
}

func TestMonitorToolDefaultsToOneRun(t *testing.T) {
	sb, _ := newTestSandbox(t, ModeWorkspace)
	sched := newScheduler(sb)
	tool := monitorTool(sched)
	out, err := tool.Run(context.Background(), json.RawMessage(`{"cmd":"true","every":"5s","background":true}`))
	must(t, err)
	if !strings.Contains(out, "armed") {
		t.Fatalf("arm: %s", out)
	}
	m := sched.live["m1"]
	if m == nil || m.Runs != 1 {
		t.Fatalf("omitted runs must mean one firing, got %+v", m)
	}
	out, err = tool.Run(context.Background(), json.RawMessage(`{"prompt":"tick","every":"5s","background":true,"runs":0}`))
	must(t, err)
	if sched.live["m2"].Runs != 0 {
		t.Fatal("explicit runs=0 means forever")
	}
	sched.CancelAll()
	if _, err := tool.Run(context.Background(), json.RawMessage(`{"cmd":"x","until":"(","background":true}`)); err == nil {
		t.Fatal("bad regexp must be rejected")
	}
	if _, err := tool.Run(context.Background(), json.RawMessage(`{}`)); err == nil {
		t.Fatal("cmd or prompt is required")
	}
}

// --- the loop, end to end against a fake provider

// fakeProvider scripts replies: each request pops the next reply. It records
// every request body so tests can assert what the model was shown.
type fakeProvider struct {
	t       *testing.T
	replies []string
	bodies  []map[string]any
}

func (f *fakeProvider) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		f.t.Errorf("bad request body: %v", err)
	}
	f.bodies = append(f.bodies, body)
	if len(f.replies) == 0 {
		w.WriteHeader(500)
		fmt.Fprint(w, `{"error":{"message":"script exhausted"}}`)
		return
	}
	reply := f.replies[0]
	f.replies = f.replies[1:]
	if strings.HasPrefix(reply, "ERR ") {
		code, msg, _ := strings.Cut(reply[4:], " ")
		w.WriteHeader(atoi(code))
		fmt.Fprintf(w, `{"error":{"message":%q}}`, msg)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprint(w, reply)
}

func atoi(s string) int {
	n := 0
	for _, c := range s {
		n = n*10 + int(c-'0')
	}
	return n
}

func textReply(s string) string {
	return sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":5}}`, s))
}

func toolReply(id, name, args string) string {
	return sse(fmt.Sprintf(`{"choices":[{"delta":{"reasoning_content":"plan","tool_calls":[{"index":0,"id":%q,"function":{"name":%q,"arguments":%q}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":5}}`, id, name, args))
}

type quietUI struct{ calls []string }

func (q *quietUI) Text(string)               {}
func (q *quietUI) Reasoning(string)          {}
func (q *quietUI) ToolCall(name, s string)   { q.calls = append(q.calls, name+":"+s) }
func (q *quietUI) ToolResult(string, string) {}
func (q *quietUI) Note(string, ...any)       {}
func (q *quietUI) Sub() UI                   { return q }
func (q *quietUI) End()                      {}

func newTestAgent(t *testing.T, fp *fakeProvider, mode Mode) (*Agent, string) {
	srv := httptest.NewServer(fp)
	t.Cleanup(srv.Close)
	sb, root := newTestSandbox(t, mode)
	p := *providers["deepseek"]
	p.BaseURL = srv.URL
	cfg := &Config{Provider: &p, APIKey: "k", Model: "m", Think: "high", Context: 1000, Root: root, Mode: mode, MaxRounds: 5}
	sched := newScheduler(sb)
	tb := newToolbox(sb)
	var a *Agent
	tb.add(tb.readTool())
	tb.add(tb.writeTool())
	tb.add(tb.editTool())
	tb.add(tb.execTool())
	tb.add(agentTool(func() *Agent { return a }))
	tb.add(monitorTool(sched))
	a = newAgent(cfg, newClient(cfg), tb, sched, "sys", tb.Specs(), &quietUI{})
	return a, root
}

func TestTurnRunsToolsUntilTextAnswer(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "write", `{"path":"hello.txt","content":"hi\n"}`),
		toolReply("c2", "exec", `{"cmd":"cat hello.txt"}`),
		textReply("done"),
	}}
	a, root := newTestAgent(t, fp, ModeWorkspace)
	out, err := a.Turn(context.Background(), "make hello")
	must(t, err)
	if out != "done" {
		t.Fatalf("final: %q", out)
	}
	if b, _ := os.ReadFile(filepath.Join(root, "hello.txt")); string(b) != "hi\n" {
		t.Fatalf("file: %q", b)
	}
	// transcript shape: user, assistant(tool), tool, assistant(tool), tool, assistant
	roles := ""
	for _, m := range a.msgs {
		roles += m.Role[:1]
	}
	if roles != "uatata" {
		t.Fatalf("roles: %s", roles)
	}
	if a.msgs[4].Content != "hi\n\n[exit 0]" {
		t.Fatalf("exec result: %q", a.msgs[4].Content)
	}
	// the third request replayed reasoning_content on both assistant turns
	msgs := fp.bodies[2]["messages"].([]any)
	for _, m := range msgs {
		mm := m.(map[string]any)
		if mm["role"] == "assistant" {
			if v, ok := mm["reasoning_content"]; !ok || v != "plan" {
				t.Fatalf("reasoning not replayed: %v", mm)
			}
		}
	}
	if a.usage.Prompt != 300 || a.usage.Completion != 15 {
		t.Fatalf("usage: %+v", a.usage)
	}
}

func TestReceiptTracksWrites(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "exec", `{"cmd":"true"}`),
		textReply("fixed it (a lie)"),
		toolReply("c2", "write", `{"path":"x.txt","content":"1"}`),
		toolReply("c3", "edit", `{"path":"x.txt","old":"1","new":"2"}`),
		textReply("done"),
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	_, err := a.Turn(context.Background(), "fix")
	must(t, err)
	if calls, written := a.tb.Receipt(); calls != 1 || len(written) != 0 {
		t.Fatalf("receipt after a talk-only turn: calls=%d written=%v", calls, written)
	}
	_, err = a.Turn(context.Background(), "really fix")
	must(t, err)
	if calls, written := a.tb.Receipt(); calls != 2 || len(written) != 1 || written[0] != "x.txt" {
		t.Fatalf("receipt: calls=%d written=%v", calls, written)
	}
	if _, written := a.tb.Receipt(); written != nil {
		t.Fatal("receipt must reset")
	}
}

func TestTurnHeadlessDeniesAndReportsToModel(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "exec", `{"cmd":"rm -rf build"}`),
		textReply("could not"),
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	_, err := a.Turn(context.Background(), "clean")
	must(t, err)
	if !strings.Contains(a.msgs[2].Content, "error: command needs confirmation") {
		t.Fatalf("model must be told why: %q", a.msgs[2].Content)
	}
}

func TestTurnCompactsOnOverflow(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		"ERR 400 This model's maximum context length is 1000 tokens",
		textReply("SUMMARY"), // compaction call
		textReply("after"),
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	a.msgs = []Message{{Role: "user", Content: "old"}, {Role: "assistant", Content: "older", ReasoningContent: new(string)}}
	out, err := a.Turn(context.Background(), "next")
	must(t, err)
	if out != "after" {
		t.Fatalf("final: %q", out)
	}
	if fp.bodies[1]["tools"] != nil {
		t.Fatal("compaction request must not carry tools")
	}
	if !strings.Contains(a.msgs[0].Content, "SUMMARY") || a.msgs[0].Role != "user" {
		t.Fatalf("compacted head: %+v", a.msgs[0])
	}
	// system, summary, ack, then the unanswered "next" kept verbatim
	last := fp.bodies[2]["messages"].([]any)
	if len(last) != 4 || last[3].(map[string]any)["content"] != "next" {
		t.Fatalf("post-compaction request has %d messages: %v", len(last), last)
	}
}

func TestTurnSkipsTruncatedToolCalls(t *testing.T) {
	cut := sse(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write","arguments":"{\"path\":\"a\",\"con"}}]},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	fp := &fakeProvider{t: t, replies: []string{cut, textReply("ok")}}
	a, root := newTestAgent(t, fp, ModeWorkspace)
	_, err := a.Turn(context.Background(), "x")
	must(t, err)
	if !strings.Contains(a.msgs[2].Content, "token cap") {
		t.Fatalf("model must be told: %q", a.msgs[2].Content)
	}
	if _, err := os.Stat(filepath.Join(root, "a")); err == nil {
		t.Fatal("truncated call must not run")
	}
}

func TestRequestCeiling(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "exec", `{"cmd":"true"}`),
		toolReply("c2", "exec", `{"cmd":"true"}`),
		textReply("never"),
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	a.cfg.MaxReqs = 2
	_, err := a.Turn(context.Background(), "x")
	if err == nil || !strings.Contains(err.Error(), "request ceiling") || len(fp.bodies) != 2 {
		t.Fatalf("ceiling: err=%v requests=%d", err, len(fp.bodies))
	}
}

func TestErrorBodyRedactsKey(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{"ERR 401 bad key k-secret-123 rejected"}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	a.cfg.APIKey = "k-secret-123"
	_, err := a.Turn(context.Background(), "x")
	if err == nil || strings.Contains(err.Error(), "k-secret-123") || !strings.Contains(err.Error(), "[redacted]") {
		t.Fatalf("key leaked: %v", err)
	}
}

func TestTurnRetriesOn429(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{"ERR 429 slow down", textReply("ok")}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	out, err := a.Turn(context.Background(), "x")
	must(t, err)
	if out != "ok" || len(fp.bodies) != 2 {
		t.Fatalf("retry: out=%q requests=%d", out, len(fp.bodies))
	}
}

func TestSubAgentIsBoundedAndReports(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "agent", `{"task":"look around","readonly":true}`),
		// child: tries to write (refused: read-only), then agent (absent), then reports
		toolReply("k1", "write", `{"path":"x","content":"y"}`),
		toolReply("k2", "agent", `{"task":"recurse"}`),
		textReply("child report"),
		textReply("parent done"),
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	out, err := a.Turn(context.Background(), "delegate")
	must(t, err)
	if out != "parent done" {
		t.Fatalf("final: %q", out)
	}
	if a.msgs[2].Content != "child report" {
		t.Fatalf("parent must see only the child's report: %q", a.msgs[2].Content)
	}
	childTools := fp.bodies[1]["tools"].([]any)
	for _, tl := range childTools {
		name := tl.(map[string]any)["function"].(map[string]any)["name"]
		if name == "agent" || name == "monitor" {
			t.Fatalf("child must not have %s", name)
		}
	}
	sys := fp.bodies[1]["messages"].([]any)[0].(map[string]any)["content"].(string)
	if !strings.Contains(sys, "sub-agent") || !strings.Contains(sys, "sandbox: read-only") {
		t.Fatalf("child system prompt: %s", sys)
	}
	if a.usage.Prompt != 500 {
		t.Fatalf("child usage must roll up: %+v", a.usage)
	}
}

func TestRoundBudget(t *testing.T) {
	fp := &fakeProvider{t: t}
	for i := 0; i < 5; i++ {
		fp.replies = append(fp.replies, toolReply(fmt.Sprint("c", i), "exec", `{"cmd":"true"}`))
	}
	fp.replies = append(fp.replies, textReply("stopped"))
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	out, err := a.Turn(context.Background(), "loop forever")
	must(t, err)
	if out != "stopped" || !strings.Contains(a.msgs[len(a.msgs)-2].Content, "round budget") {
		t.Fatalf("budget: out=%q last tool msg=%q", out, a.msgs[len(a.msgs)-2].Content)
	}
}

// --- sessions

func TestSessionAppendResumeAndTrailingToolCall(t *testing.T) {
	root := t.TempDir()
	cfg := &Config{Root: root, Model: "m"}
	s, hist, err := openSession(cfg, "")
	must(t, err)
	if hist != nil {
		t.Fatal("fresh session has no history")
	}
	if b, err := os.ReadFile(filepath.Join(root, runtimeDir, ".gitignore")); err != nil || string(b) != "sessions/\n" {
		t.Fatalf("runtime dir must ignore its sessions: %q %v", b, err)
	}
	s.Append(Message{Role: "user", Content: "a"})
	if info, err := os.Stat(s.path); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("session file mode: %v %v", info.Mode(), err)
	}
	s.Append(Message{Role: "assistant", Content: "b", ReasoningContent: new(string)})
	s.Append(Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "c", Type: "function"}}})
	_, hist, err = openSession(cfg, "last")
	must(t, err)
	if len(hist) != 2 || hist[1].Content != "b" || hist[1].ReasoningContent == nil {
		t.Fatalf("resume: %+v", hist)
	}
	if _, _, err := openSession(cfg, "nope"); err == nil {
		t.Fatal("unknown id must fail")
	}
	sub := s.Sub()
	sub.Append(Message{Role: "user", Content: "child"})
	if !strings.HasSuffix(sub.id, "-sub1") {
		t.Fatalf("sub id: %s", sub.id)
	}
	if _, hist, err := openSession(cfg, "last"); err != nil || len(hist) != 2 {
		t.Fatalf("resume last must skip sub transcripts: %v %d", err, len(hist))
	}
	if _, hist, err := openSession(cfg, sub.id); err != nil || len(hist) != 1 || hist[0].Content != "child" {
		t.Fatalf("sub transcript by id: %v %v", err, hist)
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
