package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newTestConsole(quiet bool) (*Console, *bytes.Buffer, *bytes.Buffer) {
	out, errb := &bytes.Buffer{}, &bytes.Buffer{}
	return &Console{out: out, err: errb, quiet: quiet, input: make(chan string), atStart: true}, out, errb
}

// stdout carries the answer and nothing else; everything the harness says
// goes to stderr. That is the whole contract for piping.
func TestConsoleSplitsAnswerFromChatter(t *testing.T) {
	c, out, errb := newTestConsole(false)
	c.Reasoning("thinking")
	c.Text("hello ")
	c.Text("world\n")
	c.ToolCall("exec", "ls")
	c.ToolResult("exec", "a\nb\n[exit 0]")
	c.ToolResult("exec", "error: nope")
	c.Note("note %d", 1)
	c.End()
	if out.String() != "hello world\n" {
		t.Fatalf("stdout: %q", out.String())
	}
	for _, want := range []string{"thinking", "> exec: ls", "3 lines, 12 bytes [exit 0]", "error: nope", "min: note 1"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("stderr lacks %q:\n%s", want, errb.String())
		}
	}
	c, out, errb = newTestConsole(true)
	c.Reasoning("thinking")
	c.ToolCall("exec", "ls")
	c.ToolResult("exec", "x")
	c.Text("answer")
	c.Note("still shown")
	if out.String() != "answer\n" || strings.Contains(errb.String(), "thinking") || strings.Contains(errb.String(), "exec") || !strings.Contains(errb.String(), "still shown") {
		t.Fatalf("quiet: out=%q err=%q", out.String(), errb.String())
	}
}

func TestConsoleSubIndentsAndStaysOffStdout(t *testing.T) {
	c, out, errb := newTestConsole(false)
	c.prefix = ""
	sub := c.Sub().(*Console)
	sub.Text("child says\nmore")
	sub.End()
	sub.Note("n")
	if out.Len() != 0 {
		t.Fatalf("a sub-agent must not write to stdout: %q", out.String())
	}
	if !strings.Contains(errb.String(), "  | child says\n  | more\n") || !strings.Contains(errb.String(), "  | min: n") {
		t.Fatalf("sub output: %q", errb.String())
	}
}

func TestConfirmAnswersAndCancels(t *testing.T) {
	c, _, errb := newTestConsole(false)
	confirm := c.Confirm()
	go func() { c.input <- "y" }()
	if confirm(context.Background(), "rm x") != AllowOnce {
		t.Fatal("y must allow once")
	}
	if !strings.Contains(errb.String(), "allow exec? rm x") {
		t.Fatalf("prompt: %q", errb.String())
	}
	go func() { c.input <- "" }()
	if confirm(context.Background(), "rm x") != Deny {
		t.Fatal("enter must deny")
	}
	go func() { c.input <- "a" }()
	if confirm(context.Background(), "rm x") != AllowAlways {
		t.Fatal("a must answer always; the sandbox applies it")
	}
	// a cancelled turn answers no without waiting for a line
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan bool)
	go func() { done <- confirm(ctx, "rm y") == Deny }()
	select {
	case denied := <-done:
		if !denied {
			t.Fatal("cancel must deny")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("confirm blocked on stdin after cancel")
	}
	close(c.input)
	if confirm(context.Background(), "rm z") != Deny {
		t.Fatal("closed stdin must deny")
	}
}

func TestExitCodes(t *testing.T) {
	cases := map[int]error{
		0:   nil,
		1:   errors.New("x"),
		2:   flag.ErrHelp,
		3:   errBudget,
		4:   errTruncated,
		130: context.Canceled,
	}
	for want, err := range cases {
		if got := exitCode(err); got != want {
			t.Errorf("%v -> %d, want %d", err, got, want)
		}
	}
	if exitCode(errors.Join(errors.New("wrapped"), errBudget)) != 3 {
		t.Error("wrapped budget must map to 3")
	}
	names := map[string]error{"done": nil, "failed": errors.New("x"), "budget": errBudget, "truncated": errTruncated, "cancelled": context.Canceled}
	for want, err := range names {
		if outcome(err) != want {
			t.Errorf("outcome(%v) = %s", err, outcome(err))
		}
	}
}

func TestParseConfig(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "k")
	t.Setenv(envPrefix+"BASE_URL", "")
	cfg, err := parseConfig([]string{"-cwd", t.TempDir(), "-yolo", "-think", "low"})
	must(t, err)
	if cfg.Mode != ModeFull || cfg.Think != "low" || cfg.Provider.Name != "deepseek" || cfg.Context != 0 {
		t.Fatalf("cfg: %+v", cfg)
	}
	if _, err := parseConfig([]string{"-think", "wild"}); err == nil {
		t.Fatal("bad think level must fail")
	}
	// levels are per provider: astra takes xhigh, deepseek does not
	if _, err := parseConfig([]string{"-think", "xhigh", "-cwd", t.TempDir()}); err == nil {
		t.Fatal("deepseek must refuse xhigh")
	}
	t.Setenv("OPENAI_API_KEY", "k")
	if _, err := parseConfig([]string{"-provider", "openai", "-think", "xhigh", "-cwd", t.TempDir()}); err != nil {
		t.Fatal(err)
	}
	if cfg.Model != "deepseek-flash" {
		t.Fatalf("deepseek default model: %s", cfg.Model)
	}
	// a local server has no completion cap of its own; min gives it one,
	// and -max-tokens overrides it
	lc, err := parseConfig([]string{"-provider", "local", "-cwd", t.TempDir()})
	must(t, err)
	if lc.MaxTokens != 4096 || lc.Provider.Dialect != "" {
		t.Fatalf("local defaults: %+v", lc)
	}
	lc, err = parseConfig([]string{"-provider", "local", "-cwd", t.TempDir(), "-max-tokens", "100"})
	must(t, err)
	if lc.MaxTokens != 100 {
		t.Fatalf("max-tokens override: %d", lc.MaxTokens)
	}
	if cfg.MaxTokens != 0 {
		t.Fatalf("cloud providers keep their own cap: %d", cfg.MaxTokens)
	}
	t.Setenv("OPENROUTER_API_KEY", "k")
	orc, err := parseConfig([]string{"-provider", "openrouter", "-cwd", t.TempDir()})
	must(t, err)
	if orc.Model != "deepseek/deepseek-v4.1-flash" || orc.Provider.Headers["X-Title"] != "min" {
		t.Fatalf("openrouter defaults: %+v", orc)
	}
	// explicit boundaries fail closed without a fence, -unfenced accepts
	none := Fence{Kind: "none"}
	if err := requireFence(&Config{NoNet: true, Mode: ModeWorkspace}, none); err == nil || !strings.Contains(err.Error(), "-unfenced") {
		t.Fatalf("no-net without a fence must fail: %v", err)
	}
	if err := requireFence(&Config{Mode: ModeReadOnly}, none); err == nil {
		t.Fatal("read-only without a fence must fail")
	}
	if err := requireFence(&Config{Mode: ModeReadOnly, Unfenced: true}, none); err != nil {
		t.Fatal(err)
	}
	if err := requireFence(&Config{Mode: ModeWorkspace}, none); err != nil {
		t.Fatal("workspace never asked for a fence")
	}
	if err := requireFence(&Config{NoNet: true, Mode: ModeReadOnly}, Fence{Kind: "bwrap"}); err != nil {
		t.Fatal("a fence satisfies both")
	}
	if _, err := parseConfig([]string{"-provider", "nope"}); err == nil {
		t.Fatal("bad provider must fail")
	}
	if _, err := parseConfig([]string{"-max-rounds", "0"}); err == nil {
		t.Fatal("zero rounds must fail")
	}
	t.Setenv(envPrefix+"BASE_URL", "http://127.example.invalid/v1")
	if _, err := parseConfig(nil); err == nil || !strings.Contains(err.Error(), "https is required") {
		t.Fatalf("lookalike loopback must be refused: %v", err)
	}
	t.Setenv(envPrefix+"BASE_URL", "http://127.0.0.1:1/v1")
	if _, err := parseConfig(nil); err != nil {
		t.Fatalf("loopback http must be allowed: %v", err)
	}
	t.Setenv("DEEPSEEK_API_KEY", "")
	if _, err := parseConfig(nil); err == nil || !strings.Contains(err.Error(), "DEEPSEEK_API_KEY") {
		t.Fatalf("missing key: %v", err)
	}
	if _, err := parseConfig([]string{"-show-prompt"}); err != nil {
		t.Fatalf("offline flags need no key: %v", err)
	}
	if _, err := parseConfig([]string{"-sessions"}); err != nil {
		t.Fatalf("offline flags need no key: %v", err)
	}
	if _, err := parseConfig([]string{"-provider", "local"}); err != nil {
		t.Fatalf("local needs no key: %v", err)
	}
	if _, err := parseConfig([]string{"-h"}); !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("help: %v", err)
	}
}

// captureStdout runs f with os.Stdout redirected and returns what it printed.
func captureStdout(t *testing.T, f func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	must(t, err)
	old := os.Stdout
	os.Stdout = w
	done := make(chan string)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()
	f()
	w.Close()
	os.Stdout = old
	return <-done
}

// run end to end, headless, against the fake provider: the exit code and
// the JSON object are the contract a job runner reads.
func TestRunHeadlessOutcomeAndJSON(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "write", `{"path":"out.txt","content":"x"}`),
		textReply("all done"),
		toolReply("c2", "exec", `{"cmd":"true"}`), // the budget run: one request allowed
		textReply("resumed answer"),               // the resume run
	}}
	srv := httptest.NewServer(fp)
	defer srv.Close()
	root := t.TempDir()
	t.Setenv("DEEPSEEK_API_KEY", "k")
	t.Setenv(envPrefix+"BASE_URL", srv.URL)
	t.Setenv(envPrefix+"HOME", t.TempDir())

	out := captureStdout(t, func() {
		if err := run([]string{"-cwd", root, "-quiet", "-json", "-p", "write out"}); err != nil {
			t.Errorf("run: %v", err)
		}
	})
	var res struct {
		Outcome  string   `json:"outcome"`
		Answer   string   `json:"answer"`
		Written  []string `json:"written"`
		Session  string   `json:"session"`
		Requests int      `json:"requests"`
		Usage    Usage    `json:"usage"`
	}
	must(t, json.Unmarshal([]byte(strings.TrimSpace(out)), &res))
	if res.Outcome != "done" || res.Answer != "all done" || len(res.Written) != 1 || res.Written[0] != "out.txt" || res.Requests != 2 || res.Usage.Prompt != 200 {
		t.Fatalf("json result: %+v", res)
	}
	if _, err := os.Stat(filepath.Join(root, runtimeDir, "sessions", res.Session+".jsonl")); err != nil {
		t.Fatalf("session file: %v", err)
	}
	first := res.Session

	// a budget outcome is exit 3 and outcome "budget"
	out = captureStdout(t, func() {
		err := run([]string{"-cwd", root, "-quiet", "-json", "-max-requests", "1", "-p", "loop"})
		if exitCode(err) != 3 {
			t.Errorf("want exit 3, got %d (%v)", exitCode(err), err)
		}
	})
	must(t, json.Unmarshal([]byte(strings.TrimSpace(out)), &res))
	if res.Outcome != "budget" || res.Session == first {
		t.Fatalf("json outcome: %+v", res)
	}

	// -sessions lists both; -show renders one without a model, answer on stdout
	out = captureStdout(t, func() { must(t, run([]string{"-cwd", root, "-sessions"})) })
	if strings.Count(out, "\n") != 2 || !strings.Contains(out, "write out") || !strings.Contains(out, "loop") {
		t.Fatalf("sessions: %q", out)
	}
	out = captureStdout(t, func() { must(t, run([]string{"-cwd", root, "-show", first})) })
	if strings.TrimSpace(out) != "all done" {
		t.Fatalf("show: %q", out)
	}
	if fp.requests() != 3 {
		t.Fatalf("listing and showing must not call the model: %d requests", fp.requests())
	}
	// resuming the first session continues it with the same tool memory rules
	out = captureStdout(t, func() {
		must(t, run([]string{"-cwd", root, "-quiet", "-json", "-resume", first, "-p", "again"}))
	})
	must(t, json.Unmarshal([]byte(strings.TrimSpace(out)), &res))
	if res.Session != first || res.Outcome != "done" {
		t.Fatalf("resume: %+v", res)
	}
	sent := fp.body(3)["messages"].([]any)
	if len(sent) != 6 || sent[5].(map[string]any)["content"] != "again" {
		t.Fatalf("resumed request must carry the old transcript: %d messages", len(sent))
	}
}

func TestReplayRendersATranscript(t *testing.T) {
	c, out, errb := newTestConsole(false)
	r := "why"
	replay(c, []Message{
		{Role: "user", Content: "fix it"},
		{Role: "assistant", ReasoningContent: &r, ToolCalls: []ToolCall{{ID: "1", Function: FuncCall{Name: "exec", Arguments: `{"cmd":"ls"}`}}}},
		{Role: "tool", ToolCallID: "1", Content: "a\n[exit 0]"},
		{Role: "assistant", Content: "done"},
	})
	if out.String() != "done\n" {
		t.Fatalf("stdout: %q", out.String())
	}
	for _, want := range []string{"> fix it", "why", "> exec: ls", "[exit 0]"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("stderr lacks %q:\n%s", want, errb.String())
		}
	}
}
