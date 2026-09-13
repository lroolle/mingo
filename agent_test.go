package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

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
	if roles(a.msgs) != "uatata" {
		t.Fatalf("roles: %s", roles(a.msgs))
	}
	assertClosedBatches(t, a.msgs)
	if a.msgs[4].Content != "hi\n\n[exit 0]" {
		t.Fatalf("exec result: %q", a.msgs[4].Content)
	}
	// the third request replayed reasoning_content on both assistant turns
	for _, m := range fp.body(2)["messages"].([]any) {
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
	if calls, written := lastReceipt(a); calls != 3 || len(written) != 1 {
		t.Fatalf("run-wide receipt: calls=%d written=%v", calls, written)
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

// A small model repeats a failing call verbatim. The loop answers the
// repeat without running it, so the round is not spent twice.
func TestRepeatedFailingCallIsNotRerun(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "exec", `{"cmd":"rm -rf build"}`),
		toolReply("c2", "exec", `{"cmd":"rm -rf build"}`),
		toolReply("c3", "exec", `{"cmd":"ls"}`),
		toolReply("c4", "exec", `{"cmd":"ls"}`), // a repeat of a success is fine
		textReply("ok"),
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	_, err := a.Turn(context.Background(), "clean")
	must(t, err)
	if !strings.Contains(a.msgs[4].Content, "same call with the same arguments") {
		t.Fatalf("repeat not caught: %q", a.msgs[4].Content)
	}
	if !strings.HasSuffix(a.msgs[8].Content, "[exit 0]") {
		t.Fatalf("a repeated success must run: %q", a.msgs[8].Content)
	}
	ui := a.ui.(*quietUI)
	if len(ui.calls) != 3 {
		t.Fatalf("tool calls shown: %v", ui.calls)
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
	if fp.body(1)["tools"] != nil {
		t.Fatal("compaction request must not carry tools")
	}
	// the summary request saw the history but not the unanswered message
	sum := fp.body(1)["messages"].([]any)
	for _, m := range sum {
		if m.(map[string]any)["content"] == "next" {
			t.Fatal("the pending user message must not be summarized away")
		}
	}
	if !strings.Contains(a.msgs[0].Content, "SUMMARY") || a.msgs[0].Role != "user" {
		t.Fatalf("compacted head: %+v", a.msgs[0])
	}
	// system, summary, ack, then the unanswered "next" kept verbatim
	last := fp.body(2)["messages"].([]any)
	if len(last) != 4 || last[3].(map[string]any)["content"] != "next" {
		t.Fatalf("post-compaction request has %d messages: %v", len(last), last)
	}
}

// Compaction is a transaction: when the summary cannot be obtained, the
// transcript, including the pending user message, is exactly as before.
func TestCompactionFailureLeavesTranscriptIntact(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		"ERR 400 This model's maximum context length is 1000 tokens",
		"ERR 401 key rejected", // not retryable: the summary fails at once
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	a.msgs = []Message{{Role: "user", Content: "old"}, {Role: "assistant", Content: "older", ReasoningContent: new(string)}}
	_, err := a.Turn(context.Background(), "next")
	if err == nil || !strings.Contains(err.Error(), "compaction also failed") {
		t.Fatalf("want a double failure, got %v", err)
	}
	if roles(a.msgs) != "uau" || a.msgs[2].Content != "next" {
		t.Fatalf("transcript changed by a failed compaction: %s %+v", roles(a.msgs), a.msgs)
	}
}

// When the summary request itself overflows, compaction drops the oldest
// half and tries again, so it always shrinks instead of failing for the
// reason it was called.
func TestCompactionShrinksUntilItFits(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		"ERR 400 This model's maximum context length is 1000 tokens", // the turn
		"ERR 400 This model's maximum context length is 1000 tokens", // summary of everything
		textReply("SUMMARY OF HALF"),                                 // summary of the newer half
		textReply("after"),
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	a.msgs = []Message{
		{Role: "user", Content: "one"}, {Role: "assistant", Content: "two", ReasoningContent: new(string)},
		{Role: "user", Content: "three"}, {Role: "assistant", Content: "four", ReasoningContent: new(string)},
	}
	out, err := a.Turn(context.Background(), "next")
	must(t, err)
	if out != "after" {
		t.Fatalf("final: %q", out)
	}
	whole := fp.body(1)["messages"].([]any)
	half := fp.body(2)["messages"].([]any)
	if len(half) >= len(whole) {
		t.Fatalf("second summary request did not shrink: %d vs %d", len(half), len(whole))
	}
	if !strings.Contains(fmt.Sprint(half), "already dropped") {
		t.Fatalf("the model must be told the head is gone: %v", half)
	}
	if !strings.Contains(a.msgs[0].Content, "SUMMARY OF HALF") {
		t.Fatalf("compacted head: %+v", a.msgs[0])
	}
}

// Compaction gets the scrutiny of a turn: a summary cut by the token cap,
// an empty one, or one that does not shrink the transcript is refused,
// and the transcript stays exactly as it was.
func TestCompactionRefusesCutEmptyAndGrowingSummaries(t *testing.T) {
	old := []Message{
		{Role: "user", Content: "one"}, {Role: "assistant", Content: "two", ReasoningContent: new(string)},
	}
	cases := map[string]string{
		"cut":     textReplyCut("SUMMARY that ran out of"),
		"empty":   textReply("   "),
		"growing": textReply(strings.Repeat("a very long summary ", 20)),
	}
	for name, reply := range cases {
		t.Run(name, func(t *testing.T) {
			fp := &fakeProvider{t: t, replies: []string{reply}}
			a, _ := newTestAgent(t, fp, ModeWorkspace)
			s, _, err := openSession(a.cfg, "")
			must(t, err)
			defer s.Close()
			a.session = s
			a.msgs = append([]Message(nil), old...)
			before := s.id
			if err := a.Compact(context.Background()); err == nil {
				t.Fatal("compaction must fail")
			}
			if roles(a.msgs) != "ua" || a.msgs[0].Content != "one" || s.id != before {
				t.Fatalf("state changed by a refused compaction: %s %s", roles(a.msgs), s.id)
			}
		})
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
	assertClosedBatches(t, a.msgs)
}

// A text answer cut by the token cap is not a success: the text comes back
// with errTruncated so a job runner sees exit 4.
func TestTextReplyCutByTokenCapIsTruncatedOutcome(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{textReplyCut("half an ans")}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	out, err := a.Turn(context.Background(), "x")
	if !errors.Is(err, errTruncated) || out != "half an ans" {
		t.Fatalf("out=%q err=%v", out, err)
	}
	if exitCode(err) != 4 {
		t.Fatalf("exit code %d", exitCode(err))
	}
}

// The round budget is enforced by the loop, not by the model's manners:
// after MaxRounds the batch is closed unexecuted, one tool-less request
// gets a report, and the turn ends in errBudget.
func TestRoundBudgetIsTerminal(t *testing.T) {
	fp := &fakeProvider{t: t}
	for i := 0; i < 12; i++ {
		fp.replies = append(fp.replies, toolReply(fmt.Sprint("c", i), "exec", `{"cmd":"true"}`))
	}
	a, root := newTestAgent(t, fp, ModeWorkspace)
	a.cfg.MaxRounds = 2
	// the third request is the report; give it text so the loop can end
	fp.replies = append([]string{fp.replies[0], fp.replies[1], textReply("stopped: did A, not B")}, fp.replies[2:]...)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := a.Turn(ctx, "keep going forever")
	if !errors.Is(err, errBudget) {
		t.Fatalf("want errBudget, got %v", err)
	}
	if out != "stopped: did A, not B" {
		t.Fatalf("the report must come back as the text: %q", out)
	}
	if n := fp.requests(); n != 3 {
		t.Fatalf("budget depended on model cooperation: %d requests, want 3", n)
	}
	if fp.body(2)["tools"] != nil {
		t.Fatal("the report request must not offer tools")
	}
	assertClosedBatches(t, a.msgs)
	if !strings.Contains(a.msgs[len(a.msgs)-2].Content, "round budget exhausted; not executed") {
		t.Fatalf("second batch must be closed unexecuted: %q", a.msgs[len(a.msgs)-2].Content)
	}
	if exitCode(err) != 3 {
		t.Fatalf("exit code %d", exitCode(err))
	}
	// the second call ran (round 1), only the third was refused
	if _, err := os.Stat(root); err != nil {
		t.Fatal(err)
	}
}

func TestRequestCeilingEndsTheTurn(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "exec", `{"cmd":"true"}`),
		toolReply("c2", "exec", `{"cmd":"true"}`),
		textReply("never"),
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	a.cfg.MaxReqs = 2
	_, err := a.Turn(context.Background(), "x")
	if !errors.Is(err, errBudget) || !strings.Contains(err.Error(), "request ceiling") || fp.requests() != 2 {
		t.Fatalf("ceiling: err=%v requests=%d", err, fp.requests())
	}
	assertClosedBatches(t, a.msgs)
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
	if out != "ok" || fp.requests() != 2 {
		t.Fatalf("retry: out=%q requests=%d", out, fp.requests())
	}
}

// A stream cut before anything reached the console is a dropped connection:
// nothing from it may run or enter the transcript, and the request is
// retried once the backoff passes. The retried reply is the turn's.
func TestTurnRejectsCutStreamBeforeExecuting(t *testing.T) {
	cut := "RAW " + `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write","arguments":"{\"path\":\"x\",\"content\":\"y\"}"}}]}}]}` + "\n\n"
	fp := &fakeProvider{t: t, replies: []string{cut, textReply("retried")}}
	a, root := newTestAgent(t, fp, ModeWorkspace)
	out, err := a.Turn(context.Background(), "x")
	must(t, err)
	if out != "retried" || fp.requests() != 2 {
		t.Fatalf("out=%q requests=%d", out, fp.requests())
	}
	if _, err := os.Stat(filepath.Join(root, "x")); err == nil {
		t.Fatal("a call from a cut stream ran")
	}
	if roles(a.msgs) != "ua" || len(a.msgs[1].ToolCalls) != 0 {
		t.Fatalf("nothing from the cut stream may enter the transcript: %s %+v", roles(a.msgs), a.msgs)
	}
}

// Cancelling mid-batch closes the rest of the batch, so the transcript in
// memory and on disk can take the next user message without a 400.
func TestCancelClosesTheBatch(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply2("c1", "exec", `{"cmd":"sleep 10"}`, "c2", "exec", `{"cmd":"echo second"}`),
		textReply("never"),
	}}
	a, _ := newTestAgent(t, fp, ModeFull)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(200 * time.Millisecond); cancel() }()
	_, err := a.Turn(ctx, "x")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want cancel, got %v", err)
	}
	assertClosedBatches(t, a.msgs)
	if roles(a.msgs) != "uatt" {
		t.Fatalf("roles: %s", roles(a.msgs))
	}
	if !strings.Contains(a.msgs[3].Content, "cancelled by the user before this call ran") {
		t.Fatalf("second call must be closed as not executed: %q", a.msgs[3].Content)
	}
	if exitCode(err) != 130 {
		t.Fatalf("exit code %d", exitCode(err))
	}
}

// A cancellation the loop can already see is a reason not to start: the
// context is cancelled from the UI's ToolCall hook, right before dispatch,
// and the tool body must not run.
func TestCancelObservedBeforeDispatchSkipsTheTool(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "write", `{"path":"marker","content":"ran"}`),
		textReply("never"),
	}}
	a, root := newTestAgent(t, fp, ModeFull)
	ctx, cancel := context.WithCancel(context.Background())
	a.ui = &cancellingUI{quietUI: &quietUI{}, cancel: cancel}
	_, err := a.Turn(ctx, "x")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want cancel, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "marker")); err == nil {
		t.Fatal("tool ran after cancellation was observable")
	}
	assertClosedBatches(t, a.msgs)
	if !strings.Contains(a.msgs[2].Content, "not executed") {
		t.Fatalf("call must be closed unexecuted: %q", a.msgs[2].Content)
	}
}

// A batch with duplicate ids, a missing id or a missing name is rejected
// whole, before anything runs.
func TestMalformedBatchIsRejectedBeforeDispatch(t *testing.T) {
	dup := sse(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"same","function":{"name":"write","arguments":"{\"path\":\"a\",\"content\":\"1\"}"}},{"index":1,"id":"same","function":{"name":"write","arguments":"{\"path\":\"b\",\"content\":\"2\"}"}}]},"finish_reason":"tool_calls"}]}`)
	noid := sse(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"write","arguments":"{\"path\":\"c\",\"content\":\"3\"}"}}]},"finish_reason":"tool_calls"}]}`)
	noname := sse(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x","function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
	for name, reply := range map[string]string{"duplicate id": dup, "no id": noid, "no name": noname} {
		t.Run(name, func(t *testing.T) {
			fp := &fakeProvider{t: t, replies: []string{reply}}
			a, root := newTestAgent(t, fp, ModeFull)
			_, err := a.Turn(context.Background(), "x")
			if err == nil || !strings.Contains(err.Error(), "malformed tool batch") {
				t.Fatalf("want a batch error, got %v", err)
			}
			entries, _ := os.ReadDir(root)
			for _, e := range entries {
				if e.Name() != runtimeDir {
					t.Fatalf("a tool ran from a malformed batch: %s", e.Name())
				}
			}
		})
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
	for _, tl := range fp.body(1)["tools"].([]any) {
		if name := tl.(map[string]any)["function"].(map[string]any)["name"]; name == "agent" {
			t.Fatalf("child must not have %s", name)
		}
	}
	sys := fp.body(1)["messages"].([]any)[0].(map[string]any)["content"].(string)
	if !strings.Contains(sys, "sub-agent") || !strings.Contains(sys, "sandbox: read-only") {
		t.Fatalf("child system prompt: %s", sys)
	}
	if a.usage.Prompt != 500 {
		t.Fatalf("child usage must roll up: %+v", a.usage)
	}
}

// A child that runs out of budget still hands its report to the parent,
// marked as stopped, instead of failing the parent's whole turn.
func TestSubAgentBudgetReturnsReportToParent(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "agent", `{"task":"do a lot"}`),
		toolReply("k1", "exec", `{"cmd":"true"}`),
		toolReply("k2", "exec", `{"cmd":"true"}`),
		textReply("child: did half"),
		textReply("parent done"),
	}}
	a, _ := newTestAgent(t, fp, ModeFull)
	a.cfg.MaxRounds = 2
	out, err := a.Turn(context.Background(), "delegate")
	must(t, err)
	if out != "parent done" {
		t.Fatalf("final: %q", out)
	}
	if !strings.HasPrefix(a.msgs[2].Content, "(sub-agent stopped: budget exhausted") || !strings.Contains(a.msgs[2].Content, "child: did half") {
		t.Fatalf("parent must see the stop and the report: %q", a.msgs[2].Content)
	}
	assertClosedBatches(t, a.msgs)
}

// Every message the loop appends reaches the log before the next request,
// and a resumed transcript is byte-for-byte the in-memory one.
func TestTurnPersistsAsItGoes(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{
		toolReply("c1", "exec", `{"cmd":"echo hi"}`),
		textReply("done"),
	}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	s, _, err := openSession(a.cfg, "")
	must(t, err)
	a.session = s
	_, err = a.Turn(context.Background(), "go")
	must(t, err)
	s.Close()
	_, hist, err := openSession(a.cfg, s.id)
	must(t, err)
	if len(hist) != len(a.msgs) {
		t.Fatalf("resumed %d, in memory %d", len(hist), len(a.msgs))
	}
	for i := range hist {
		if hist[i].Role != a.msgs[i].Role || hist[i].Content != a.msgs[i].Content || len(hist[i].ToolCalls) != len(a.msgs[i].ToolCalls) {
			t.Fatalf("message %d differs: %+v vs %+v", i, hist[i], a.msgs[i])
		}
	}
}

// A log that cannot be written fails the turn: an unlogged step is one
// that cannot be resumed, and silence here is how transcripts go missing.
func TestSessionWriteErrorFailsTheTurn(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{textReply("done")}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	s, _, err := openSession(a.cfg, "")
	must(t, err)
	s.Close() // simulate a closed or full disk
	a.session = s
	_, err = a.Turn(context.Background(), "go")
	if err == nil || !strings.HasPrefix(err.Error(), "session:") {
		t.Fatalf("want a session error, got %v", err)
	}
	if fp.requests() != 0 {
		t.Fatal("nothing may be sent to the model when the log is broken")
	}
}

// A model that hits the token cap with tool calls on every reply must
// still run out of rounds: truncation does not exempt a round from the
// budget, or the loop spends forever.
func TestRoundBudgetCountsTruncatedRounds(t *testing.T) {
	cut := sse(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write","arguments":"{\"path\":\"a\",\"con"}}]},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	fp := &fakeProvider{t: t}
	for i := 0; i < 20; i++ {
		fp.replies = append(fp.replies, cut)
	}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	a.cfg.MaxRounds = 3
	fp.replies = append([]string{cut, cut, cut, textReply("gave up")}, fp.replies...)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := a.Turn(ctx, "x")
	if !errors.Is(err, errBudget) || out != "gave up" {
		t.Fatalf("out=%q err=%v", out, err)
	}
	if n := fp.requests(); n != 4 {
		t.Fatalf("%d requests, want MaxRounds+1 = 4", n)
	}
	assertClosedBatches(t, a.msgs)
}

// Compaction commits to the log before memory: when the new log cannot
// be written, the old transcript is still in memory.
func TestCompactionKeepsMemoryWhenTheNewLogFails(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{textReply("SUMMARY")}}
	a, _ := newTestAgent(t, fp, ModeWorkspace)
	s, _, err := openSession(a.cfg, "")
	must(t, err)
	a.session = s
	a.msgs = []Message{{Role: "user", Content: "old"}, {Role: "assistant", Content: "older", ReasoningContent: new(string)}}
	// make the sessions dir unwritable so Rotate cannot create a file
	must(t, os.Chmod(s.dir, 0o500))
	t.Cleanup(func() { os.Chmod(s.dir, 0o700) })
	if os.Getuid() == 0 {
		t.Skip("root ignores directory modes")
	}
	err = a.Compact(context.Background())
	if err == nil {
		t.Fatal("compaction must fail when the log cannot rotate")
	}
	if roles(a.msgs) != "ua" || a.msgs[0].Content != "old" {
		t.Fatalf("memory changed by a failed commit: %s", roles(a.msgs))
	}
}
