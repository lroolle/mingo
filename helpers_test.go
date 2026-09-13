package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// --- stream builders

func sse(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func textReply(s string) string {
	return sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":5}}`, s))
}

// textReplyCut is a text answer the token cap interrupted.
func textReplyCut(s string) string {
	return sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q},"finish_reason":"length"}],"usage":{"prompt_tokens":100,"completion_tokens":5}}`, s))
}

func toolReply(id, name, args string) string {
	return sse(fmt.Sprintf(`{"choices":[{"delta":{"reasoning_content":"plan","tool_calls":[{"index":0,"id":%q,"function":{"name":%q,"arguments":%q}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":5}}`, id, name, args))
}

// toolReply2 asks for two calls in one batch.
func toolReply2(id1, name1, args1, id2, name2, args2 string) string {
	return sse(fmt.Sprintf(`{"choices":[{"delta":{"reasoning_content":"plan","tool_calls":[{"index":0,"id":%q,"function":{"name":%q,"arguments":%q}},{"index":1,"id":%q,"function":{"name":%q,"arguments":%q}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":5}}`,
		id1, name1, args1, id2, name2, args2))
}

// --- fake provider

// fakeProvider scripts replies: each request pops the next reply. It records
// every request body so tests can assert what the model was shown.
//
//	"ERR <code> <message>"  an HTTP error with that JSON body
//	"RAW <body>"            the body verbatim, no [DONE], a stream that was cut
//	anything else           an SSE stream
type fakeProvider struct {
	t       *testing.T
	mu      sync.Mutex
	replies []string
	bodies  []map[string]any
}

func (f *fakeProvider) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
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
	switch {
	case strings.HasPrefix(reply, "ERR "):
		code, msg, _ := strings.Cut(reply[4:], " ")
		w.WriteHeader(atoi(code))
		fmt.Fprintf(w, `{"error":{"message":%q}}`, msg)
	case strings.HasPrefix(reply, "RAW "):
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, reply[4:])
	default:
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, reply)
	}
}

func (f *fakeProvider) requests() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.bodies)
}

func (f *fakeProvider) body(i int) map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.bodies[i]
}

func atoi(s string) int {
	n := 0
	for _, c := range s {
		n = n*10 + int(c-'0')
	}
	return n
}

// --- UI double

type quietUI struct {
	mu    sync.Mutex
	calls []string
	notes []string
}

func (q *quietUI) Text(string)      {}
func (q *quietUI) Reasoning(string) {}
func (q *quietUI) ToolCall(name, s string) {
	q.mu.Lock()
	q.calls = append(q.calls, name+":"+s)
	q.mu.Unlock()
}
func (q *quietUI) ToolResult(string, string) {}
func (q *quietUI) Note(f string, a ...any) {
	q.mu.Lock()
	q.notes = append(q.notes, fmt.Sprintf(f, a...))
	q.mu.Unlock()
}
func (q *quietUI) Sub() UI { return q }
func (q *quietUI) End()    {}

// cancellingUI cancels the turn from the ToolCall hook: the last moment
// before the loop dispatches a tool.
type cancellingUI struct {
	*quietUI
	cancel context.CancelFunc
}

func (c *cancellingUI) ToolCall(name, s string) { c.cancel(); c.quietUI.ToolCall(name, s) }

// --- builders

func newTestSandbox(t *testing.T, mode Mode) (*Sandbox, string) {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	must(t, err)
	sb, err := newSandbox(&Config{Root: root, Mode: mode})
	must(t, err)
	t.Cleanup(func() { sb.fs.Close() })
	return sb, root
}

func newTestAgent(t *testing.T, fp *fakeProvider, mode Mode) (*Agent, string) {
	t.Helper()
	srv := httptest.NewServer(fp)
	t.Cleanup(srv.Close)
	sb, root := newTestSandbox(t, mode)
	p := *providers["deepseek"]
	p.BaseURL = srv.URL
	cfg := &Config{Provider: &p, APIKey: "k", Model: "m", Think: "high", Context: 1000, Root: root, Mode: mode, MaxRounds: 5}
	tb := newToolbox(sb)
	var a *Agent
	tb.add(tb.readTool())
	tb.add(tb.writeTool())
	tb.add(tb.editTool())
	tb.add(tb.execTool())
	tb.add(agentTool(func() *Agent { return a }))
	a = newAgent(cfg, newClient(cfg), tb, "sys", tb.Specs(), &quietUI{})
	return a, root
}

func roles(msgs []Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Role[:1])
	}
	return b.String()
}

// assertClosedBatches fails when a transcript has a tool call without a
// result or a result without a call.
func assertClosedBatches(t *testing.T, msgs []Message) {
	t.Helper()
	pending := map[string]bool{}
	for i, m := range msgs {
		if m.Role != "tool" && len(pending) != 0 {
			t.Fatalf("message %d (%s) starts before pending tool results are closed: %v", i, m.Role, pending)
		}
		if m.Role == "assistant" {
			for _, c := range m.ToolCalls {
				pending[c.ID] = true
			}
		}
		if m.Role == "tool" {
			if !pending[m.ToolCallID] {
				t.Fatalf("orphan tool result %q at %d", m.ToolCallID, i)
			}
			delete(pending, m.ToolCallID)
		}
	}
	if len(pending) != 0 {
		t.Fatalf("transcript ends with pending results: %v", pending)
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
