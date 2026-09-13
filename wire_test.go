package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

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
	must(t, err)
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
	must(t, err)
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

// A stream that stops without a finish reason or [DONE] was cut: nothing in
// it may be acted on, least of all a tool call that happens to parse.
func TestAssembleRejectsUnterminatedStreams(t *testing.T) {
	inputs := map[string]string{
		"empty":         "",
		"text":          "data: {\"choices\":[{\"delta\":{\"content\":\"unfinished\"}}]}\n\n",
		"complete call": `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write","arguments":"{\"path\":\"x\",\"content\":\"y\"}"}}]}}]}` + "\n\n",
	}
	for name, input := range inputs {
		t.Run(name, func(t *testing.T) {
			if _, err := assemble(strings.NewReader(input), nopSink{}); err == nil {
				t.Fatal("accepted EOF with neither a finish reason nor a terminal marker")
			}
		})
	}
	// [DONE] alone is a terminator; a finish reason alone is too.
	if _, err := assemble(strings.NewReader("data: [DONE]\n"), nopSink{}); err != nil {
		t.Fatalf("[DONE] must terminate: %v", err)
	}
	r, err := assemble(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n"), nopSink{})
	if err != nil || r.Content != "x" {
		t.Fatalf("finish_reason must terminate: %v %+v", err, r)
	}
}

// Recorded streams under testdata/streams pin the wire shapes of real
// servers. Each .sse has a .want.json beside it; "error": true means the
// stream must be rejected.
func TestAssembleFixtures(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("testdata", "streams", "*.sse"))
	must(t, err)
	if len(files) == 0 {
		t.Fatal("no fixtures")
	}
	for _, f := range files {
		t.Run(filepath.Base(f), func(t *testing.T) {
			raw, err := os.ReadFile(f)
			must(t, err)
			wantRaw, err := os.ReadFile(strings.TrimSuffix(f, ".sse") + ".want.json")
			must(t, err)
			var want struct {
				Error     bool   `json:"error"`
				Content   string `json:"content"`
				Finish    string `json:"finish"`
				Reasoning bool   `json:"reasoning"`
				ToolCalls []struct {
					ID, Name, Arguments string
				} `json:"tool_calls"`
				Usage Usage `json:"usage"`
			}
			must(t, json.Unmarshal(wantRaw, &want))
			r, err := assemble(strings.NewReader(string(raw)), nopSink{})
			if want.Error {
				if err == nil {
					t.Fatalf("must be rejected, got %+v", r)
				}
				return
			}
			must(t, err)
			if r.Content != want.Content || r.Finish != want.Finish || (r.Reasoning != "") != want.Reasoning || r.Usage != want.Usage {
				t.Fatalf("got content=%q finish=%q reasoning=%v usage=%+v", r.Content, r.Finish, r.Reasoning != "", r.Usage)
			}
			if len(r.ToolCalls) != len(want.ToolCalls) {
				t.Fatalf("tool calls: %+v", r.ToolCalls)
			}
			for i, c := range want.ToolCalls {
				got := r.ToolCalls[i]
				if got.ID != c.ID || got.Function.Name != c.Name || got.Function.Arguments != c.Arguments {
					t.Fatalf("call %d: %+v", i, got)
				}
			}
		})
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

	lc := &Client{cfg: &Config{Provider: providers["local"], Model: "m", Think: "off"}}
	raw, _ = json.Marshal(lc.request(msgs, tools))
	if !strings.Contains(string(raw), `"chat_template_kwargs":{"enable_thinking":false}`) || strings.Contains(string(raw), "reasoning_content") {
		t.Fatalf("local thinking-off shape wrong: %s", raw)
	}
}

// Requests are a pure function of the transcript: the same messages produce
// byte-identical bodies, which is what a prompt cache needs.
func TestRequestIsDeterministic(t *testing.T) {
	c := &Client{cfg: &Config{Provider: providers["deepseek"], Model: "m", Think: "high"}}
	msgs := []Message{{Role: "system", Content: "s"}, {Role: "user", Content: "a"}, {Role: "assistant", Content: "b", ToolCalls: []ToolCall{{ID: "1", Type: "function"}}}, {Role: "tool", ToolCallID: "1", Content: "r"}}
	tools := []ToolSpec{{Name: "x", Parameters: schema(map[string]any{"a": str("a"), "b": num("b")}, "a")}}
	a, _ := json.Marshal(c.request(msgs, tools))
	b, _ := json.Marshal(c.request(msgs, tools))
	if string(a) != string(b) {
		t.Fatal("request bodies differ between identical calls")
	}
}

func TestOverflowDetection(t *testing.T) {
	for _, msg := range []string{
		"This model's maximum context length is 131072 tokens",
		"the request exceeds the available context size. try increasing the context size or enable context shift", // llama-server
		"context_length_exceeded",
	} {
		if !isOverflow(&apiError{400, msg}) {
			t.Fatalf("overflow not detected: %s", msg)
		}
	}
	if isOverflow(&apiError{429, "rate limit"}) || isOverflow(&apiError{400, "bad json"}) {
		t.Fatal("false overflow")
	}
}

func TestIsLoopback(t *testing.T) {
	yes := []string{"http://localhost:8080/v1", "http://127.0.0.1:8080/v1", "http://[::1]:8080/v1", "http://127.5.5.5/v1"}
	no := []string{"http://127.example.invalid/v1", "http://127.0.0.1.example.invalid/v1", "http://10.0.0.1/v1", "http://localhost.evil.com/v1", "::junk"}
	for _, u := range yes {
		if !isLoopback(u) {
			t.Errorf("%s should be loopback", u)
		}
	}
	for _, u := range no {
		if isLoopback(u) {
			t.Errorf("%s must not be loopback", u)
		}
	}
}

func TestRequestCeilingIsABudget(t *testing.T) {
	c := newClient(&Config{MaxReqs: 1})
	must(t, c.take())
	err := c.take()
	if !errors.Is(err, errBudget) {
		t.Fatalf("ceiling must be a budget outcome, got %v", err)
	}
	if c.Requests() != 1 {
		t.Fatalf("requests counted %d", c.Requests())
	}
}

// A 5xx before any output is retried; a stream that dies after output is
// not, because the retry would print a second answer into the same line.
func TestCompleteRetriesOnlyBeforeOutput(t *testing.T) {
	fp := &fakeProvider{t: t, replies: []string{"ERR 500 hiccup", textReply("ok")}}
	srv := httptest.NewServer(fp)
	defer srv.Close()
	p := *providers["deepseek"]
	p.BaseURL = srv.URL
	c := newClient(&Config{Provider: &p, Model: "m", Think: "off"})
	r, err := c.Complete(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nopSink{})
	if err != nil || r.Content != "ok" || fp.requests() != 2 {
		t.Fatalf("5xx before output must retry: err=%v requests=%d", err, fp.requests())
	}

	fp = &fakeProvider{t: t, replies: []string{"RAW data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n", textReply("never")}}
	srv2 := httptest.NewServer(fp)
	defer srv2.Close()
	p.BaseURL = srv2.URL
	c = newClient(&Config{Provider: &p, Model: "m", Think: "off"})
	_, err = c.Complete(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nopSink{})
	if err == nil || !strings.Contains(err.Error(), "before the reply was complete") || fp.requests() != 1 {
		t.Fatalf("a stream cut after output must fail without retry: err=%v requests=%d", err, fp.requests())
	}
}

func TestProbeContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/props" {
			w.WriteHeader(404)
			return
		}
		fmt.Fprint(w, `{"default_generation_settings":{"n_ctx":32768,"params":{}},"total_slots":1}`)
	}))
	defer srv.Close()
	if n := probeContext(srv.URL + "/v1"); n != 32768 {
		t.Fatalf("n_ctx: %d", n)
	}
	if n := probeContext("http://127.0.0.1:1/v1"); n != 0 {
		t.Fatalf("unreachable server must yield 0, got %d", n)
	}
}
