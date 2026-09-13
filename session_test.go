package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFreshSessionsHaveUniquePaths(t *testing.T) {
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	seen := map[string]bool{}
	for i := 0; i < 20; i++ {
		s, _, err := openSession(cfg, "")
		must(t, err)
		if seen[s.path] {
			t.Fatalf("new session reused an existing path: %s", s.path)
		}
		seen[s.path] = true
		s.Close()
	}
}

func TestRotationAlwaysGetsANewPath(t *testing.T) {
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	s, _, err := openSession(cfg, "")
	must(t, err)
	defer s.Close()
	for i := 0; i < 10; i++ {
		old := s.path
		must(t, s.Rotate())
		if s.path == old {
			t.Fatalf("rotation reused %s", old)
		}
		// every file has exactly one header
		b, _ := os.ReadFile(s.path)
		if bytes.Count(b, []byte("\n")) != 1 {
			t.Fatalf("fresh file has %d lines", bytes.Count(b, []byte("\n")))
		}
	}
}

func TestSessionAppendResumeAndSub(t *testing.T) {
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
	must(t, s.Append(Message{Role: "user", Content: "a"}))
	if info, err := os.Stat(s.path); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("session file mode: %v %v", info.Mode(), err)
	}
	must(t, s.Append(Message{Role: "assistant", Content: "b", ReasoningContent: new(string)}))
	must(t, s.Append(Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "c", Type: "function"}}}))
	r, hist, err := openSession(cfg, "last")
	must(t, err)
	r.Close()
	// the dangling call is closed, not dropped: the model's message stays
	if len(hist) != 4 || hist[1].Content != "b" || hist[1].ReasoningContent == nil || hist[3].Role != "tool" || hist[3].ToolCallID != "c" {
		t.Fatalf("resume: %+v", hist)
	}
	if _, _, err := openSession(cfg, "nope"); err == nil {
		t.Fatal("unknown id must fail")
	}
	sub, err := s.Sub()
	must(t, err)
	must(t, sub.Append(Message{Role: "user", Content: "child"}))
	if !strings.HasSuffix(sub.id, "-sub1") {
		t.Fatalf("sub id: %s", sub.id)
	}
	sub.Close()
	if r, hist, err := openSession(cfg, "last"); err != nil || len(hist) != 4 {
		t.Fatalf("resume last must skip sub transcripts: %v %d", err, len(hist))
	} else {
		r.Close()
	}
	if r, hist, err := openSession(cfg, sub.id); err != nil || len(hist) != 1 || hist[0].Content != "child" {
		t.Fatalf("sub transcript by id: %v %v", err, hist)
	} else {
		r.Close()
	}
	// a resumed session's own sub-transcripts never collide with the old ones
	r, _, err = openSession(cfg, s.id)
	must(t, err)
	sub2, err := r.Sub()
	must(t, err)
	if sub2.id == sub.id {
		t.Fatalf("resumed session reused a sub id: %s", sub2.id)
	}
	sub2.Close()
	r.Close()
	s.Close()
}

func TestRepairHistory(t *testing.T) {
	call := func(ids ...string) Message {
		m := Message{Role: "assistant", Content: "c"}
		for _, id := range ids {
			m.ToolCalls = append(m.ToolCalls, ToolCall{ID: id, Type: "function"})
		}
		return m
	}
	res := func(id string) Message { return Message{Role: "tool", ToolCallID: id, Content: "r"} }
	user := Message{Role: "user", Content: "u"}
	cases := []struct {
		name  string
		in    []Message
		roles string
		tail  int
	}{
		{"clean", []Message{user, call("a"), res("a"), {Role: "assistant", Content: "x"}}, "uata", 0},
		{"trailing dangling", []Message{user, call("a")}, "uat", 1},
		{"partial batch at tail", []Message{user, call("a", "b"), res("a")}, "uatt", 1},
		{"gap in the middle", []Message{user, call("a", "b"), res("a"), user, {Role: "assistant", Content: "x"}}, "uattua", 0},
		{"orphan result", []Message{user, res("zombie"), {Role: "assistant", Content: "x"}}, "ua", 0},
		{"two batches", []Message{user, call("a"), res("a"), call("b", "c"), res("c")}, "uatatt", 1},
		{"empty", nil, "", 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out, tail := repairHistory(c.in)
			if roles(out) != c.roles || len(tail) != c.tail {
				t.Fatalf("roles=%s tail=%d, want %s %d", roles(out), len(tail), c.roles, c.tail)
			}
			assertClosedBatches(t, out)
			for _, m := range tail {
				if m.Content != interruptedResult {
					t.Fatalf("tail content: %q", m.Content)
				}
			}
			// idempotent: repairing the repaired history changes nothing
			again, tail2 := repairHistory(out)
			if roles(again) != roles(out) || len(tail2) != 0 {
				t.Fatalf("repair is not idempotent: %s %d", roles(again), len(tail2))
			}
		})
	}
}

// The scenario from the review: a two-call batch with one result, then a
// resume, then more work, then another resume. The transcript must be
// well-formed each time and the repair made once, on disk.
func TestResumeRepairsPartialBatchAndStaysStable(t *testing.T) {
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	s, _, err := openSession(cfg, "")
	must(t, err)
	must(t, s.Append(Message{Role: "user", Content: "work"}))
	must(t, s.Append(Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "a", Type: "function"}, {ID: "b", Type: "function"}}}))
	must(t, s.Append(Message{Role: "tool", ToolCallID: "a", Content: "done"}))
	s.Close()

	r1, hist, err := openSession(cfg, s.id)
	must(t, err)
	assertClosedBatches(t, hist)
	if roles(hist) != "uatt" || hist[3].ToolCallID != "b" || hist[3].Content != interruptedResult {
		t.Fatalf("first resume: %s %+v", roles(hist), hist)
	}
	lines := func() int {
		b, _ := os.ReadFile(s.path)
		return bytes.Count(b, []byte("\n"))
	}
	if lines() != 5 { // header + 3 + the persisted repair
		t.Fatalf("repair not persisted: %d lines", lines())
	}
	must(t, r1.Append(Message{Role: "user", Content: "second"}))
	must(t, r1.Append(Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "dangling", Type: "function"}}}))
	r1.Close()

	r2, hist, err := openSession(cfg, s.id)
	must(t, err)
	r2.Close()
	assertClosedBatches(t, hist)
	if roles(hist) != "uattuat" {
		t.Fatalf("second resume: %s", roles(hist))
	}
	if lines() != 8 {
		t.Fatalf("second repair not persisted once: %d lines", lines())
	}
	r3, hist3, err := openSession(cfg, s.id)
	must(t, err)
	r3.Close()
	if roles(hist3) != roles(hist) || lines() != 8 {
		t.Fatalf("a third resume must change nothing: %s %d", roles(hist3), lines())
	}
}

func TestLoadSessionToleratesTornLastLine(t *testing.T) {
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	s, _, err := openSession(cfg, "")
	must(t, err)
	must(t, s.Append(Message{Role: "user", Content: "a"}))
	must(t, s.Append(Message{Role: "assistant", Content: "b"}))
	s.Close()
	f, err := os.OpenFile(s.path, os.O_WRONLY|os.O_APPEND, 0o600)
	must(t, err)
	f.WriteString(`{"role":"user","content":"cut off mid-wri`)
	f.Close()
	msgs, err := loadSession(s.path)
	must(t, err)
	if roles(msgs) != "ua" {
		t.Fatalf("torn line must be dropped: %s", roles(msgs))
	}
	// but a corrupt line in the middle is fatal
	must(t, os.WriteFile(s.path, []byte("{}\n{\"role\":\"user\",\"content\":\"a\"}\nnot json\n{\"role\":\"user\",\"content\":\"b\"}\n"), 0o600))
	if _, err := loadSession(s.path); err == nil || !strings.Contains(err.Error(), "corrupt line 3") {
		t.Fatalf("mid-file corruption must fail: %v", err)
	}
	// the resumed session then continues appending after the torn line was ignored
	must(t, os.WriteFile(s.path, []byte("{}\n{\"role\":\"user\",\"content\":\"a\"}\n"), 0o600))
	r, hist, err := openSession(cfg, s.id)
	must(t, err)
	must(t, r.Append(Message{Role: "assistant", Content: "c"}))
	r.Close()
	if len(hist) != 1 {
		t.Fatalf("hist: %+v", hist)
	}
}

func TestSessionHeaderAndListing(t *testing.T) {
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	s, _, err := openSession(cfg, "")
	must(t, err)
	must(t, s.Append(Message{Role: "user", Content: "first question here"}))
	must(t, s.Append(Message{Role: "assistant", Content: "answer"}))
	b, _ := os.ReadFile(s.path)
	head := string(bytes.SplitN(b, []byte("\n"), 2)[0])
	for _, want := range []string{`"mote":"` + version + `"`, `"id":"` + s.id + `"`, `"model":"m"`, `"created":`} {
		if !strings.Contains(head, want) {
			t.Fatalf("header lacks %s: %s", want, head)
		}
	}
	s.Close()
	lines, err := describeSessions(sessionDir(cfg))
	must(t, err)
	if len(lines) != 1 || !strings.Contains(lines[0], s.id) || !strings.Contains(lines[0], "2 msgs") || !strings.Contains(lines[0], "first question here") {
		t.Fatalf("listing: %v", lines)
	}
	id, err := lastSession(sessionDir(cfg))
	if err != nil || id != s.id {
		t.Fatalf("last: %s %v", id, err)
	}
	if _, err := lastSession(t.TempDir()); err == nil {
		t.Fatal("empty dir has no last session")
	}
}
