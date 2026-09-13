package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
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
	s.Close() // the log is exclusive while open; see TestSessionIsExclusiveWhileOpen
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

// The review's scenario: a torn tail must not only load, it must be
// repaired on disk before anything is appended after it. The test is the
// cycle: resume, append, resume, append, resume, with history intact.
func TestTornTailIsRepairedOnDiskAndSurvivesResumeCycles(t *testing.T) {
	cases := map[string]string{
		"partial record":       `{"role":"user","content":"cut off mid-wri`,
		"complete, no newline": `{"role":"user","content":"whole"}`,
	}
	for name, tail := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := &Config{Root: t.TempDir(), Model: "m"}
			s, _, err := openSession(cfg, "")
			must(t, err)
			must(t, s.Append(Message{Role: "user", Content: "a"}))
			must(t, s.Append(Message{Role: "assistant", Content: "b", ReasoningContent: new(string)}))
			s.Close()
			f, err := os.OpenFile(s.path, os.O_WRONLY|os.O_APPEND, 0o600)
			must(t, err)
			f.WriteString(tail)
			f.Close()
			want := "ua"
			if name == "complete, no newline" {
				want = "uau" // a whole record is history, newline or not
			}
			for cycle := 1; cycle <= 3; cycle++ {
				r, hist, err := openSession(cfg, s.id)
				if err != nil {
					t.Fatalf("resume %d: %v", cycle, err)
				}
				if roles(hist) != want {
					t.Fatalf("resume %d: roles %s, want %s", cycle, roles(hist), want)
				}
				must(t, r.Append(Message{Role: "user", Content: fmt.Sprint("more", cycle)}))
				must(t, r.Append(Message{Role: "assistant", Content: "ok", ReasoningContent: new(string)}))
				r.Close()
				want += "ua"
			}
			b, _ := os.ReadFile(s.path)
			if !bytes.HasSuffix(b, []byte("\n")) || bytes.Contains(b, []byte("mid-wri")) {
				t.Fatalf("log not repaired on disk: %q", b)
			}
			// every line is one whole record: the file is readable by anything
			for i, line := range bytes.Split(bytes.TrimSuffix(b, []byte("\n")), []byte("\n")) {
				if !json.Valid(line) {
					t.Fatalf("line %d is not JSON: %q", i+1, line)
				}
			}
		})
	}
	// a corrupt line in the middle is still fatal: that is not a torn write
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	s, _, err := openSession(cfg, "")
	must(t, err)
	s.Close()
	must(t, os.WriteFile(s.path, []byte("{}\n{\"role\":\"user\",\"content\":\"a\"}\nnot json\n{\"role\":\"user\",\"content\":\"b\"}\n"), 0o600))
	if _, _, err := loadSession(s.path); err == nil || !strings.Contains(err.Error(), "corrupt line 3") {
		t.Fatalf("mid-file corruption must fail: %v", err)
	}
	if _, _, err := openSession(cfg, s.id); err == nil {
		t.Fatal("resume of a corrupt log must fail, not truncate it")
	}
	if b, _ := os.ReadFile(s.path); !bytes.Contains(b, []byte("not json")) {
		t.Fatal("a failed resume must not rewrite the file")
	}
}

func TestSessionIsExclusiveWhileOpen(t *testing.T) {
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	s, _, err := openSession(cfg, "")
	must(t, err)
	must(t, s.Append(Message{Role: "user", Content: "a"}))
	if _, _, err := openSession(cfg, s.id); err == nil || !strings.Contains(err.Error(), "another process") {
		t.Fatalf("a second writer must be refused: %v", err)
	}
	if _, _, err := openSession(cfg, "last"); err == nil {
		t.Fatal("resume last must be refused too")
	}
	s.Close()
	r, hist, err := openSession(cfg, s.id)
	must(t, err)
	r.Close()
	if roles(hist) != "u" {
		t.Fatalf("after close: %s", roles(hist))
	}
}

// Two sessions created in the same second order by chance in their ids.
// "last" is the one most recently written to, whatever its id.
func TestLastSessionIsTheLastActive(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sessions")
	must(t, os.MkdirAll(dir, 0o700))
	older := filepath.Join(dir, "20260913-120000-ffffff.jsonl")
	newer := filepath.Join(dir, "20260913-120000-000000.jsonl")
	must(t, os.WriteFile(older, []byte("{}\n"), 0o600))
	must(t, os.WriteFile(newer, []byte("{}\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(dir, "20260913-130000-aaaaaa-sub1.jsonl"), []byte("{}\n"), 0o600))
	base := time.Date(2026, 9, 13, 12, 0, 0, 0, time.UTC)
	must(t, os.Chtimes(older, base, base))
	must(t, os.Chtimes(newer, base.Add(time.Second), base.Add(time.Second)))
	must(t, os.Chtimes(filepath.Join(dir, "20260913-130000-aaaaaa-sub1.jsonl"), base.Add(time.Hour), base.Add(time.Hour)))
	if id, err := lastSession(dir); err != nil || id != "20260913-120000-000000" {
		t.Fatalf("last: %s %v", id, err)
	}
	// the older-named one becomes last when it is written to again
	must(t, os.Chtimes(older, base.Add(time.Minute), base.Add(time.Minute)))
	if id, _ := lastSession(dir); id != "20260913-120000-ffffff" {
		t.Fatalf("last after activity: %s", id)
	}
	// a resumed session is the last one afterwards, by its own append
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	a, _, err := openSession(cfg, "")
	must(t, err)
	a.Close()
	b, _, err := openSession(cfg, "")
	must(t, err)
	b.Close()
	past := time.Now().Add(-2 * time.Hour)
	must(t, os.Chtimes(a.path, past, past))
	r, _, err := openSession(cfg, a.id)
	must(t, err)
	must(t, r.Append(Message{Role: "user", Content: "back"}))
	r.Close()
	if id, _ := lastSession(sessionDir(cfg)); id != a.id {
		t.Fatalf("resumed session must be last: %s want %s", id, a.id)
	}
}

// Replace stages the whole new log before the session switches to it. A
// write that fails after the header must leave the old log as the session
// and no half-written file behind. The failure is a file-size limit, so
// the header fits and the summary record does not.
func TestReplaceKeepsTheOldLogWhenStagingFails(t *testing.T) {
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	s, _, err := openSession(cfg, "")
	must(t, err)
	defer s.Close()
	must(t, s.Append(Message{Role: "user", Content: "old"}))
	oldID, oldPath := s.id, s.path
	var lim syscall.Rlimit
	must(t, syscall.Getrlimit(syscall.RLIMIT_FSIZE, &lim))
	small := lim
	small.Cur = 200 // a header fits, a 4 KB record does not
	if err := syscall.Setrlimit(syscall.RLIMIT_FSIZE, &small); err != nil {
		t.Skip("cannot lower RLIMIT_FSIZE:", err)
	}
	t.Cleanup(func() { syscall.Setrlimit(syscall.RLIMIT_FSIZE, &lim) })
	big := Message{Role: "user", Content: strings.Repeat("x", 4096)}
	err = s.Replace([]Message{big})
	syscall.Setrlimit(syscall.RLIMIT_FSIZE, &lim)
	if err == nil {
		t.Fatal("replace must fail when the staged log cannot be written")
	}
	if s.id != oldID || s.path != oldPath {
		t.Fatalf("session switched to a partial log: %s", s.id)
	}
	must(t, s.Append(Message{Role: "assistant", Content: "still here", ReasoningContent: new(string)}))
	entries, _ := os.ReadDir(s.dir)
	if len(entries) != 1 {
		t.Fatalf("staged file left behind: %d files", len(entries))
	}
	s.Close()
	_, hist, err := openSession(cfg, oldID)
	must(t, err)
	if roles(hist) != "ua" {
		t.Fatalf("old log damaged: %s", roles(hist))
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
	for _, want := range []string{`"min":"` + version + `"`, `"id":"` + s.id + `"`, `"model":"m"`, `"created":`} {
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

// llama.cpp numbers calls per reply (call_0, call_1 ...), so ids repeat
// across batches. Repair must treat each batch on its own: one result per
// call, never a duplicate that would brick the log.
func TestRepairHistoryWithRepeatedIDs(t *testing.T) {
	call := func(ids ...string) Message {
		m := Message{Role: "assistant"}
		for _, id := range ids {
			m.ToolCalls = append(m.ToolCalls, ToolCall{ID: id, Type: "function"})
		}
		return m
	}
	res := func(id string) Message { return Message{Role: "tool", ToolCallID: id, Content: "r"} }
	in := []Message{
		{Role: "user", Content: "u"},
		call("call_0"), res("call_0"),
		call("call_0", "call_1"), res("call_0"),
		call("call_0"), // crash here, no result
	}
	out, tail := repairHistory(in)
	if len(tail) != 1 || tail[0].ToolCallID != "call_0" {
		t.Fatalf("tail: %+v", tail)
	}
	if roles(out) != "uatattat" {
		t.Fatalf("roles: %s", roles(out))
	}
	assertClosedBatches(t, out)
	// the mid-file gap (call_1) was closed in memory, in place
	if out[5].ToolCallID != "call_1" || out[5].Content != interruptedResult {
		t.Fatalf("gap: %+v", out[5])
	}
	again, tail2 := repairHistory(out)
	if roles(again) != roles(out) || len(tail2) != 0 {
		t.Fatal("not idempotent")
	}
}

func TestSessionIDsAreBareNames(t *testing.T) {
	cfg := &Config{Root: t.TempDir(), Model: "m"}
	outside := filepath.Join(t.TempDir(), "victim.jsonl")
	must(t, os.WriteFile(outside, []byte("{}\n"), 0o600))
	for _, id := range []string{"../../" + strings.TrimSuffix(outside, ".jsonl"), "/etc/passwd", "a/b", "..", ".hidden"} {
		if _, _, err := openSession(cfg, id); err == nil || !strings.Contains(err.Error(), "bad session id") {
			t.Errorf("%q: %v", id, err)
		}
	}
	if b, _ := os.ReadFile(outside); string(b) != "{}\n" {
		t.Fatal("a path-shaped id touched a file outside the sessions dir")
	}
	if _, err := sessionPath("/d", "20260913-083314-7d3384-sub1"); err != nil {
		t.Fatal(err)
	}
}
