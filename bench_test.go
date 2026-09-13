package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Benchmarks for the pure paths the loop runs on every round. They exist so
// a change that makes the harness slower than the model shows up in
// `make bench`, not in a user's wall clock. Run: go test -bench . -benchmem -run ^$

func BenchmarkAssemble(b *testing.B) {
	var sb strings.Builder
	for i := 0; i < 400; i++ {
		fmt.Fprintf(&sb, "data: {\"choices\":[{\"delta\":{\"content\":\"token %d \"}}]}\n\n", i)
	}
	sb.WriteString(`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":4}}` + "\n\ndata: [DONE]\n\n")
	stream := sb.String()
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := assemble(strings.NewReader(stream), nopSink{}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAssembleRecordedDeepSeek(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "streams", "deepseek-v4-flash-tool-call.sse"))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := assemble(strings.NewReader(string(raw)), nopSink{}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAutoRun(b *testing.B) {
	cmds := []string{
		"ls -la", "git status && git diff --stat | head -50", "find . -name '*.go' | xargs grep -l foo",
		"go vet ./... 2>&1", "FOO=1 grep -rn 'needle' src/ | sort | uniq -c", "rm -rf build", "cat a > b",
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		autoRun(cmds[i%len(cmds)])
	}
}

func BenchmarkReplaceUnique(b *testing.B) {
	var sb strings.Builder
	for i := 0; i < 5000; i++ {
		fmt.Fprintf(&sb, "func f%d(a, b int) int { return a - b }\n", i)
	}
	text := sb.String()
	old := "func f4999(a, b int) int { return a - b }"
	b.Run("exact", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, _, err := replaceUnique(text, old, "x"); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("whitespace-retry", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, _, err := replaceUnique(text, old+"  ", "x"); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkResolve(b *testing.B) {
	root, _ := filepath.EvalSymlinks(b.TempDir())
	sb, err := newSandbox(&Config{Root: root, Mode: ModeWorkspace})
	if err != nil {
		b.Fatal(err)
	}
	defer sb.fs.Close()
	os.MkdirAll(filepath.Join(root, "a", "b", "c"), 0o755)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := sb.Resolve("a/b/c/new.go"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCapWriter(b *testing.B) {
	chunk := []byte(strings.Repeat("x", 4096))
	b.SetBytes(int64(len(chunk)))
	b.ReportAllocs()
	w := newCapWriter(execMaxBytes)
	for i := 0; i < b.N; i++ {
		w.Write(chunk)
	}
}

func BenchmarkRepairHistory(b *testing.B) {
	var msgs []Message
	for i := 0; i < 500; i++ {
		id := fmt.Sprint("c", i)
		msgs = append(msgs,
			Message{Role: "user", Content: "u"},
			Message{Role: "assistant", ToolCalls: []ToolCall{{ID: id}}},
			Message{Role: "tool", ToolCallID: id, Content: "r"},
		)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		repairHistory(msgs)
	}
}

func BenchmarkSessionAppendAndLoad(b *testing.B) {
	cfg := &Config{Root: b.TempDir(), Model: "m"}
	s, _, err := openSession(cfg, "")
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	m := Message{Role: "tool", ToolCallID: "c", Content: strings.Repeat("line of tool output\n", 50)}
	b.Run("append+fsync", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := s.Append(m); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("load", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := loadSession(s.path); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkSystemPrompt(b *testing.B) {
	cfg := &Config{Root: b.TempDir(), Mode: ModeWorkspace}
	skills := []Skill{{Name: "a", Description: "aa"}, {Name: "b", Description: "bb"}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		buildSystemPrompt(cfg, defaultSoul, skills, false)
	}
}
