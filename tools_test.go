package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

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

func newToolboxWithFiles(t *testing.T, mode Mode) (*Toolbox, string, func(name, args string) string) {
	t.Helper()
	sb, root := newTestSandbox(t, mode)
	tb := newToolbox(sb)
	tb.add(tb.readTool())
	tb.add(tb.writeTool())
	tb.add(tb.editTool())
	tb.add(tb.execTool())
	call := func(name, args string) string {
		return tb.Call(context.Background(), ToolCall{Function: FuncCall{Name: name, Arguments: args}}, tb.Specs())
	}
	return tb, root, call
}

func TestReadWriteEditRules(t *testing.T) {
	tb, root, call := newToolboxWithFiles(t, ModeWorkspace)
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
	// atomic replace leaves no temp file behind and keeps the mode
	must(t, os.Chmod(filepath.Join(root, "b.txt"), 0o600))
	call("read", `{"path":"b.txt"}`)
	call("edit", `{"path":"b.txt","old":"l1","new":"L1"}`)
	if info, err := os.Stat(filepath.Join(root, "b.txt")); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("mode not preserved: %v %v", info.Mode(), err)
	}
	if m, _ := filepath.Glob(filepath.Join(root, "*.tmp")); len(m) != 0 {
		t.Fatalf("temp files left behind: %v", m)
	}
	tb.sb.Mode = ModeReadOnly
	if out := call("edit", `{"path":"b.txt","old":"l1","new":"x"}`); !strings.Contains(out, "read-only") {
		t.Fatalf("read-only must refuse edits: %s", out)
	}
	if out := call("read", `{"path":"../etc"}`); !strings.Contains(out, "outside") {
		t.Fatalf("jail: %s", out)
	}
	if out := call("nope", `{}`); !strings.Contains(out, "no tool") {
		t.Fatalf("unknown tool: %s", out)
	}
	if calls, written := tb.Receipt(); calls != 0 || strings.Join(written, ",") != "deep/a.txt,b.txt" {
		t.Fatalf("receipt: calls=%d written=%v", calls, written)
	}
}

// The temp file behind an atomic write is private to the call: writers
// racing on one target never share it, an unrelated file beside the target
// is never truncated, and nothing is left behind.
func TestAtomicWriteTempIsPrivate(t *testing.T) {
	tb, root, _ := newToolboxWithFiles(t, ModeWorkspace)
	bystander := filepath.Join(root, "t.txt.mingo-tmp")
	must(t, os.WriteFile(bystander, []byte("mine"), 0o600))
	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for i := range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- tb.writeAtomic("t.txt", []byte(fmt.Sprintf("writer %d\n", i)))
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("a racing writer failed: %v", err)
		}
	}
	got, _ := os.ReadFile(filepath.Join(root, "t.txt"))
	if !strings.HasPrefix(string(got), "writer ") {
		t.Fatalf("target holds %q", got)
	}
	if b, _ := os.ReadFile(bystander); string(b) != "mine" {
		t.Fatalf("bystander truncated: %q", b)
	}
	if m, _ := filepath.Glob(filepath.Join(root, "*.tmp")); len(m) != 0 {
		t.Fatalf("temp files left behind: %v", m)
	}
	// a symlink at a would-be temp name is never followed: the temp name is
	// random, and O_EXCL refuses an existing path anyway
	victim := filepath.Join(root, "victim.txt")
	must(t, os.WriteFile(victim, []byte("keep"), 0o600))
	must(t, os.Symlink("victim.txt", filepath.Join(root, "u.txt.mingo-00000000.tmp")))
	must(t, tb.writeAtomic("u.txt", []byte("new")))
	if b, _ := os.ReadFile(victim); string(b) != "keep" {
		t.Fatalf("victim overwritten through a temp-name symlink: %q", b)
	}
}

// An omitted required field is not an empty one. JSON decoding cannot tell
// them apart; the tool must, or a missing "content" empties a file.
func TestWriteAndEditRequireFields(t *testing.T) {
	_, root, call := newToolboxWithFiles(t, ModeWorkspace)
	path := filepath.Join(root, "keep.txt")
	must(t, os.WriteFile(path, []byte("keep me\n"), 0o600))
	call("read", `{"path":"keep.txt"}`)
	cases := map[string]string{
		"write no content":   `{"path":"keep.txt"}`,
		"write null content": `{"path":"keep.txt","content":null}`,
		"edit no new":        `{"path":"keep.txt","old":"keep me"}`,
		"edit null new":      `{"path":"keep.txt","old":"keep me","new":null}`,
		"edit no old":        `{"path":"keep.txt","new":"x"}`,
	}
	for name, args := range cases {
		tool := strings.Fields(name)[0]
		out := call(tool, args)
		if !strings.HasPrefix(out, "error: missing required argument") {
			t.Errorf("%s: %s", name, out)
		}
		if got, _ := os.ReadFile(path); string(got) != "keep me\n" {
			t.Fatalf("%s changed the file to %q", name, got)
		}
	}
	// an explicit empty new is a deletion, and an explicit empty content is an empty file
	if out := call("edit", `{"path":"keep.txt","old":"keep me\n","new":""}`); !strings.HasPrefix(out, "edited") {
		t.Fatalf("explicit empty new: %s", out)
	}
	if got, _ := os.ReadFile(path); string(got) != "" {
		t.Fatalf("explicit delete: %q", got)
	}
}

func TestNormalizeArgs(t *testing.T) {
	cases := map[string]string{
		``:                   `{}`,
		`  `:                 `{}`,
		`{"a":1}`:            `{"a":1}`,
		`"{\"path\":\"x\"}"`: `{"path":"x"}`, // double-encoded by a small model
		`"just a string"`:    `"just a string"`,
		"{\"cmd\":\"ls\"}\n": `{"cmd":"ls"}`,
		`[1]`:                `[1]`,
	}
	for in, want := range cases {
		if got := string(normalizeArgs(in)); got != want {
			t.Errorf("%q -> %q, want %q", in, got, want)
		}
	}
}

func TestDecodeRequired(t *testing.T) {
	var a struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := decode(json.RawMessage(`{"path":"x","content":""}`), &a, "path", "content"); err != nil {
		t.Fatalf("explicit empty is present: %v", err)
	}
	if err := decode(json.RawMessage(`{"path":"x"}`), &a, "path", "content"); err == nil || !strings.Contains(err.Error(), `"content"`) {
		t.Fatalf("omitted must fail naming the field: %v", err)
	}
	if err := decode(json.RawMessage(`{"path":"x","content":null}`), &a, "path", "content"); err == nil {
		t.Fatal("null must count as missing")
	}
	if err := decode(json.RawMessage(`{"path":"x",`), &a, "path"); err == nil || !strings.HasPrefix(err.Error(), "bad arguments") {
		t.Fatalf("truncated JSON: %v", err)
	}
	if err := decode(json.RawMessage(`{"path":1}`), &a, "path"); err == nil {
		t.Fatal("wrong type must fail")
	}
}

func TestReadRefusesDirsBinaryAndPastEnd(t *testing.T) {
	_, root, call := newToolboxWithFiles(t, ModeWorkspace)
	must(t, os.MkdirAll(filepath.Join(root, "d"), 0o755))
	must(t, os.WriteFile(filepath.Join(root, "bin"), []byte("a\x00b"), 0o644))
	must(t, os.WriteFile(filepath.Join(root, "t.txt"), []byte("1\n2\n"), 0o644))
	if out := call("read", `{"path":"d"}`); !strings.Contains(out, "directory") {
		t.Fatalf("dir: %s", out)
	}
	if out := call("read", `{"path":"bin"}`); !strings.Contains(out, "binary") {
		t.Fatalf("binary: %s", out)
	}
	if out := call("read", `{"path":"t.txt","offset":9}`); !strings.Contains(out, "past the end") {
		t.Fatalf("past end: %s", out)
	}
	if out := call("read", `{}`); !strings.Contains(out, "missing required") {
		t.Fatalf("no path: %s", out)
	}
}

func TestExecToolCwdTimeoutAndExitText(t *testing.T) {
	_, root, call := newToolboxWithFiles(t, ModeFull)
	must(t, os.MkdirAll(filepath.Join(root, "sub"), 0o755))
	out := call("exec", `{"cmd":"pwd","cwd":"sub"}`)
	if !strings.Contains(out, filepath.Join(root, "sub")) || !strings.HasSuffix(out, "[exit 0]") {
		t.Fatalf("cwd: %s", out)
	}
	if out := call("exec", `{"cmd":"pwd","cwd":"../.."}`); !strings.Contains(out, "outside") {
		t.Fatalf("cwd outside root: %s", out)
	}
	if out := call("exec", `{"cmd":"exit 7"}`); !strings.HasSuffix(out, "[exit 7]") {
		t.Fatalf("exit code: %s", out)
	}
	if out := call("exec", `{"cmd":"sleep 5","timeout":1}`); !strings.Contains(out, "[killed: timeout after 1s]") {
		t.Fatalf("timeout text: %s", out)
	}
	if out := call("exec", `{"cmd":"  "}`); !strings.Contains(out, "must not be empty") {
		t.Fatalf("empty cmd: %s", out)
	}
	if out := call("exec", `{"cmd":"true"}`); out != "\n[exit 0]" {
		t.Fatalf("silent command: %q", out)
	}
}

func TestSkillTool(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "skills", "deploy")
	must(t, os.MkdirAll(dir, 0o755))
	must(t, os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte("---\nname: deploy\ndescription: ship\n---\nstep one\n"), 0o644))
	skills := loadSkills(filepath.Join(root, "skills"))
	tool := skillTool(skills)
	out, err := tool.Run(context.Background(), json.RawMessage(`{"name":"deploy"}`))
	must(t, err)
	if !strings.Contains(out, "skill deploy") || !strings.Contains(out, dir) || !strings.HasSuffix(out, "step one") {
		t.Fatalf("skill body: %q", out)
	}
	if _, err := tool.Run(context.Background(), json.RawMessage(`{"name":"nope"}`)); err == nil {
		t.Fatal("unknown skill must fail")
	}
}
