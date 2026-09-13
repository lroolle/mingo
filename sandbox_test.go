package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestResolveJail(t *testing.T) {
	sb, root := newTestSandbox(t, ModeWorkspace)
	outside := t.TempDir()
	must(t, os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("x"), 0o644))
	must(t, os.Symlink(filepath.Join(outside, "secret.txt"), filepath.Join(root, "link")))
	must(t, os.MkdirAll(filepath.Join(root, runtimeDir, "skills", "s"), 0o755))
	must(t, os.MkdirAll(filepath.Join(root, runtimeDir, "sessions"), 0o755))

	ok := []string{"a.go", "sub/dir/new.txt", root + "/b.go", runtimeDir + "/skills/s/SKILL.md", "./x/../y.txt"}
	for _, p := range ok {
		if _, err := sb.Resolve(p); err != nil {
			t.Errorf("%s should resolve: %v", p, err)
		}
	}
	bad := map[string]string{
		"../escape.txt":                  "outside",
		"/etc/passwd":                    "outside",
		"link":                           "outside",
		runtimeDir + "/sessions/x.jsonl": "off limits",
		runtimeDir + "/SOUL.md":          "off limits",
		".env":                           "credential",
		".env.local":                     "credential",
		"deploy/server.pem":              "credential",
		"keys/id_ed25519":                "credential",
		".ssh/config":                    "credential",
		"config/credentials.json":        "credential",
		"":                               "empty",
		filepath.Join(root, "..", filepath.Base(root)+"x", "f"): "outside",
	}
	for p, want := range bad {
		_, err := sb.Resolve(p)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%q: want error containing %q, got %v", p, want, err)
		}
	}
}

// autoRun is the "runs without asking" gate. Every entry here was either in
// the original allowlist or was a hole a review found; the no list must
// stay adversarial.
func TestAutoRunAllowlist(t *testing.T) {
	yes := []string{
		"ls -la", "cat a.go | grep foo", "git status && git diff --stat", "go build ./... ; go vet ./...",
		"find . -name '*.go' | wc -l", "FOO=1 grep -r x .", "git log --oneline -5", "docker ps",
		"go vet ./... 2>&1", "go build ./... 2>/dev/null", "git branch --show-current", "git branch -a",
		"git tag", "git remote -v", "hostname -f", "date -u", "rg -n foo", "find . -type f -newer x",
		"cat a.go\ncat b.go", "grep -o pattern file", "sort -r file", "tree -L 2",
	}
	no := []string{
		"", "rm -rf x", "cat a > b", "git commit -m x", "go test ./...", "ls $(echo x)", "echo `id`",
		"find . | xargs rm", "sudo ls", "git push", "npm install", "python3 x.py", "go run .", "sed -i s/a/b/ f",
		"ls; make", "git status | tee out", "eval ls", "exec ls", "su -c ls", "doas ls",
		// the review's holes
		"find . -name victim -delete", "find . -exec rm {} \\;", "go fmt ./...", "git branch -D victim",
		"git branch -m old new", "sort -o victim input", "./cat", "/bin/ls", "echo safe\ntouch marker",
		"echo safe & touch marker", "cat a.go &", "PATH=/tmp/evil cat x", "LD_PRELOAD=x.so ls",
		"GOFLAGS=-toolexec=evil go build ./...", "go env -w GOFLAGS=x", "go build -o /usr/local/bin/x .",
		"git tag v1", "git tag -d v1", "git remote add x url", "hostname evil", "date -s now",
		"tree -o out.txt", "rg --pre evil x", "fd -x rm", "git log --output=f",
		"ls >&2 && touch x", "cat a 2>&1 > b", "(cd x && rm y)", "ls | cat > out",
	}
	for _, c := range yes {
		if !autoRun(c) {
			t.Errorf("%q should run without asking", c)
		}
	}
	for _, c := range no {
		if autoRun(c) {
			t.Errorf("%q must NOT run without asking", c)
		}
	}
}

// The allowlist admitted "find -delete" once; this proves an admitted
// command cannot delete a file in a fixture, for every entry of a
// representative set, by running them.
func TestAutoRunAdmittedCommandsDoNotMutate(t *testing.T) {
	sb, root := newTestSandbox(t, ModeReadOnly)
	victim := filepath.Join(root, "victim")
	must(t, os.WriteFile(victim, []byte("keep\n"), 0o644))
	cmds := []string{
		"find . -name victim", "sort victim", "cat victim", "grep keep victim", "ls -la",
		"git status", "date", "tree", "head victim", "wc -l victim",
	}
	for _, c := range cmds {
		if !autoRun(c) {
			continue
		}
		_, _, err := sb.Run(context.Background(), c, root, 5*time.Second, execMaxBytes)
		must(t, err)
		got, err := os.ReadFile(victim)
		if err != nil || string(got) != "keep\n" {
			t.Fatalf("%q changed the fixture: %q %v", c, got, err)
		}
	}
	entries, _ := os.ReadDir(root)
	if len(entries) != 1 {
		t.Fatalf("an admitted command created files: %v", entries)
	}
}

func TestExecPolicy(t *testing.T) {
	ctx := context.Background()
	sb, _ := newTestSandbox(t, ModeReadOnly)
	must(t, sb.Exec(ctx, "ls"))
	if err := sb.Exec(ctx, "rm x"); err == nil {
		t.Fatal("read-only must refuse")
	}
	sb.Mode = ModeWorkspace
	if err := sb.Exec(ctx, "rm x"); err == nil || !strings.Contains(err.Error(), "nobody is at the console") {
		t.Fatalf("headless Ask must deny, got %v", err)
	}
	asked := 0
	sb.Confirm = func(context.Context, string) bool { asked++; return asked > 1 }
	if err := sb.Exec(ctx, "rm x"); err == nil {
		t.Fatal("declined must refuse")
	}
	if err := sb.Exec(ctx, "rm x"); err != nil {
		t.Fatal("approved must run")
	}
	sb.Always("exec")
	if err := sb.Exec(ctx, "rm y"); err != nil || asked != 2 {
		t.Fatalf("always must skip the prompt: err=%v asked=%d", err, asked)
	}
	sb.Mode = ModeFull
	sb.Confirm = func(context.Context, string) bool { t.Fatal("full mode never asks"); return false }
	must(t, sb.Exec(ctx, "rm z"))
	child := sb.child(true)
	if child.Mode != ModeReadOnly {
		t.Fatal("readonly child must be read-only")
	}
	if sb.child(false).Mode != ModeFull {
		t.Fatal("child inherits the parent's mode")
	}
	// the child's prompts are labelled and its "always" is its own
	sb.Mode = ModeWorkspace
	var seen string
	sb.Confirm = func(_ context.Context, a string) bool { seen = a; return true }
	c := sb.child(false)
	must(t, c.Exec(ctx, "rm q"))
	if !strings.HasPrefix(seen, "[sub-agent] ") {
		t.Fatalf("child prompt not labelled: %q", seen)
	}
	sb.Always("exec")
	if c.always["exec"] {
		t.Fatal("always must not leak into the child")
	}
}

func TestRunScrubsSecretsAndKillsTree(t *testing.T) {
	t.Setenv("MY_API_KEY", "hunter2")
	t.Setenv(envPrefix+"MODEL", "x")
	sb, root := newTestSandbox(t, ModeFull)
	out, code, err := sb.Run(context.Background(), "env; echo exit-test; exit 3", root, 5*time.Second, execMaxBytes)
	if err != nil || code != 3 {
		t.Fatalf("code=%d err=%v", code, err)
	}
	if strings.Contains(out, "hunter2") || strings.Contains(out, envPrefix+"MODEL") || !strings.Contains(out, "MOTE=1") {
		t.Fatalf("env not scrubbed:\n%s", out)
	}
	start := time.Now()
	_, code, err = sb.Run(context.Background(), "sh -c 'sleep 30' & sleep 30", root, 300*time.Millisecond, execMaxBytes)
	if err != nil || code != 124 {
		t.Fatalf("timeout: code=%d err=%v", code, err)
	}
	if time.Since(start) > 5*time.Second {
		t.Fatal("timeout did not kill the process tree promptly")
	}
	if _, _, err := sb.Run(context.Background(), "cat", root, time.Second, execMaxBytes); err != nil {
		t.Fatalf("closed stdin must not block: %v", err)
	}
	// cancellation is distinct from timeout and reported as an error
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(100 * time.Millisecond); cancel() }()
	_, _, err = sb.Run(ctx, "sleep 10", root, 10*time.Second, execMaxBytes)
	if err == nil || !strings.Contains(err.Error(), "canceled") {
		t.Fatalf("cancel must surface: %v", err)
	}
}

// The output cap holds while the command runs: a command that prints
// megabytes never fills memory, and what comes back is head, a cut marker
// with the count, and tail.
func TestRunBoundsOutputWhileRunning(t *testing.T) {
	sb, root := newTestSandbox(t, ModeFull)
	out, code, err := sb.Run(context.Background(), "i=0; while [ $i -lt 200000 ]; do echo line-$i; i=$((i+1)); done", root, 30*time.Second, 3000)
	if err != nil || code != 0 {
		t.Fatalf("code=%d err=%v", code, err)
	}
	if len(out) > 3000+60 {
		t.Fatalf("output not bounded: %d bytes", len(out))
	}
	if !strings.HasPrefix(out, "line-0\n") || !strings.HasSuffix(out, "line-199999\n") || !strings.Contains(out, "bytes cut]") {
		t.Fatalf("head/tail shape wrong:\n%s", out)
	}
}

func TestCapWriter(t *testing.T) {
	w := newCapWriter(30)
	for i := 0; i < 20; i++ {
		fmt.Fprintf(w, "%05d\n", i) // 6 bytes each, 120 total
		if len(w.tail) > 2*(30-10) {
			t.Fatalf("tail buffer grew past its bound: %d", len(w.tail))
		}
	}
	out := w.String()
	if !strings.HasPrefix(out, "00000\n0000") || !strings.HasSuffix(out, "00018\n00019\n") || !strings.Contains(out, "[90 bytes cut]") {
		t.Fatalf("cap: %q", out)
	}
	w = newCapWriter(30)
	fmt.Fprint(w, "short")
	if w.String() != "short" {
		t.Fatalf("short output must pass through: %q", w.String())
	}
	w = newCapWriter(30)
	fmt.Fprint(w, strings.Repeat("a", 30))
	if w.String() != strings.Repeat("a", 30) {
		t.Fatalf("exactly max must pass through: %q", w.String())
	}
}

func TestScrubEnv(t *testing.T) {
	in := []string{"PATH=/bin", "OPENAI_API_KEY=x", "GITHUB_TOKEN=y", "MY_SECRET=z", "PASSWORD=p", "AWS_CREDENTIALS=c", "AUTH_HEADER=h", envPrefix + "HOME=/x", "HOME=/h", "LANG=C"}
	out := strings.Join(scrubEnv(in), " ")
	for _, bad := range []string{"OPENAI", "TOKEN", "SECRET", "PASSWORD", "CREDENTIALS", "AUTH_HEADER", envPrefix + "HOME"} {
		if strings.Contains(out, bad) {
			t.Errorf("%s leaked: %s", bad, out)
		}
	}
	for _, good := range []string{"PATH=/bin", "HOME=/h", "LANG=C", "MOTE=1"} {
		if !strings.Contains(out, good) {
			t.Errorf("%s missing: %s", good, out)
		}
	}
}

// --- the fence

// The fence argv is a pure function of mode, root and the machine, so the
// exact isolation exec gets is pinned here, not assumed.
func TestFenceArgv(t *testing.T) {
	home := t.TempDir()
	must(t, os.MkdirAll(filepath.Join(home, ".ssh"), 0o700))
	must(t, os.MkdirAll(filepath.Join(home, ".cache"), 0o755))
	must(t, os.WriteFile(filepath.Join(home, ".netrc"), []byte("machine x"), 0o600))
	root := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", "")
	t.Setenv("GOCACHE", "")
	t.Setenv("GOMODCACHE", "")
	t.Setenv("GOPATH", "")
	t.Setenv("CARGO_HOME", "")

	f := Fence{Kind: "bwrap", Home: home}
	got := strings.Join(f.wrap(root, ModeWorkspace, []string{"/bin/sh", "-c", "true"}), " ")
	for _, want := range []string{
		"bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --unshare-all --die-with-parent --share-net",
		"--bind " + filepath.Join(home, ".cache") + " " + filepath.Join(home, ".cache"),
		"--bind " + root + " " + root,
		"--tmpfs " + filepath.Join(home, ".ssh"),
		"--ro-bind /dev/null " + filepath.Join(home, ".netrc"),
		"-- /bin/sh -c true",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("bwrap argv lacks %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, filepath.Join(home, ".aws")) {
		t.Error("a credential dir that does not exist must not be masked (bwrap would fail)")
	}
	ro := strings.Join(f.wrap(root, ModeReadOnly, []string{"true"}), " ")
	if !strings.Contains(ro, "--ro-bind "+root+" "+root) || strings.Contains(ro, "--bind "+root) {
		t.Errorf("read-only must bind the root read-only:\n%s", ro)
	}
	nonet := strings.Join(Fence{Kind: "bwrap", Home: home, NoNet: true}.wrap(root, ModeFull, []string{"true"}), " ")
	if strings.Contains(nonet, "--share-net") {
		t.Errorf("-no-net must not share the network:\n%s", nonet)
	}
	// the tmpfs on /tmp comes before the root bind, or a root under /tmp would vanish
	args := f.wrap(root, ModeWorkspace, []string{"true"})
	tmpAt, rootAt := -1, -1
	for i, a := range args {
		if a == "--tmpfs" && args[i+1] == "/tmp" {
			tmpAt = i
		}
		if a == "--bind" && args[i+1] == root {
			rootAt = i
		}
	}
	if tmpAt < 0 || rootAt < tmpAt {
		t.Errorf("tmpfs /tmp (%d) must precede the root bind (%d)", tmpAt, rootAt)
	}

	s := Fence{Kind: "seatbelt", Home: home, NoNet: true}
	prof := s.profile(root, ModeReadOnly)
	for _, want := range []string{
		"(version 1)\n(allow default)\n(deny file-write*)\n",
		`(allow file-write* (subpath "/private/tmp"))`,
		`(deny file-write* (subpath "` + root + `"))`,
		`(deny file-read* (subpath "` + filepath.Join(home, ".ssh") + `"))`,
		`(deny file-read* (literal "` + filepath.Join(home, ".netrc") + `"))`,
		"(deny network*)",
	} {
		if !strings.Contains(prof, want) {
			t.Errorf("seatbelt profile lacks %q:\n%s", want, prof)
		}
	}
	// later rules win: the root deny must come after the tmp allow
	if strings.Index(prof, `(deny file-write* (subpath "`+root) < strings.Index(prof, "/private/var/folders") {
		t.Errorf("root deny must follow the tmp allows:\n%s", prof)
	}
	rw := s.profile(root, ModeWorkspace)
	if !strings.Contains(rw, `(allow file-write* (subpath "`+root+`"))`) || strings.Contains(rw, "(deny network*)") && !s.NoNet {
		t.Errorf("workspace profile wrong:\n%s", rw)
	}
	if none := (Fence{Kind: "none"}).wrap(root, ModeFull, []string{"/bin/sh", "-c", "x"}); strings.Join(none, " ") != "/bin/sh -c x" {
		t.Errorf("no fence must pass argv through: %v", none)
	}
}

// On CI the fence must be real, or every fence test below silently skips
// and the README's claims go unproven. Locally a machine without bwrap
// or sandbox-exec is allowed to run with fence=none.
func TestFenceIsRealOnCI(t *testing.T) {
	if os.Getenv("CI") == "" || (runtime.GOOS != "linux" && runtime.GOOS != "darwin") {
		t.Skip("not CI")
	}
	f := detectFence(t.TempDir(), userHome(), false)
	if f.Kind == "none" {
		t.Fatalf("CI on %s must have a working fence (install bubblewrap on linux; sandbox-exec ships with macOS)", runtime.GOOS)
	}
}

func realFence(t *testing.T, mode Mode, noNet bool) (*Sandbox, string) {
	t.Helper()
	sb, root := newTestSandbox(t, mode)
	sb.Fence = detectFence(root, userHome(), noNet)
	if sb.Fence.Kind == "none" {
		t.Skip("no fence on this machine")
	}
	return sb, root
}

// The guarantees the README makes about the fence, proven by running
// commands through it and looking at the disk afterwards.
func TestFenceWorkspaceWritesStayInRoot(t *testing.T) {
	sb, root := realFence(t, ModeWorkspace, false)
	out, code, err := sb.Run(context.Background(), "echo hi > inside.txt && cat inside.txt", root, 10*time.Second, execMaxBytes)
	if err != nil || code != 0 || !strings.Contains(out, "hi") {
		t.Fatalf("write inside root must work: code=%d err=%v out=%s", code, err, out)
	}
	if _, err := os.Stat(filepath.Join(root, "inside.txt")); err != nil {
		t.Fatalf("file must exist on the host: %v", err)
	}
	probe := filepath.Join(userHome(), fmt.Sprintf(".mote-fence-probe-%d", time.Now().UnixNano()))
	defer os.Remove(probe)
	_, code, err = sb.Run(context.Background(), "echo x > "+probe, root, 10*time.Second, execMaxBytes)
	must(t, err)
	if code == 0 {
		t.Fatalf("a write outside the root succeeded (exit %d)", code)
	}
	if _, err := os.Stat(probe); err == nil {
		t.Fatalf("the write outside the root landed on the host")
	}
}

func TestFenceReadOnlyRootCannotBeWritten(t *testing.T) {
	sb, root := realFence(t, ModeReadOnly, false)
	must(t, os.WriteFile(filepath.Join(root, "victim"), []byte("keep\n"), 0o644))
	_, code, err := sb.Run(context.Background(), "echo x > new.txt; rm -f victim; echo x >> victim", root, 10*time.Second, execMaxBytes)
	must(t, err)
	if code == 0 {
		t.Fatal("writes in a read-only root must fail")
	}
	if got, _ := os.ReadFile(filepath.Join(root, "victim")); string(got) != "keep\n" {
		t.Fatalf("victim changed: %q", got)
	}
	if _, err := os.Stat(filepath.Join(root, "new.txt")); err == nil {
		t.Fatal("new.txt was created in a read-only root")
	}
	// reading still works, and so does a scratch write to tmp
	out, code, err := sb.Run(context.Background(), "cat victim && echo scratch > /tmp/mote-scratch && cat /tmp/mote-scratch", root, 10*time.Second, execMaxBytes)
	if err != nil || code != 0 || !strings.Contains(out, "keep") || !strings.Contains(out, "scratch") {
		t.Fatalf("read-only must still read the root and write to tmp: code=%d err=%v out=%s", code, err, out)
	}
}

func TestFenceNoNetBlocksLoopback(t *testing.T) {
	if _, err := exec.LookPath("curl"); err != nil {
		t.Skip("curl not installed")
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "reachable") }))
	defer srv.Close()
	open, root := realFence(t, ModeFull, false)
	out, code, _ := open.Run(context.Background(), "curl -s -m 3 "+srv.URL, root, 10*time.Second, execMaxBytes)
	if code != 0 || !strings.Contains(out, "reachable") {
		t.Fatalf("with the network shared the server must be reachable: code=%d out=%s", code, out)
	}
	closed, _ := realFence(t, ModeFull, true)
	out, code, _ = closed.Run(context.Background(), "curl -s -m 3 "+srv.URL, root, 10*time.Second, execMaxBytes)
	if code == 0 || strings.Contains(out, "reachable") {
		t.Fatalf("-no-net must cut the network: code=%d out=%s", code, out)
	}
}

func TestFenceMasksCredentialDirs(t *testing.T) {
	sb, root := realFence(t, ModeFull, false)
	ssh := filepath.Join(userHome(), ".ssh")
	if !isDir(ssh) {
		t.Skip("no ~/.ssh to mask")
	}
	out, _, err := sb.Run(context.Background(), "ls -A "+ssh+" 2>&1; cat "+ssh+"/* 2>/dev/null | head -c 100", root, 10*time.Second, execMaxBytes)
	must(t, err)
	if strings.Contains(out, "id_") || strings.Contains(out, "PRIVATE") || strings.Contains(out, "known_hosts") || strings.Contains(out, "config") {
		t.Fatalf("~/.ssh visible through the fence:\n%s", out)
	}
}

func TestFenceTimeoutKillsTree(t *testing.T) {
	sb, root := realFence(t, ModeFull, false)
	start := time.Now()
	_, code, err := sb.Run(context.Background(), "sh -c 'sleep 30' & sleep 30", root, 300*time.Millisecond, execMaxBytes)
	if err != nil || code != 124 {
		t.Fatalf("timeout inside the fence: code=%d err=%v", code, err)
	}
	if time.Since(start) > 5*time.Second {
		t.Fatal("the fenced tree outlived its timeout")
	}
}
