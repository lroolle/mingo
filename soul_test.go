package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

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
	must(t, os.MkdirAll(filepath.Join(root, runtimeDir, "skills", "deploy"), 0o755))
	must(t, os.WriteFile(filepath.Join(root, runtimeDir, "skills", "deploy", "SKILL.md"), []byte("---\nname: deploy\ndescription: ship it\n---\nsteps\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(root, runtimeDir, "skills", "deploy", "notes.md"), []byte("---\nname: notes\ndescription: not a skill\n---\n"), 0o644))
	must(t, os.MkdirAll(filepath.Join(home, "skills"), 0o755))
	must(t, os.WriteFile(filepath.Join(home, "skills", "review.md"), []byte("---\ndescription: review code\n---\nlook\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(home, "skills", "deploy.md"), []byte("---\ndescription: shadowed by project\n---\n"), 0o644))
	skills := loadSkills(filepath.Join(root, runtimeDir, "skills"), filepath.Join(home, "skills"))
	if len(skills) != 2 || skills[0].Name != "deploy" || skills[0].Description != "ship it" || skills[1].Name != "review" {
		t.Fatalf("skills: %+v", skills)
	}
	must(t, os.WriteFile(filepath.Join(home, "SOUL.md"), []byte("user layer"), 0o644))
	must(t, os.WriteFile(filepath.Join(root, runtimeDir, "SOUL.md"), []byte("project layer"), 0o644))
	soul := readSoul(home, root)
	if !strings.HasPrefix(soul, defaultSoul) || !strings.Contains(soul, "user layer\n\nproject layer") {
		t.Fatalf("soul layers wrong:\n%s", soul)
	}
	cfg := &Config{Root: root, Mode: ModeWorkspace}
	sys := buildSystemPrompt(cfg, soul, skills, false)
	for _, want := range []string{"# Soul", "- deploy: ship it", "- review: review code", "sandbox: workspace", "You talk to one person"} {
		if !strings.Contains(sys, want) {
			t.Fatalf("system prompt lacks %q", want)
		}
	}
	sub := buildSystemPrompt(cfg, soul, nil, true)
	if !strings.Contains(sub, "sub-agent") || strings.Contains(sub, "# Skills") {
		t.Fatalf("sub-agent prompt wrong:\n%s", sub)
	}
	must(t, os.WriteFile(filepath.Join(root, "AGENTS.md"), []byte("project rule"), 0o644))
	if sys := buildSystemPrompt(cfg, soul, nil, false); !strings.Contains(sys, "# Project instructions (AGENTS.md)\nproject rule") {
		t.Fatalf("AGENTS.md not included:\n%s", sys)
	}
}

// The system prompt is the prompt-cache prefix: the same inputs must yield
// the same bytes, and only the inputs that matter may change it.
func TestSystemPromptIsStable(t *testing.T) {
	cfg := &Config{Root: t.TempDir(), Mode: ModeWorkspace}
	a := buildSystemPrompt(cfg, "soul", nil, false)
	b := buildSystemPrompt(cfg, "soul", nil, false)
	if a != b {
		t.Fatal("prompt differs between identical calls")
	}
	cfg.Mode = ModeReadOnly
	if buildSystemPrompt(cfg, "soul", nil, false) == a {
		t.Fatal("mode must be part of the prompt")
	}
}
