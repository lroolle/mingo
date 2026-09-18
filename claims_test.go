package main

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// The default model each provider publishes, asserted against the providers
// table, so a README or site that drifts from the code turns CI red. Every
// occurrence is checked, not the first.

func readFile(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestReadmeProviderDefaultsMatchCode(t *testing.T) {
	doc := readFile(t, "README.md")
	_, table, ok := strings.Cut(doc, "## Providers")
	if !ok {
		t.Fatal("README has no Providers section")
	}
	header := regexp.MustCompile(`(?m)^\| \| (.+) \|$`).FindStringSubmatch(table)
	row := regexp.MustCompile(`(?m)^\| default model \| (.+) \|$`).FindAllStringSubmatch(doc, -1)
	if header == nil || len(row) != 1 {
		t.Fatalf("providers table not found (header %v, %d default-model rows)", header != nil, len(row))
	}
	names := strings.Split(header[1], " | ")
	cells := strings.Split(row[0][1], " | ")
	if len(names) != len(cells) {
		t.Fatalf("%d providers, %d default-model cells", len(names), len(cells))
	}
	checked := 0
	for i, name := range names {
		name = strings.TrimSuffix(name, " (default)")
		p, ok := providers[name]
		if !ok {
			t.Errorf("README lists provider %q the code does not have", name)
			continue
		}
		first := regexp.MustCompile("`([^`]+)`").FindStringSubmatch(cells[i])
		if first == nil {
			continue // local: "whatever is loaded"
		}
		checked++
		if first[1] != p.DefaultModel {
			t.Errorf("README says %s defaults to %q; the code says %q", name, first[1], p.DefaultModel)
		}
	}
	if checked < 3 {
		t.Errorf("only %d default-model cells checked", checked)
	}
	// The sample session line shows the default provider's default model.
	for _, m := range regexp.MustCompile(`model=deepseek:(\S+)`).FindAllStringSubmatch(doc, -1) {
		if m[1] != providers["deepseek"].DefaultModel {
			t.Errorf("README sample shows model=deepseek:%s; the default is %s", m[1], providers["deepseek"].DefaultModel)
		}
	}
}

func TestSiteProviderDefaultsMatchCode(t *testing.T) {
	rows := regexp.MustCompile(`<tr><td>(\w+)</td><td>[^<]*</td><td>([^<]+)</td>`).
		FindAllStringSubmatch(readFile(t, "docs/index.html"), -1)
	if len(rows) != len(providers) {
		t.Fatalf("site lists %d providers, the code has %d", len(rows), len(providers))
	}
	for _, r := range rows {
		p, ok := providers[r[1]]
		if !ok {
			t.Errorf("site lists provider %q the code does not have", r[1])
			continue
		}
		if r[1] == "local" {
			continue // "whatever is loaded"
		}
		if r[2] != p.DefaultModel {
			t.Errorf("site says %s defaults to %q; the code says %q", r[1], r[2], p.DefaultModel)
		}
	}
}
