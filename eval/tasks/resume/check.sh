#!/bin/sh
# Both fixes present and behaving, one top-level session (the resume
# continued it rather than starting over), outcome done.
cd "$1" || exit 1
cat > zz_check_test.go <<'GO'
package main

import "testing"

func TestBoth(t *testing.T) {
	if Add(2, 3) != 5 || Add(-1, 1) != 0 {
		t.Fatal("Add")
	}
	if Mul(2, 3) != 6 || Mul(4, 0) != 0 {
		t.Fatal("Mul")
	}
}
GO
go vet ./... >/dev/null 2>&1 || exit 1
go test ./... >/dev/null 2>&1 || exit 1
rm -f zz_check_test.go
grep -q '"outcome":"done"' "$2" || exit 1
test "$(ls .mingo/sessions/*.jsonl | grep -vc sub)" -eq 1
