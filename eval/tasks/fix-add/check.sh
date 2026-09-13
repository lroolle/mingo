#!/bin/sh
# Behaviour, not text: a hidden test calls Add and must pass. A comment
# saying "return a + b" over a body that subtracts fails here. The receipt
# must name the file, and only that file may have changed.
cd "$1" || exit 1
cat > zz_check_test.go <<'GO'
package main

import "testing"

func TestAddBehaves(t *testing.T) {
	for _, c := range []struct{ a, b, want int }{{2, 3, 5}, {0, 0, 0}, {-1, 1, 0}, {10, -4, 6}} {
		if got := Add(c.a, c.b); got != c.want {
			t.Fatalf("Add(%d,%d)=%d want %d", c.a, c.b, got, c.want)
		}
	}
}
GO
go vet ./... >/dev/null 2>&1 || exit 1
go test ./... >/dev/null 2>&1 || exit 1
rm -f zz_check_test.go
grep -q '"outcome":"done"' "$2" || exit 1
grep -q '"written":\["add.go"\]' "$2" || exit 1
# nothing else appeared or changed
test "$(ls -A | grep -v '^\.min$' | sort | tr '\n' ' ')" = "add.go go.mod "
