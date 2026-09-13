#!/bin/sh
set -e
cd "$1"
printf 'module smoke\n\ngo 1.26\n' > go.mod
cat > strutil.go <<'GO'
package smoke

// Reverse returns s with its runes in reverse order.
func Reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}
GO
# the checker compares against this; it lives beside the root, outside it
sha256sum strutil.go | cut -c1-64 > "$1.strutil.sha256"
