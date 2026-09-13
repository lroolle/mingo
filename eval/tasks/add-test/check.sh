#!/bin/sh
# The generated test must pass against the correct Reverse and FAIL against
# a broken one: a TestReverse with no assertions passes both and is refused
# here. strutil.go itself must be untouched.
cd "$1" || exit 1
test -f strutil_test.go || exit 1
grep -q 'func TestReverse' strutil_test.go || exit 1
grep -q '"outcome":"done"' "$2" || exit 1
test "$(sha256sum strutil.go | cut -c1-64)" = "$(cat "$1.strutil.sha256")" || exit 1
go test ./... >/dev/null 2>&1 || exit 1
# mutate: Reverse returns its input unchanged; the test must catch it
cp strutil.go .strutil.orig
cat > strutil.go <<'GO'
package smoke

// Reverse returns s with its runes in reverse order.
func Reverse(s string) string {
	return s
}
GO
if go test ./... >/dev/null 2>&1; then
	mv .strutil.orig strutil.go
	echo "vacuous test: passes against a broken Reverse" >&2
	exit 1
fi
mv .strutil.orig strutil.go
# and a second mutation: bytes instead of runes, which only a multi-byte case catches
cat > strutil.go <<'GO'
package smoke

// Reverse returns s with its runes in reverse order.
func Reverse(s string) string {
	b := []byte(s)
	for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}
	return string(b)
}
GO
if go test ./... >/dev/null 2>&1; then
	mv .strutil.orig strutil.go 2>/dev/null
	echo "test misses the multi-byte case" >&2
	exit 1
fi
exit 0
