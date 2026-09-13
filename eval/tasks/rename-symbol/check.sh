#!/bin/sh
# The rename is complete (no OldName anywhere, both call sites moved, the
# doc comment follows), the program still builds and runs the same.
cd "$1" || exit 1
! grep -rq 'OldName' --include='*.go' . || exit 1
grep -q '^// Greet ' greet.go || exit 1
grep -q 'func Greet(' greet.go || exit 1
test "$(grep -c 'Greet(' main.go)" -eq 2 || exit 1
go build -o /dev/null ./... || exit 1
test "$(go run . 2>/dev/null)" = "hi world
hi again" || exit 1
grep -q '"outcome":"done"' "$2"
