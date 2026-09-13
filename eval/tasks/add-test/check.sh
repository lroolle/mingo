#!/bin/sh
cd "$1" && test -f strutil_test.go && grep -q 'func TestReverse' strutil_test.go && go test ./... >/dev/null 2>&1 && grep -q '"outcome":"done"' "$2"
