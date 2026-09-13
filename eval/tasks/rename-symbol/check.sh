#!/bin/sh
cd "$1" && ! grep -rq 'OldName' --include='*.go' . && grep -q 'func Greet(' greet.go && grep -c 'Greet(' main.go | grep -q '^2$' && go build ./... && grep -q '"outcome":"done"' "$2"
