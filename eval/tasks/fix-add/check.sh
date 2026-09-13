#!/bin/sh
cd "$1" && grep -q 'return a + b' add.go && go vet ./... && grep -q '"outcome":"done"' "$2" && grep -q '"written":\["add.go"\]' "$2"
