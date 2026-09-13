#!/bin/sh
cd "$1" && grep -q 'return a + b' math.go && grep -q 'return a \* b' math.go && go vet ./... && grep -q '"outcome":"done"' "$2" && test "$(ls .mote/sessions/*.jsonl | grep -vc sub)" -eq 1
