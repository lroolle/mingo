#!/bin/sh
set -e
cd "$1"
printf 'module smoke\n\ngo 1.26\n' > go.mod
printf 'package main\n\nfunc Add(a, b int) int { return a - b }\n\nfunc main() {}\n' > add.go
