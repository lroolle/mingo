#!/bin/sh
set -e
cd "$1"
printf 'module smoke\n\ngo 1.26\n' > go.mod
printf 'package smoke\n\nfunc Alpha() {}\nfunc Beta() {}\nfunc gamma() {}\n' > a.go
printf 'package smoke\n\nfunc Delta() {}\nfunc epsilon() {}\nfunc Zeta() {}\nfunc Eta() {}\n' > b.go
printf 'package smoke\n\nfunc Theta() {}\nfunc iota2() {}\n' > c.go
find . -type f -print | sort | xargs sha256sum > "$1.manifest"
