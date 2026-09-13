#!/bin/sh
set -e
cd "$1"
printf 'module smoke\n\ngo 1.26\n' > go.mod
printf 'package main\n\n// OldName greets.\nfunc OldName(who string) string { return "hi " + who }\n' > greet.go
printf 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println(OldName("world"))\n\tfmt.Println(OldName("again"))\n}\n' > main.go
