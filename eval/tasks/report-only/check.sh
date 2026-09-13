#!/bin/sh
# 6 exported: Alpha Beta Delta Zeta Eta Theta. Nothing written, outcome done.
cd "$1" && grep -q '"outcome":"done"' "$2" && grep -q '"written":null' "$2" && sed -n 's/.*"answer":"\([^"]*\)".*/\1/p' "$2" | grep -Eq '(^|[^0-9])6([^0-9]|$)'
