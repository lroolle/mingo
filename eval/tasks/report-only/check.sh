#!/bin/sh
# 6 exported: Alpha Beta Delta Zeta Eta Theta. The tree must be byte-for-byte
# what setup made (the manifest lives beside the root, where the model
# cannot reach), the receipt must be empty, the answer must say 6.
cd "$1" || exit 1
grep -q '"outcome":"done"' "$2" || exit 1
grep -q '"written":null' "$2" || exit 1
find . -path ./.mingo -prune -o -type f -print | sort | xargs sha256sum > "$1.after"
cmp -s "$1.manifest" "$1.after" || { echo "workspace changed" >&2; exit 1; }
sed -n 's/.*"answer":"\([^"]*\)".*/\1/p' "$2" | grep -Eq '(^|[^0-9])6([^0-9]|$)'
