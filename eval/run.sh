#!/bin/sh
# Live evaluation of mote against a real model. Every task under eval/tasks
# is set up in a fresh directory, run headless with -json, and judged by
# its own check.sh. Spends real requests: N runs per task, PROVIDER picks
# the backend. Results go to eval/results/<timestamp>.jsonl.
#
#   PROVIDER=local N=3 ./eval/run.sh            # every task
#   PROVIDER=deepseek ./eval/run.sh fix-add     # one task
#
# A task directory holds:
#   setup.sh    creates the project in $1 (the fresh root)
#   prompt.txt  the user prompt
#   check.sh    exits 0 when $1 (the root) shows the task was done;
#               $2 is the JSON result mote printed
#   args        optional extra flags for mote (one line)
#   run.sh      optional: replaces the generic run; gets MOTE, DIR, PROVIDER
set -u
cd "$(dirname "$0")/.."
MOTE=${MOTE:-./mote}
PROVIDER=${PROVIDER:-local}
N=${N:-1}
export MOTE PROVIDER
mkdir -p eval/results
out="eval/results/$(date -u +%Y%m%dT%H%M%SZ)-$PROVIDER.jsonl"
tasks=${1:-$(ls eval/tasks)}
total=0; passed=0
for task in $tasks; do
  tdir="eval/tasks/$task"
  [ -d "$tdir" ] || { echo "no task $task" >&2; exit 2; }
  ok=0
  i=1
  while [ "$i" -le "$N" ]; do
    dir=$(mktemp -d)
    sh "$tdir/setup.sh" "$dir" || { echo "$task: setup failed" >&2; exit 2; }
    args=""; [ -f "$tdir/args" ] && args=$(cat "$tdir/args")
    start=$(date +%s)
    if [ -f "$tdir/run.sh" ]; then
      DIR="$dir" sh "$tdir/run.sh" > "$dir/.result.json" 2> "$dir/.stderr.log"; code=$?
    else
      # shellcheck disable=SC2086
      "$MOTE" -provider "$PROVIDER" -json -quiet -cwd "$dir" $args -p "$(cat "$tdir/prompt.txt")" > "$dir/.result.json" 2> "$dir/.stderr.log"; code=$?
    fi
    wall=$(( $(date +%s) - start ))
    res=$(cat "$dir/.result.json")
    outcome=$(printf '%s' "$res" | sed -n 's/.*"outcome":"\([a-z]*\)".*/\1/p')
    prompt=$(printf '%s' "$res" | sed -n 's/.*"prompt":\([0-9]*\).*/\1/p')
    cached=$(printf '%s' "$res" | sed -n 's/.*"cached":\([0-9]*\).*/\1/p')
    completion=$(printf '%s' "$res" | sed -n 's/.*"completion":\([0-9]*\).*/\1/p')
    requests=$(printf '%s' "$res" | sed -n 's/.*"requests":\([0-9]*\).*/\1/p')
    pass=no
    if sh "$tdir/check.sh" "$dir" "$dir/.result.json" >/dev/null 2>&1; then pass=yes; ok=$((ok+1)); fi
    printf '%-14s run=%d pass=%-3s outcome=%-9s exit=%-3s wall=%4ss requests=%-3s prompt=%-6s cached=%-6s completion=%s\n' \
      "$task" "$i" "$pass" "${outcome:-none}" "$code" "$wall" "${requests:-0}" "${prompt:-0}" "${cached:-0}" "${completion:-0}"
    printf '{"task":"%s","run":%d,"provider":"%s","pass":%s,"outcome":"%s","exit":%d,"wall_s":%d,"requests":%s,"prompt":%s,"cached":%s,"completion":%s}\n' \
      "$task" "$i" "$PROVIDER" "$( [ $pass = yes ] && echo true || echo false )" "${outcome:-none}" "$code" "$wall" "${requests:-0}" "${prompt:-0}" "${cached:-0}" "${completion:-0}" >> "$out"
    if [ "$pass" = no ] && [ -n "${KEEP:-}" ]; then echo "  kept $dir" ; else rm -rf "$dir"; fi
    i=$((i+1))
  done
  total=$((total+N)); passed=$((passed+ok))
  echo "$task: $ok of $N"
done
echo "total: $passed of $total  ($out)"
[ "$passed" -eq "$total" ]
