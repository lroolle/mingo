#!/bin/sh
# First run is capped at two model requests, so it ends in a budget outcome
# (exit 3) mid-task. The second run resumes the same session with no cap.
# The JSON of the second run is the result; the first is kept in stderr.
"$MOTE" -provider "$PROVIDER" -json -quiet -cwd "$DIR" -max-requests 2 -p "$(cat "$(dirname "$0")/prompt.txt")" >&2
echo "first run exit=$? (expected 3)" >&2
exec "$MOTE" -provider "$PROVIDER" -json -quiet -cwd "$DIR" -resume last -p "Continue where you left off and finish the task."
