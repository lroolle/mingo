#!/bin/sh
# First run is capped at two model requests, so it must end in a budget
# outcome (exit 3) mid-task; a first run that finishes is not a resume test
# and fails here. The second run resumes the same session with no cap. The
# JSON of the second run is the result; the first is kept in stderr.
"$MINGO" -provider "$PROVIDER" -json -quiet -cwd "$DIR" -max-requests 2 -p "$(cat "$(dirname "$0")/prompt.txt")" >&2
code=$?
echo "first run exit=$code (expected 3)" >&2
if [ "$code" -ne 3 ]; then
	echo "{\"outcome\":\"eval-error\",\"error\":\"first run exited $code, not 3; nothing to resume\"}"
	exit 1
fi
exec "$MINGO" -provider "$PROVIDER" -json -quiet -cwd "$DIR" -resume last -p "Continue where you left off and finish the task."
