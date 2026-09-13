# Live evaluation

`make eval` runs every task here against a real model and reports pass
rates, wall time and token counts. It spends real requests; nothing in CI
runs it. The unit suite proves the harness; this proves a model can
deliver through it.

    make eval PROVIDER=local N=3        # a llama-server on loopback
    make eval PROVIDER=deepseek         # one run per task
    make eval TASK=fix-add N=5          # one task
    KEEP=1 ./eval/run.sh rename-symbol  # keep the directory of a failed run

Each task is a directory: `setup.sh` builds the project in a fresh root,
`prompt.txt` is the user's message, `check.sh` judges the root and the JSON
result on disk (not the model's prose), `args` adds flags, and `run.sh`
replaces the generic invocation when a task needs more than one run.

| task | proves | mode |
| --- | --- | --- |
| fix-add | read, edit, verify, report; the receipt names the file | workspace |
| add-test | writing a new file that compiles and passes | workspace |
| report-only | a read-only run answers a question and writes nothing | read-only |
| rename-symbol | one change carried across two files, verified by a build | workspace |
| resume | a run cut by a request cap is resumed and finished in the same session | workspace |

Every check reads the file system and the JSON outcome. A model that says
"fixed" without editing fails `fix-add` because `add.go` still subtracts
and the result's `written` list is empty.

Results land in `eval/results/*.jsonl` (ignored by git), one line per run.
