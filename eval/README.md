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
| add-test | writing a new file that compiles and passes; `go test` is not allowlisted, so the run is `-yolo` as an unattended person's would be | full |
| report-only | a read-only run answers a question and writes nothing | read-only |
| rename-symbol | one change carried across two files, verified by a build | workspace |
| resume | a run cut by a request cap is resumed and finished in the same session | workspace |

Every check judges the work, not the model's account of it, and not the
runtime's either: `fix-add` and `resume` drop a hidden test into the root
and call the functions; `add-test` runs the generated test against the
correct code and then against two mutants (a `Reverse` that returns its
input, and one that reverses bytes), and a test that passes a mutant is
vacuous; `report-only` hashes the tree at setup into a manifest beside
the root, where the model cannot reach, and compares afterwards;
`rename-symbol` builds and runs the program; `resume` refuses a first run
that did not end in exit 3, because a run that finished has nothing to
resume. The runner's own result and log live beside the root too
(`<root>.result.json`, `<root>.stderr.log`), so the root is exactly what
setup made plus what the model did.

A second review found the previous graders accepting a comment that said
`return a + b` over a body that subtracted, an empty `TestReverse`, and a
receipt that said nothing was written over a tree that had changed. The
runtime's `outcome: done` means the model answered within its budgets; the
grader is what turns that into pass or fail.

Results land in `eval/results/*.jsonl` (ignored by git), one line per run.
