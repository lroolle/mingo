.PHONY: build test race vet bench prompt local eval

build:
	go build -o mingo .

test:
	go test ./...

# The suite under the race detector, with the fence tests running for real
# when this machine has bwrap or sandbox-exec. CI runs exactly this.
race:
	go test -race -count=1 ./...

vet:
	go vet ./... && test -z "$$(gofmt -l .)"

# Go benchmarks of the pure paths the loop runs every round.
bench:
	go test -run '^$$' -bench . -benchmem ./...

prompt: build
	./mingo -show-prompt | wc -c

# A local backend: llama-server serving a GGUF on loopback. The flags are
# the ones measured on a 10-core arm64 CPU with Spark-X2.5-4B; any
# OpenAI-compatible server works, mingo reads the context size from /props.
MODEL ?= $(HOME)/models/Spark-X2.5-4B-Q4_K_M.gguf
local:
	llama-server -m $(MODEL) --host 127.0.0.1 --port 8080 -c 32768 -fa on --jinja \
	  --cache-reuse 256 --parallel 1 --reasoning-format deepseek

# The live evaluation: every task under eval/tasks, N runs each, against
# PROVIDER. Spends real requests. See eval/README.md.
N ?= 1
PROVIDER ?= local
eval: build
	N=$(N) PROVIDER=$(PROVIDER) ./eval/run.sh $(TASK)
