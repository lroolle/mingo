.PHONY: build test vet prompt local bench

build:
	go build -o spark .

test:
	go test ./...

vet:
	go vet ./... && test -z "$$(gofmt -l .)"

prompt: build
	./spark -show-prompt | wc -c

# A local backend: llama-server from the XHToken llama.cpp fork serving
# Spark-X2.5-4B. Flags measured on a 10-core arm64 CPU, see README.
MODEL ?= $(HOME)/models/Spark-X2.5-4B-Q4_K_M.gguf
local:
	llama-server -m $(MODEL) --host 127.0.0.1 --port 8080 -c 32768 -fa on --jinja \
	  --cache-reuse 256 --parallel 1 --reasoning-format deepseek

# Reproduce the README's local-model table: N runs of the fix-a-bug task
# against whatever -provider local is serving. Prints fixed/wall/tokens per run.
N ?= 3
THINK ?= high
bench: build
	@for i in $$(seq 1 $(N)); do \
	  d=$$(mktemp -d); printf 'package main\n\nfunc Add(a, b int) int { return a - b }\n' > $$d/add.go; printf 'module smoke\n\ngo 1.26\n' > $$d/go.mod; \
	  s=$$(date +%s); out=$$(./spark -provider local -yolo -think $(THINK) -quiet -cwd $$d -p "Add has a bug. Fix it, verify with go vet, and tell me in two sentences." 2>&1 | tail -1); \
	  f=no; grep -q 'a + b' $$d/add.go && f=yes; echo "run=$$i think=$(THINK) fixed=$$f wall=$$(( $$(date +%s) - s ))s $$out"; rm -rf $$d; \
	done
