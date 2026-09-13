---
name: go-verify
description: how to verify a change to mote.go before reporting it done
---
1. `gofmt -l .` must print nothing.
2. `go vet ./...` must be clean.
3. `go test -race ./...` must pass; the suite drives the loop against a fake
   provider and runs the fence tests for real when the machine has one.
4. If the change touched the wire (request shape, SSE parsing, reasoning
   replay), say so: the suite cannot prove the live API, only a real run can.
