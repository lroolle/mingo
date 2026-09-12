---
name: go-verify
description: how to verify a change to spark.go before reporting it done
---
1. `gofmt -l .` must print nothing.
2. `go vet ./...` must be clean.
3. `go test ./...` must pass; the suite drives the loop against a fake provider.
4. If the change touched the wire (request shape, SSE parsing, reasoning
   replay), say so: the suite cannot prove the live API, only a real run can.
