// Command lm15-vet is the lm15 vet shim: newline-delimited JSON on
// stdin/stdout per lm15-contract/harness/PROTOCOL.md. It never opens a
// network connection.
package main

import (
	"bufio"
	"os"
	"strings"

	lm15 "github.com/lm15-dev/lm15-go"
)

func main() {
	in := bufio.NewReaderSize(os.Stdin, 16<<20)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	for {
		line, err := in.ReadBytes('\n')
		if len(strings.TrimSpace(string(line))) > 0 {
			out.Write(lm15.HandleVetLine(line))
			out.WriteByte('\n')
			out.Flush()
		}
		if err != nil {
			return
		}
	}
}
