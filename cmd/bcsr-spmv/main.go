// bcsr-spmv 是 CSR / BCSR 构建与 SpMV 的命令行入口。
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	bcsrspmv "example.com/bcsrspmv"
)

func main() {
	entriesText := flag.String("entries", "", "非零元素，逗号分隔的 行-列:值")
	spmvText := flag.String("spmv", "", "向量 x，逗号分隔")
	block := flag.Int("block", 0, "额外做 B×B 分块压缩")
	flag.Parse()

	entries, err := bcsrspmv.ParseEntries(*entriesText)
	if err != nil {
		fail(err)
	}
	rows, cols := 0, 0
	for _, entry := range entries {
		if entry.Row+1 > rows {
			rows = entry.Row + 1
		}
		if entry.Col+1 > cols {
			cols = entry.Col + 1
		}
	}
	matrix := bcsrspmv.Build(rows, cols, entries)
	fmt.Printf("rows=%d cols=%d nnz=%d offsets=%v indices=%s\n",
		matrix.Rows, matrix.Cols, len(matrix.Indices),
		matrix.Offsets, joinInts(matrix.Indices))

	if *block > 0 {
		blocked, err := matrix.Blocks(*block)
		if err != nil {
			fail(err)
		}
		fmt.Printf("block=%d blocks=%d index_ints=%d\n",
			blocked.Size, len(blocked.Indices), len(blocked.Indices))
	}

	if *spmvText != "" {
		x, err := parseVector(*spmvText)
		if err != nil {
			fail(err)
		}
		y, err := matrix.SpMV(x)
		if err != nil {
			fail(err)
		}
		fmt.Printf("y=%s\n", joinFloats(y))
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}

func parseVector(text string) ([]float64, error) {
	parts := strings.Split(text, ",")
	vector := make([]float64, 0, len(parts))
	for index, part := range parts {
		value, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("向量第 %d 个分量 %q 不是数字", index+1, strings.TrimSpace(part))
		}
		vector = append(vector, value)
	}
	return vector, nil
}

func joinInts(values []int) string {
	parts := make([]string, len(values))
	for index, value := range values {
		parts[index] = strconv.Itoa(value)
	}
	return strings.Join(parts, ",")
}

func joinFloats(values []float64) string {
	parts := make([]string, len(values))
	for index, value := range values {
		parts[index] = strconv.FormatFloat(value, 'g', -1, 64)
	}
	return strings.Join(parts, ",")
}
