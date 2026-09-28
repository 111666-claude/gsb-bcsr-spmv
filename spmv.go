// Package bcsrspmv 把稀疏矩阵编成 CSR / BCSR 并做 SpMV，只用标准库。
package bcsrspmv

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

var (
	// ErrEntry 表示边表格式不对。
	ErrEntry = errors.New("bcsrspmv: 非零元素格式不对")
	// ErrShape 表示向量长度与矩阵列数不匹配。
	ErrShape = errors.New("bcsrspmv: 向量长度与矩阵列数不匹配")
	// ErrBlock 表示分块大小不合法。
	ErrBlock = errors.New("bcsrspmv: 分块大小必须是正整数")
)

// Entry 是边表里的一条非零元素。
type Entry struct {
	Row   int
	Col   int
	Value float64
}

// CSR 是压缩稀疏行结构。
type CSR struct {
	Rows    int
	Cols    int
	Offsets []int
	Indices []int
	Values  []float64
}

// ParseEntries 解析 "0-1:2,1-2:3"，出错时报第几条（从 1 开始数）。
func ParseEntries(text string) ([]Entry, error) {
	entries := []Entry{}
	for index, item := range strings.Split(text, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		position, rawValue, found := strings.Cut(item, ":")
		if !found {
			return nil, fmt.Errorf("%w：第 %d 条 %q 缺少 ':'", ErrEntry, index+1, item)
		}
		rowText, colText, found := strings.Cut(position, "-")
		if !found {
			return nil, fmt.Errorf("%w：第 %d 条 %q 缺少 '-'", ErrEntry, index+1, item)
		}
		row, errRow := strconv.Atoi(strings.TrimSpace(rowText))
		col, errCol := strconv.Atoi(strings.TrimSpace(colText))
		value, errValue := strconv.ParseFloat(strings.TrimSpace(rawValue), 64)
		if errRow != nil || errCol != nil || errValue != nil {
			return nil, fmt.Errorf("%w：第 %d 条 %q 含有非数字", ErrEntry, index+1, item)
		}
		if row < 0 || col < 0 {
			return nil, fmt.Errorf("%w：第 %d 条 %q 的行或列是负数", ErrEntry, index+1, item)
		}
		entries = append(entries, Entry{Row: row, Col: col, Value: value})
	}
	return entries, nil
}

// Build 把边表编成 CSR：同一 (row,col) 的重复项累加，显式的 0 不进结构，
// 行内列索引严格递增。内存 O(nnz + 行数)，不构造稠密矩阵。
func Build(rows, cols int, entries []Entry) *CSR {
	kept := make([]Entry, 0, len(entries))
	for _, entry := range entries {
		if entry.Row < 0 || entry.Row >= rows || entry.Col < 0 || entry.Col >= cols {
			continue
		}
		if entry.Value == 0 {
			continue
		}
		kept = append(kept, entry)
	}
	sort.Slice(kept, func(i, j int) bool {
		if kept[i].Row != kept[j].Row {
			return kept[i].Row < kept[j].Row
		}
		return kept[i].Col < kept[j].Col
	})
	// 排序后相同 (row,col) 相邻，原地累加重复项。
	merged := 0
	for _, entry := range kept {
		if merged > 0 && kept[merged-1].Row == entry.Row && kept[merged-1].Col == entry.Col {
			kept[merged-1].Value += entry.Value
			continue
		}
		kept[merged] = entry
		merged++
	}
	kept = kept[:merged]

	matrix := &CSR{Rows: rows, Cols: cols, Offsets: make([]int, rows+1)}
	for _, entry := range kept {
		if entry.Value != 0 {
			matrix.Offsets[entry.Row+1]++
		}
	}
	for row := 0; row < rows; row++ {
		matrix.Offsets[row+1] += matrix.Offsets[row]
	}
	total := matrix.Offsets[rows]
	matrix.Indices = make([]int, total)
	matrix.Values = make([]float64, total)
	cursor := make([]int, rows)
	copy(cursor, matrix.Offsets[:rows])
	for _, entry := range kept {
		if entry.Value == 0 {
			continue
		}
		at := cursor[entry.Row]
		matrix.Indices[at] = entry.Col
		matrix.Values[at] = entry.Value
		cursor[entry.Row]++
	}
	return matrix
}

// NNZ 返回非零元素个数。
func (m *CSR) NNZ() int {
	return len(m.Indices)
}

// SpMV 计算 y = A·x。
func (m *CSR) SpMV(x []float64) ([]float64, error) {
	if len(x) != m.Cols {
		return nil, fmt.Errorf("%w：向量长度是 %d，矩阵列数是 %d", ErrShape, len(x), m.Cols)
	}
	y := make([]float64, m.Rows)
	for row := 0; row < m.Rows; row++ {
		sum := 0.0
		for at := m.Offsets[row]; at < m.Offsets[row+1]; at++ {
			sum += m.Values[at] * x[m.Indices[at]]
		}
		y[row] = sum
	}
	return y, nil
}

// Transpose 把 CSR 转成列主序，等价于转置后的 CSR。
// 内存 O(nnz + 列数)，转置两次与原矩阵逐项相同。
func (m *CSR) Transpose() *CSR {
	total := len(m.Indices)
	transposed := &CSR{
		Rows:    m.Cols,
		Cols:    m.Rows,
		Offsets: make([]int, m.Cols+1),
		Indices: make([]int, total),
		Values:  make([]float64, total),
	}
	for _, col := range m.Indices {
		transposed.Offsets[col+1]++
	}
	for col := 0; col < m.Cols; col++ {
		transposed.Offsets[col+1] += transposed.Offsets[col]
	}
	cursor := make([]int, m.Cols)
	copy(cursor, transposed.Offsets[:m.Cols])
	for row := 0; row < m.Rows; row++ {
		for at := m.Offsets[row]; at < m.Offsets[row+1]; at++ {
			col := m.Indices[at]
			to := cursor[col]
			transposed.Indices[to] = row
			transposed.Values[to] = m.Values[at]
			cursor[col]++
		}
	}
	return transposed
}

// BCSR 是 B×B 分块压缩结构：块内稠密存储（块内原有的 0 也占位），
// 块级只存列块号。
type BCSR struct {
	Size      int
	Rows      int
	Cols      int
	Offsets   []int     // 块行偏移，长度 块行数+1
	BlockCols []int     // 每个非零块的列块号，块行内严格递增
	Values    []float64 // 每块 Size*Size 个，块内按行主序稠密存储
}

// Compress 把 CSR 编成 size×size 的 BCSR。
func (m *CSR) Compress(size int) (*BCSR, error) {
	if size <= 0 {
		return nil, fmt.Errorf("%w：收到 %d", ErrBlock, size)
	}
	blockRows := (m.Rows + size - 1) / size
	blocked := &BCSR{
		Size:    size,
		Rows:    m.Rows,
		Cols:    m.Cols,
		Offsets: make([]int, blockRows+1),
	}
	total := len(m.Indices)
	elemRow := make([]int, total)
	row := 0
	for at := 0; at < total; at++ {
		for at >= m.Offsets[row+1] {
			row++
		}
		elemRow[at] = row
	}
	// 按 (块行, 列块号) 稳定排序；同一块内保持 CSR 的 (行,列) 顺序。
	order := make([]int, total)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		a, b := order[i], order[j]
		blockRowA, blockRowB := elemRow[a]/size, elemRow[b]/size
		if blockRowA != blockRowB {
			return blockRowA < blockRowB
		}
		return m.Indices[a]/size < m.Indices[b]/size
	})
	// 第一遍：数出每个块行的非零块数。
	lastBlockRow, lastBlockCol := -1, -1
	for _, at := range order {
		blockRow, blockCol := elemRow[at]/size, m.Indices[at]/size
		if blockRow != lastBlockRow || blockCol != lastBlockCol {
			blocked.Offsets[blockRow+1]++
			lastBlockRow, lastBlockCol = blockRow, blockCol
		}
	}
	for blockRow := 0; blockRow < blockRows; blockRow++ {
		blocked.Offsets[blockRow+1] += blocked.Offsets[blockRow]
	}
	blockCount := blocked.Offsets[blockRows]
	blocked.BlockCols = make([]int, blockCount)
	blocked.Values = make([]float64, blockCount*size*size)
	// 第二遍：记录列块号并把值填进稠密块。
	lastBlockRow, lastBlockCol = -1, -1
	blockIndex := -1
	for _, at := range order {
		blockRow, blockCol := elemRow[at]/size, m.Indices[at]/size
		if blockRow != lastBlockRow || blockCol != lastBlockCol {
			blockIndex++
			blocked.BlockCols[blockIndex] = blockCol
			lastBlockRow, lastBlockCol = blockRow, blockCol
		}
		base := blockIndex * size * size
		blocked.Values[base+(elemRow[at]%size)*size+m.Indices[at]%size] += m.Values[at]
	}
	return blocked, nil
}

// Blocks 返回 size×size 分块后的非零块数和索引整数个数（列块号的个数）。
func (m *CSR) Blocks(size int) (int, int, error) {
	blocked, err := m.Compress(size)
	if err != nil {
		return 0, 0, err
	}
	return len(blocked.BlockCols), len(blocked.BlockCols), nil
}
