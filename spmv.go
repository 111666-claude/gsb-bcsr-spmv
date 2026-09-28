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
	ErrBlock = errors.New("bcsrspmv: 分块大小不合法")
)

// Entry 是边表里的一条非零元素。
type Entry struct {
	Row   int
	Col   int
	Value float64
}

// CSR 是压缩稀疏行结构。行内列索引严格递增，同一 (row,col) 只出现一次，
// 显式的 0 值不进结构，Offsets 长度是 Rows+1 且 Offsets[Rows] == nnz。
type CSR struct {
	Rows    int
	Cols    int
	Offsets []int
	Indices []int
	Values  []float64
}

// BCSR 是 B×B 分块压缩结构：块级只存列块号，块内稠密存储 Size*Size 个值，
// 块内原有的 0 也占位。
type BCSR struct {
	Size    int
	Rows    int
	Cols    int
	Offsets []int // 块行偏移，长度是块行数+1
	Indices []int // 每个非零块的列块号
	Values  []float64
}

// ParseEntries 解析 "0-1:2,1-2:3"，出错时报第几条（从 1 数起）。
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
			return nil, fmt.Errorf("%w：第 %d 条 %q 含非数字", ErrEntry, index+1, item)
		}
		if row < 0 || col < 0 {
			return nil, fmt.Errorf("%w：第 %d 条 %q 行列不能为负", ErrEntry, index+1, item)
		}
		entries = append(entries, Entry{Row: row, Col: col, Value: value})
	}
	return entries, nil
}

// Build 把边表编成 CSR：同一 (row,col) 重复出现时累加，行内列索引严格递增，
// 显式的 0 值（含累加后为 0 的项）不进结构。内存 O(nnz + 行数)，
// 同一输入重复构建结果逐项相同。
func Build(rows, cols int, entries []Entry) *CSR {
	filtered := make([]Entry, 0, len(entries))
	for _, entry := range entries {
		if entry.Row < 0 || entry.Col < 0 || entry.Row >= rows || entry.Col >= cols {
			continue
		}
		if entry.Value == 0 {
			continue
		}
		filtered = append(filtered, entry)
	}
	sort.Slice(filtered, func(i, j int) bool {
		if filtered[i].Row != filtered[j].Row {
			return filtered[i].Row < filtered[j].Row
		}
		return filtered[i].Col < filtered[j].Col
	})
	matrix := &CSR{Rows: rows, Cols: cols, Offsets: make([]int, rows+1)}
	row := 0
	for _, entry := range filtered {
		for row < entry.Row {
			row++
			matrix.Offsets[row] = len(matrix.Indices)
		}
		last := len(matrix.Indices) - 1
		if last >= matrix.Offsets[row] && matrix.Indices[last] == entry.Col {
			matrix.Values[last] += entry.Value
			if matrix.Values[last] == 0 {
				matrix.Indices = matrix.Indices[:last]
				matrix.Values = matrix.Values[:last]
			}
			continue
		}
		matrix.Indices = append(matrix.Indices, entry.Col)
		matrix.Values = append(matrix.Values, entry.Value)
	}
	for row++; row <= rows; row++ {
		matrix.Offsets[row] = len(matrix.Indices)
	}
	return matrix
}

// SpMV 计算 y = A·x。
func (m *CSR) SpMV(x []float64) ([]float64, error) {
	if len(x) != m.Cols {
		return nil, fmt.Errorf("%w：矩阵列数 %d，向量长度 %d", ErrShape, m.Cols, len(x))
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

// Blocks 按 size×size 分块压缩：块级只存列块号，块内稠密存储 size*size 个值
// （块内原有的 0 也占位），索引整数个数是 len(Indices)。
func (m *CSR) Blocks(size int) (*BCSR, error) {
	if size <= 0 {
		return nil, fmt.Errorf("%w：%d", ErrBlock, size)
	}
	blockRows := (m.Rows + size - 1) / size
	out := &BCSR{Size: size, Rows: m.Rows, Cols: m.Cols, Offsets: make([]int, blockRows+1)}
	for blockRow := 0; blockRow < blockRows; blockRow++ {
		rowStart := blockRow * size
		rowEnd := rowStart + size
		if rowEnd > m.Rows {
			rowEnd = m.Rows
		}
		seen := map[int]bool{}
		cols := []int{}
		for row := rowStart; row < rowEnd; row++ {
			for at := m.Offsets[row]; at < m.Offsets[row+1]; at++ {
				blockCol := m.Indices[at] / size
				if !seen[blockCol] {
					seen[blockCol] = true
					cols = append(cols, blockCol)
				}
			}
		}
		sort.Ints(cols)
		out.Offsets[blockRow+1] = out.Offsets[blockRow] + len(cols)
		slot := make(map[int]int, len(cols))
		for index, blockCol := range cols {
			at := out.Offsets[blockRow] + index
			out.Indices = append(out.Indices, blockCol)
			out.Values = append(out.Values, make([]float64, size*size)...)
			slot[blockCol] = at
		}
		for row := rowStart; row < rowEnd; row++ {
			r := row - rowStart
			for at := m.Offsets[row]; at < m.Offsets[row+1]; at++ {
				blockCol := m.Indices[at] / size
				c := m.Indices[at] % size
				base := slot[blockCol] * size * size
				out.Values[base+r*size+c] = m.Values[at]
			}
		}
	}
	return out, nil
}

// Transpose 把 CSR 转成列主序（等价于转置后的 CSR），转置两次与原矩阵逐项相同。
func (m *CSR) Transpose() *CSR {
	nnz := len(m.Indices)
	out := &CSR{
		Rows:    m.Cols,
		Cols:    m.Rows,
		Offsets: make([]int, m.Cols+1),
		Indices: make([]int, nnz),
		Values:  make([]float64, nnz),
	}
	for _, col := range m.Indices {
		out.Offsets[col+1]++
	}
	for col := 0; col < m.Cols; col++ {
		out.Offsets[col+1] += out.Offsets[col]
	}
	cursor := make([]int, m.Cols)
	copy(cursor, out.Offsets[:m.Cols])
	for row := 0; row < m.Rows; row++ {
		for at := m.Offsets[row]; at < m.Offsets[row+1]; at++ {
			col := m.Indices[at]
			dst := cursor[col]
			cursor[col]++
			out.Indices[dst] = row
			out.Values[dst] = m.Values[at]
		}
	}
	return out
}
