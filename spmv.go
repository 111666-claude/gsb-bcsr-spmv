// Package bcsrspmv 把稀疏矩阵编成 CSR / BCSR 并做 SpMV，只用标准库。
package bcsrspmv

import (
	"errors"
	"strconv"
	"strings"
)

var (
	// ErrEntry 表示边表格式不对。
	ErrEntry = errors.New("bcsrspmv: 非零元素格式不对")
	// ErrShape 表示向量长度与矩阵列数不匹配。
	ErrShape = errors.New("bcsrspmv: 向量长度与矩阵列数不匹配")
	// ErrNoBlock 表示分块压缩还没有实现。
	ErrNoBlock = errors.New("bcsrspmv: 分块压缩还没有实现")
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

// ParseEntries 解析 "0-1:2,1-2:3"。
// 缺陷：出错只给一句笼统错误，没有条目下标。
func ParseEntries(text string) ([]Entry, error) {
	entries := []Entry{}
	for _, item := range strings.Split(text, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		position, rawValue, found := strings.Cut(item, ":")
		if !found {
			return nil, ErrEntry
		}
		rowText, colText, found := strings.Cut(position, "-")
		if !found {
			return nil, ErrEntry
		}
		row, errRow := strconv.Atoi(strings.TrimSpace(rowText))
		col, errCol := strconv.Atoi(strings.TrimSpace(colText))
		value, errValue := strconv.ParseFloat(strings.TrimSpace(rawValue), 64)
		if errRow != nil || errCol != nil || errValue != nil || row < 0 || col < 0 {
			return nil, ErrEntry
		}
		entries = append(entries, Entry{Row: row, Col: col, Value: value})
	}
	return entries, nil
}

// Build 缺陷：同一 (row,col) 重复出现时直接覆盖，行内按首次出现顺序，
// 也不剔除显式的零值。
func Build(rows, cols int, entries []Entry) *CSR {
	rowsData := make([][]Entry, rows)
	for _, entry := range entries {
		if entry.Row >= rows || entry.Col >= cols {
			continue
		}
		row := rowsData[entry.Row]
		replaced := false
		for index := range row {
			if row[index].Col == entry.Col {
				row[index].Value = entry.Value
				replaced = true
				break
			}
		}
		if !replaced {
			rowsData[entry.Row] = append(row, entry)
		}
	}
	matrix := &CSR{Rows: rows, Cols: cols}
	for row := 0; row < rows; row++ {
		matrix.Offsets = append(matrix.Offsets, len(matrix.Indices))
		for _, entry := range rowsData[row] {
			matrix.Indices = append(matrix.Indices, entry.Col)
			matrix.Values = append(matrix.Values, entry.Value)
		}
	}
	matrix.Offsets = append(matrix.Offsets, len(matrix.Indices))
	return matrix
}

// SpMV 计算 y = A·x。
func (m *CSR) SpMV(x []float64) ([]float64, error) {
	if len(x) != m.Cols {
		return nil, ErrShape
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

// Blocks 缺陷：还没实现。
func (m *CSR) Blocks(size int) (int, int, error) {
	return 0, 0, ErrNoBlock
}

// Transpose 缺陷：还没实现。
func (m *CSR) Transpose() *CSR {
	return nil
}
