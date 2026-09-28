package bcsrspmv

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestParseEntries(t *testing.T) {
	entries, err := ParseEntries("0-1:2, 1-2:3.5")
	if err != nil {
		t.Fatalf("ParseEntries 出错：%v", err)
	}
	if len(entries) != 2 || entries[0] != (Entry{Row: 0, Col: 1, Value: 2}) {
		t.Fatalf("解析结果不对：%+v", entries)
	}
	if _, err := ParseEntries("0-1"); err == nil {
		t.Fatal("缺 : 应该报错")
	}
}

func TestParseEntriesReportsIndex(t *testing.T) {
	cases := map[string]string{
		"0-1:2,1-2":    "第 2 条", // 缺 ':'
		"0-1:2,1:2":    "第 2 条", // 缺 '-'
		"0-1:2,1-a:3":  "第 2 条", // 列不是数字
		"0-1:2,1-2:x":  "第 2 条", // 值不是数字
		"0-1:2,1--2:3": "第 2 条", // 负数列
		"0-1:2,-1-2:3": "第 2 条", // 负数千位
	}
	for text, want := range cases {
		_, err := ParseEntries(text)
		if err == nil {
			t.Fatalf("%q 应该报错", text)
		}
		if !errors.Is(err, ErrEntry) {
			t.Fatalf("%q 的错误应该包 ErrEntry：%v", text, err)
		}
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("%q 的错误应指出是第几条：%v", text, err)
		}
	}
}

func TestBuildSingleEntry(t *testing.T) {
	entries, _ := ParseEntries("1-0:5")
	matrix := Build(2, 1, entries)
	if matrix.Rows != 2 || matrix.Cols != 1 || len(matrix.Indices) != 1 {
		t.Fatalf("结构不对：%+v", matrix)
	}
	if matrix.Indices[0] != 0 || matrix.Values[0] != 5 {
		t.Fatalf("数据不对：%+v", matrix)
	}
	if matrix.Offsets[0] != 0 || matrix.Offsets[2] != 1 {
		t.Fatalf("行偏移不对：%+v", matrix.Offsets)
	}
}

func TestBuildAccumulatesDuplicates(t *testing.T) {
	entries, _ := ParseEntries("0-1:1,0-1:2")
	matrix := Build(1, 2, entries)
	if matrix.NNZ() != 1 || matrix.Indices[0] != 1 || matrix.Values[0] != 3 {
		t.Fatalf("重复项应该累加：%+v", matrix)
	}
	y, err := matrix.SpMV([]float64{1, 1})
	if err != nil {
		t.Fatalf("SpMV 出错：%v", err)
	}
	if math.Abs(y[0]-3) > 1e-12 {
		t.Fatalf("y 应该是 3：%v", y)
	}
}

func TestBuildSortsColumns(t *testing.T) {
	entries, _ := ParseEntries("0-3:1,0-1:2")
	matrix := Build(1, 4, entries)
	if !reflect.DeepEqual(matrix.Indices, []int{1, 3}) {
		t.Fatalf("列索引应该递增：%v", matrix.Indices)
	}
	if !reflect.DeepEqual(matrix.Values, []float64{2, 1}) {
		t.Fatalf("值应该跟着列走：%v", matrix.Values)
	}
	if !reflect.DeepEqual(matrix.Offsets, []int{0, 2}) {
		t.Fatalf("行偏移不对：%v", matrix.Offsets)
	}
}

func TestBuildDropsExplicitZeros(t *testing.T) {
	entries, _ := ParseEntries("0-1:0,0-2:5,1-0:0")
	matrix := Build(2, 3, entries)
	if matrix.NNZ() != 1 || matrix.Indices[0] != 2 {
		t.Fatalf("显式的 0 不该占 nnz：%+v", matrix)
	}
	if !reflect.DeepEqual(matrix.Offsets, []int{0, 1, 1}) {
		t.Fatalf("行偏移不对：%v", matrix.Offsets)
	}
}

func TestBuildDeterministic(t *testing.T) {
	entries, _ := ParseEntries("2-0:1,0-3:4,0-1:2,2-0:5,1-2:3")
	first := Build(3, 4, entries)
	second := Build(3, 4, entries)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("同一输入两次构建应该逐项相同：%+v vs %+v", first, second)
	}
}

func TestBuildIgnoresOutOfRange(t *testing.T) {
	entries, _ := ParseEntries("0-0:1,5-5:9")
	matrix := Build(1, 1, entries)
	if len(matrix.Indices) != 1 {
		t.Fatalf("越界的元素应该被忽略：%+v", matrix)
	}
}

func TestSpMV(t *testing.T) {
	entries, _ := ParseEntries("0-0:1,0-1:2,1-1:3")
	matrix := Build(2, 2, entries)
	y, err := matrix.SpMV([]float64{1, 1})
	if err != nil {
		t.Fatalf("SpMV 出错：%v", err)
	}
	if math.Abs(y[0]-3) > 1e-12 || math.Abs(y[1]-3) > 1e-12 {
		t.Fatalf("乘法结果不对：%v", y)
	}
}

func TestSpMVMatchesDense(t *testing.T) {
	entries, _ := ParseEntries("0-0:1.5,0-3:2,1-1:-4,2-2:0.25,2-3:3,3-0:1,3-3:1")
	matrix := Build(4, 4, entries)
	x := []float64{1, -2, 0.5, 4}
	y, err := matrix.SpMV(x)
	if err != nil {
		t.Fatalf("SpMV 出错：%v", err)
	}
	dense := make([][]float64, 4)
	for row := range dense {
		dense[row] = make([]float64, 4)
	}
	for _, entry := range entries {
		dense[entry.Row][entry.Col] += entry.Value
	}
	for row := 0; row < 4; row++ {
		want := 0.0
		for col := 0; col < 4; col++ {
			want += dense[row][col] * x[col]
		}
		if math.Abs(y[row]-want) > 1e-12 {
			t.Fatalf("第 %d 行：SpMV 得 %v，稠密乘法得 %v", row, y[row], want)
		}
	}
}

func TestSpMVShapeMismatch(t *testing.T) {
	entries, _ := ParseEntries("0-0:1")
	matrix := Build(1, 1, entries)
	if _, err := matrix.SpMV([]float64{1, 2}); err == nil {
		t.Fatal("向量长度不匹配应该报错")
	}
}

func TestSpMVShapeMismatchMessage(t *testing.T) {
	entries, _ := ParseEntries("0-0:1,0-1:2,0-2:3")
	matrix := Build(1, 3, entries)
	_, err := matrix.SpMV([]float64{1, 2})
	if err == nil {
		t.Fatal("向量长度不匹配应该报错")
	}
	if !errors.Is(err, ErrShape) {
		t.Fatalf("错误应该包 ErrShape：%v", err)
	}
	message := err.Error()
	if !strings.Contains(message, "2") || !strings.Contains(message, "3") {
		t.Fatalf("错误应说明两边长度：%v", err)
	}
}

func TestTransposeTwice(t *testing.T) {
	entries, _ := ParseEntries("0-1:2,0-3:4,2-0:1,2-3:5,1-1:1,0-1:6")
	matrix := Build(3, 4, entries)
	twice := matrix.Transpose().Transpose()
	if !reflect.DeepEqual(matrix, twice) {
		t.Fatalf("转置两次应该逐项相同：%+v vs %+v", matrix, twice)
	}
}

func TestTransposeMatchesDense(t *testing.T) {
	entries, _ := ParseEntries("0-1:2,0-3:4,2-0:1,2-3:5")
	matrix := Build(3, 4, entries)
	transposed := matrix.Transpose()
	if transposed.Rows != 4 || transposed.Cols != 3 {
		t.Fatalf("转置后形状不对：%+v", transposed)
	}
	y, err := transposed.SpMV([]float64{1, 2, 3})
	if err != nil {
		t.Fatalf("SpMV 出错：%v", err)
	}
	// A^T · [1 2 3]^T，按稠密转置手算。
	want := []float64{3, 2, 0, 19}
	for col := range want {
		if math.Abs(y[col]-want[col]) > 1e-12 {
			t.Fatalf("第 %d 列：得 %v，想要 %v", col, y[col], want[col])
		}
	}
}

func TestBlocks(t *testing.T) {
	entries, _ := ParseEntries("0-0:1,0-3:2,3-0:3,3-3:4")
	matrix := Build(4, 4, entries)
	blocks, indexInts, err := matrix.Blocks(4)
	if err != nil {
		t.Fatalf("Blocks 出错：%v", err)
	}
	if blocks != 1 {
		t.Fatalf("应该只有 1 个非零块：%d", blocks)
	}
	if indexInts > matrix.NNZ()/4 {
		t.Fatalf("索引整数个数 %d 应该 ≤ nnz/4 = %d", indexInts, matrix.NNZ()/4)
	}
	blocked, err := matrix.Compress(4)
	if err != nil {
		t.Fatalf("Compress 出错：%v", err)
	}
	if len(blocked.Values) != 16 {
		t.Fatalf("块内应该稠密存储 4×4=16 个数：%d", len(blocked.Values))
	}
	// 块内行主序：(0,0)=1 (0,3)=2 (3,0)=3 (3,3)=4，其余是占位的 0。
	want := []float64{1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 4}
	if !reflect.DeepEqual(blocked.Values, want) {
		t.Fatalf("块内稠密存储不对：%v", blocked.Values)
	}
	if !reflect.DeepEqual(blocked.BlockCols, []int{0}) {
		t.Fatalf("列块号不对：%v", blocked.BlockCols)
	}
}

func TestBlocksMultipleBlockRows(t *testing.T) {
	entries, _ := ParseEntries("0-0:1,0-5:2,1-1:3,4-4:4,4-9:5,5-5:6")
	matrix := Build(8, 12, entries)
	blocked, err := matrix.Compress(4)
	if err != nil {
		t.Fatalf("Compress 出错：%v", err)
	}
	if !reflect.DeepEqual(blocked.BlockCols, []int{0, 1, 1, 2}) {
		t.Fatalf("列块号不对：%v", blocked.BlockCols)
	}
	if !reflect.DeepEqual(blocked.Offsets, []int{0, 2, 4}) {
		t.Fatalf("块行偏移不对：%v", blocked.Offsets)
	}
	blocks, indexInts, err := matrix.Blocks(4)
	if err != nil {
		t.Fatalf("Blocks 出错：%v", err)
	}
	if blocks != 4 || indexInts != 4 {
		t.Fatalf("非零块数与索引整数个数不对：%d %d", blocks, indexInts)
	}
}

func TestBlocksInvalidSize(t *testing.T) {
	matrix := Build(1, 1, nil)
	if _, _, err := matrix.Blocks(0); err == nil {
		t.Fatal("分块大小为 0 应该报错")
	}
	if _, err := matrix.Compress(-2); err == nil {
		t.Fatal("分块大小为负应该报错")
	}
}

func TestBuildScalesToMillionEntries(t *testing.T) {
	const rows = 100000
	const nnz = 1000000
	entries := make([]Entry, 0, nnz)
	for i := 0; i < nnz; i++ {
		entries = append(entries, Entry{Row: i / 10, Col: i % 10, Value: float64(i%7) + 1})
	}
	matrix := Build(rows, 10, entries)
	if matrix.NNZ() != nnz {
		t.Fatalf("nnz 不对：%d", matrix.NNZ())
	}
	if len(matrix.Offsets) != rows+1 {
		t.Fatalf("offsets 长度不对：%d", len(matrix.Offsets))
	}
	x := make([]float64, 10)
	for i := range x {
		x[i] = 1
	}
	y, err := matrix.SpMV(x)
	if err != nil {
		t.Fatalf("SpMV 出错：%v", err)
	}
	for row := 0; row < rows; row++ {
		want := 0.0
		for col := 0; col < 10; col++ {
			want += float64((row*10+col)%7) + 1
		}
		if math.Abs(y[row]-want) > 1e-12 {
			t.Fatalf("第 %d 行：得 %v，想要 %v", row, y[row], want)
		}
	}
}
