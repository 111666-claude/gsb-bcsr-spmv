package bcsrspmv

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func sameCSR(a, b *CSR) bool {
	return a.Rows == b.Rows && a.Cols == b.Cols &&
		slices.Equal(a.Offsets, b.Offsets) &&
		slices.Equal(a.Indices, b.Indices) &&
		slices.Equal(a.Values, b.Values)
}

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

func TestSpMVShapeMismatch(t *testing.T) {
	entries, _ := ParseEntries("0-0:1")
	matrix := Build(1, 1, entries)
	if _, err := matrix.SpMV([]float64{1, 2}); err == nil {
		t.Fatal("向量长度不匹配应该报错")
	}
}

func TestBuildIgnoresOutOfRange(t *testing.T) {
	entries, _ := ParseEntries("0-0:1,5-5:9")
	matrix := Build(1, 1, entries)
	if len(matrix.Indices) != 1 {
		t.Fatalf("越界的元素应该被忽略：%+v", matrix)
	}
}

func TestBuildAccumulatesDuplicates(t *testing.T) {
	entries, _ := ParseEntries("0-1:1,0-1:2")
	matrix := Build(1, 2, entries)
	if len(matrix.Indices) != 1 || matrix.Indices[0] != 1 || matrix.Values[0] != 3 {
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

func TestBuildDropsExplicitZero(t *testing.T) {
	entries, _ := ParseEntries("0-1:0,0-2:1")
	matrix := Build(1, 3, entries)
	if len(matrix.Indices) != 1 || matrix.Indices[0] != 2 {
		t.Fatalf("显式的 0 不该占 nnz：%+v", matrix)
	}
}

func TestBuildDropsZeroSum(t *testing.T) {
	entries, _ := ParseEntries("0-1:1,0-1:-1,0-2:5")
	matrix := Build(1, 3, entries)
	if len(matrix.Indices) != 1 || matrix.Indices[0] != 2 || matrix.Values[0] != 5 {
		t.Fatalf("累加为 0 的项不该占 nnz：%+v", matrix)
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
}

func TestBuildDeterministic(t *testing.T) {
	entries, _ := ParseEntries("2-1:3,0-3:1,2-1:2,0-1:2,1-0:4")
	first := Build(3, 4, entries)
	second := Build(3, 4, entries)
	if !sameCSR(first, second) {
		t.Fatalf("同一输入重复构建应该逐项相同：%+v vs %+v", first, second)
	}
}

func TestBuildOffsetsConsistent(t *testing.T) {
	entries, _ := ParseEntries("2-1:3,0-3:1,2-1:2,0-1:2,1-0:4")
	matrix := Build(5, 4, entries)
	if len(matrix.Offsets) != 6 || matrix.Offsets[5] != len(matrix.Indices) {
		t.Fatalf("offsets 与 nnz 不自洽：%+v", matrix)
	}
	for row := 0; row < matrix.Rows; row++ {
		for at := matrix.Offsets[row] + 1; at < matrix.Offsets[row+1]; at++ {
			if matrix.Indices[at-1] >= matrix.Indices[at] {
				t.Fatalf("第 %d 行列索引不是严格递增：%v", row, matrix.Indices)
			}
		}
	}
}

func TestParseEntriesErrors(t *testing.T) {
	cases := []struct {
		text string
		item int
	}{
		{"0-1", 1},          // 缺 ':'
		{"0-1:2,01", 2},     // 缺 '-'
		{"a-1:2", 1},        // 行非数字
		{"0-b:2", 1},        // 列非数字
		{"0-1:x", 1},        // 值非数字
		{"-1-0:1", 1},       // 负行
		{"0-1:2,0--3:1", 2}, // 负列
	}
	for _, tc := range cases {
		_, err := ParseEntries(tc.text)
		if !errors.Is(err, ErrEntry) {
			t.Fatalf("%q 应该报 ErrEntry：%v", tc.text, err)
		}
		if !strings.Contains(err.Error(), fmt.Sprintf("第 %d 条", tc.item)) {
			t.Fatalf("%q 的错误应该指出第 %d 条：%v", tc.text, tc.item, err)
		}
	}
}

func TestSpMVShapeErrorMentionsLengths(t *testing.T) {
	entries, _ := ParseEntries("0-0:1,0-2:1")
	matrix := Build(1, 3, entries)
	_, err := matrix.SpMV([]float64{1, 2})
	if !errors.Is(err, ErrShape) {
		t.Fatalf("应该报 ErrShape：%v", err)
	}
	if !strings.Contains(err.Error(), "3") || !strings.Contains(err.Error(), "2") {
		t.Fatalf("错误应该说明两边长度：%v", err)
	}
}

func TestSpMVMatchesDense(t *testing.T) {
	entries, _ := ParseEntries("0-0:1,0-3:2,1-1:-1,2-2:4,2-0:0.5,3-3:3,0-0:1")
	matrix := Build(4, 4, entries)
	x := []float64{1, 2, 3, 4}
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
			t.Fatalf("第 %d 行与稠密乘法不一致：got %v want %v", row, y[row], want)
		}
	}
}

func TestTransposeTwiceIsIdentity(t *testing.T) {
	entries, _ := ParseEntries("0-3:1,0-1:2,2-0:5,2-2:4,1-1:1")
	matrix := Build(3, 4, entries)
	transposed := matrix.Transpose()
	if transposed.Rows != 4 || transposed.Cols != 3 {
		t.Fatalf("转置后行列应该互换：%+v", transposed)
	}
	// 转置后的 (0,1) 位置应来自原 (1,0)，(2,0) 来自原 (0,2)=2。
	if transposed.Offsets[1]-transposed.Offsets[0] != 1 {
		t.Fatalf("转置结构不对：%+v", transposed)
	}
	back := transposed.Transpose()
	if !sameCSR(matrix, back) {
		t.Fatalf("转置两次应该逐项相同：%+v vs %+v", matrix, back)
	}
}

func TestTransposeEmpty(t *testing.T) {
	matrix := Build(3, 5, nil)
	back := matrix.Transpose().Transpose()
	if !sameCSR(matrix, back) {
		t.Fatalf("空矩阵转置两次应该逐项相同：%+v vs %+v", matrix, back)
	}
}

func TestBlocksDense(t *testing.T) {
	// 一个填满的 4×4 块：nnz=16，块数 1，索引整数 1 ≤ 16/4。
	parts := []string{}
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			parts = append(parts, fmt.Sprintf("%d-%d:%d", row, col, row*4+col+1))
		}
	}
	entries, _ := ParseEntries(strings.Join(parts, ","))
	matrix := Build(4, 4, entries)
	blocked, err := matrix.Blocks(4)
	if err != nil {
		t.Fatalf("Blocks 出错：%v", err)
	}
	if len(blocked.Indices) != 1 || blocked.Indices[0] != 0 {
		t.Fatalf("应该只有一个列块号 0：%+v", blocked)
	}
	if len(blocked.Indices) > len(matrix.Indices)/4 {
		t.Fatalf("索引整数个数应该 ≤ nnz/4：%d > %d", len(blocked.Indices), len(matrix.Indices)/4)
	}
	if len(blocked.Values) != 16 {
		t.Fatalf("块内应该稠密存 16 个值：%d", len(blocked.Values))
	}
	for at, value := range blocked.Values {
		if value != float64(at+1) {
			t.Fatalf("块内第 %d 个值不对：%v", at, blocked.Values)
		}
	}
}

func TestBlocksZerosTakeSpace(t *testing.T) {
	// 4×4 块里只有 2 个非零，块内 14 个 0 也要占位。
	entries, _ := ParseEntries("0-0:1,3-3:2")
	matrix := Build(4, 4, entries)
	blocked, err := matrix.Blocks(4)
	if err != nil {
		t.Fatalf("Blocks 出错：%v", err)
	}
	if len(blocked.Indices) != 1 || len(blocked.Values) != 16 {
		t.Fatalf("一个块应该占 16 个值：%+v", blocked)
	}
	if blocked.Values[0] != 1 || blocked.Values[15] != 2 {
		t.Fatalf("块内值的位置不对：%v", blocked.Values)
	}
	nonzeros := 0
	for _, value := range blocked.Values {
		if value != 0 {
			nonzeros++
		}
	}
	if nonzeros != 2 {
		t.Fatalf("块内应该只有 2 个非零：%v", blocked.Values)
	}
}

func TestBlocksMultipleBlocks(t *testing.T) {
	// 8×8 矩阵按 4×4 分块，非零落在块 (0,0)、(0,1)、(1,1)。
	entries, _ := ParseEntries("0-0:1,1-1:2,0-4:3,5-5:4")
	matrix := Build(8, 8, entries)
	blocked, err := matrix.Blocks(4)
	if err != nil {
		t.Fatalf("Blocks 出错：%v", err)
	}
	if !reflect.DeepEqual(blocked.Offsets, []int{0, 2, 3}) {
		t.Fatalf("块行偏移不对：%v", blocked.Offsets)
	}
	if !reflect.DeepEqual(blocked.Indices, []int{0, 1, 1}) {
		t.Fatalf("列块号不对：%v", blocked.Indices)
	}
	if len(blocked.Values) != 3*16 {
		t.Fatalf("3 个块应该占 48 个值：%d", len(blocked.Values))
	}
	if blocked.Values[0] != 1 || blocked.Values[5] != 2 || blocked.Values[16] != 3 || blocked.Values[32+5] != 4 {
		t.Fatalf("块内值不对：%v", blocked.Values)
	}
}

func TestBlocksInvalidSize(t *testing.T) {
	matrix := Build(1, 1, nil)
	if _, err := matrix.Blocks(0); !errors.Is(err, ErrBlock) {
		t.Fatalf("块大小 0 应该报 ErrBlock：%v", err)
	}
}

func TestBuildScale(t *testing.T) {
	// 100 万个非零、10 万行：只验证能在线性内存下构建且结构自洽。
	const rows, cols, nnz = 100000, 100000, 1000000
	entries := make([]Entry, 0, nnz)
	for at := 0; at < nnz; at++ {
		entries = append(entries, Entry{Row: at % rows, Col: (at * 7) % cols, Value: 1})
	}
	matrix := Build(rows, cols, entries)
	if len(matrix.Offsets) != rows+1 || matrix.Offsets[rows] != len(matrix.Indices) {
		t.Fatalf("offsets 与 nnz 不自洽")
	}
	y, err := matrix.SpMV(make([]float64, cols))
	if err != nil {
		t.Fatalf("SpMV 出错：%v", err)
	}
	if len(y) != rows {
		t.Fatalf("y 长度不对：%d", len(y))
	}
}
