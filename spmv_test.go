package bcsrspmv

import (
	"math"
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
