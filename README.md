# bcsr-spmv

稀疏矩阵的 CSR / BCSR 构建与 SpMV，只用 Go 标准库。

```
go test ./...
go run ./cmd/bcsr-spmv --entries '0-1:1,0-1:2' --spmv '1,1'
go run ./cmd/bcsr-spmv --entries '0-3:1,0-1:2' --spmv '1,1,1,1'
```

## 口径

- **同一位置的重复项要累加**：`--entries '0-1:1,0-1:2' --spmv '1,1'` 得 `y=3`。
- **行内列索引严格递增**：`--entries '0-3:1,0-1:2'` 得 `indices=1,3`。
- **显式的 0 值不进结构**：`0-1:0` 不占 nnz，累加后为 0 的项也不占。
- **分块压缩（BCSR）**：`--block 4` 按 4×4 分块，块内稠密、块级只存列块号，
  索引存储的整数个数 ≤ nnz/4（块内原有的 0 也占位）。
- 不变量：`SpMV` 的结果与稠密矩阵乘法逐项一致（绝对误差 ≤ 1e-12）；转置两次等于原矩阵；
  同一输入重复构建的 CSR 逐项相同。
- 规模：100 万非零元素、10 万行时内存 O(nnz + 行数)，不构造 rows×cols 的稠密矩阵。

## 计划新增的能力

```
--entries 非零元素   逗号分隔的 行-列:值
--spmv 向量          逗号分隔的 x
--block B            额外做 B×B 分块压缩
```

## 输出

```
rows=1 cols=2 nnz=1 offsets=[0 1] indices=1
y=3
```

## 目录

```
spmv.go               解析、CSR 构建与 SpMV
cmd/bcsr-spmv         命令行入口
spmv_test.go          go test 用例
```
