# bcsr-spmv

稀疏矩阵的 CSR / BCSR 构建与 SpMV，只用 Go 标准库。

```
go test ./...
go run ./cmd/bcsr-spmv --entries '0-1:1,0-1:2' --spmv '1,1'
go run ./cmd/bcsr-spmv --entries '0-3:1,0-1:2' --spmv '1,1,1,1'
```

## 口径（README 为准）

- **同一位置的重复项要累加**：`--entries '0-1:1,0-1:2' --spmv '1,1'` 按口径是 `y=3,0`，
  现在 `Build` 直接覆盖，给 `y=2,0`。
- **行内列索引要严格递增**：`--entries '0-3:1,0-1:2'` 按口径是 `indices=1,3`，现在按首次出现顺序给 `indices=3,1`。
- **显式的 0 值不进结构**：`0-1:0` 不该占 nnz。
- **分块压缩（BCSR）**：`--block 4` 按 4×4 分块，块内稠密、块级只存列块号，索引存储的整数个数要 ≤ nnz/4
  （块内原有的 0 也要占位）。现在 `--block` 直接报「分块压缩还没有实现」。
- 不变量：`SpMV` 的结果与稠密矩阵乘法逐项一致（绝对误差 ≤ 1e-12）；转置两次等于原矩阵；
  同一输入重复构建的 CSR 逐项相同。
- 规模：100 万非零元素、10 万行时内存 O(nnz + 行数)，不许构造 rows×cols 的稠密矩阵。

## 现在的行为

- `Build` 对同一 (row,col) 是覆盖而不是累加，`0-1:1,0-1:2` 只留下 2。
- 行内保持首次出现的列顺序，没有排序，也没有去掉零值。
- `Blocks` 返回 `ErrNoBlock`，`Transpose` 直接返回 `nil`。

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
