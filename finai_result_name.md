# 消融实验文件夹命名

## 一、外部知识 AntiPatternRAG（3 个）

```
ap_none
ap_chunk
ap_hier
```

含义对应：

- `ap_none`：不使用相似案例
- `ap_chunk`：使用普通分块得到的最相似案例
- `ap_hier`：使用父子分层聚合方法得到的最相似案例（你的方法）

------

## 二、内部知识 CodeGraphRAG（3 个）

```
cg_none
cg_no_irrelevant
cg_full
```

含义对应：

- `cg_none`：不使用图提取
- `cg_no_irrelevant`：不使用无关文件内容
- `cg_full`：使用完整 CodeGraphRAG（你的方法）

------

## 三、最终你会用到的 6 个实验文件夹名（汇总）

```
ap_none
ap_chunk
ap_hier
cg_none
cg_no_irrelevant
cg_full
```