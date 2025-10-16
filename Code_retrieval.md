# code_retrieval.py

所有labels的处理方法



| Labels           | Solution                                  |
| ---------------- |-------------------------------------------|
| Annotation       | 包含location                                |
| AnnotationMember | 包含location                                |
| Class            | 包含location                                |
| Enum             | 包含location                                |
| EnumConstant     | 需找到parentId,使用parentId的location           |
| File             | 不包含location，直接获取整个文件的代码                   |
| Interface        | 包含location                                |
| Method           | 包含location                                |
| Module           | 不包含location                               |
| Package          | 不包含location，报错，报错信息为：提供的是package，无法获取代码片段 |
| Record           | 包含location                                |
| TypeParameter    | 包含location,但只有一行。                         |
| Variable         | 包含location,但只有一行。                         |

目前对于 TypeParameter 和 Variable 的提取逻辑就是只提取当前行



# embedding_builder.py

| Labels           |
| ---------------- |
| Annotation       |
| AnnotationMember |
| Class            |
| Enum             |
| EnumConstant     |
| File             |
| Interface        |
| Method           |
| Module           |
| Package          |
| Record           |
| TypeParameter    |
| Variable         |
