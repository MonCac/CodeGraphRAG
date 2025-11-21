# CodeGraphRAG

中间文件输出说明：

```
/tmp
	before-out.json
	kafka-out.json           反模式文件集 + 项目的 enre 输出结果
	
	before-graph.json
	kafka-graph.json         反模式文件集 + 项目的 graph 映射
	
	subgraph-file.json       反模式文件集在项目中的 subgraph 提取结果
	
	
	semantic_total_result.json
	semantic_file_result.json
	
	antipattern_relevance_result.json
	
	/repair_outputs
		...
	final-result.json        最终的 graph 输出结果

```









今日要做的事的步骤
1. 根据 graph_data 提取所有 files 的路径
2. 将路径拆分为3类。
	1. 首先根据 _antipattern.json 文件，得到直接关联的 files
	2. 根据 具体被调用的 call 方法，找到所有关联的文件，得到不属于直接关联的文件
	3. 剩余的文件为提供修复信息的文件集
3. 编写两个 method，生成直接关联 files 的描述和间接关联文件的描述
4. 构造最终的 prompt 来实现反模式修复











# Linux 服务器部署

若出现 `mgclient.cpython-312-x86_64-linux-gnu.so: undefined symbol: SSL_get1_peer_certificate`报错，说明此时调用的 OpenSSL为v1，需要配置环境为OpenSSLv3.

在用户目录下载OpenSSLv3，然后配置环境变量：

```bash
export LDFLAGS="-L$HOME/openssl-3/lib64"
export CPPFLAGS="-I$HOME/openssl-3/include"
export PKG_CONFIG_PATH="$HOME/openssl-3/lib64/pkgconfig"
export LD_LIBRARY_PATH="$HOME/openssl-3/lib64:$LD_LIBRARY_PATH"
```

接着：

清除环境中原有的 pymgclient

```bash
uv pip uninstall pymgclient
```

下载新的 pymgclient，动态链接用户环境中的 OpenSSLv3

```bash
uv pip install --no-binary :all: --no-cache-dir pymgclient 2>&1 | tee compile.log
```

安装时，`compile.log` 中应该出现：

```bash
-L/xxx/openssl-3/lib64
-I/xxx/openssl-3/include
```

编译完成后验证链接

```bash
MGCLIENT_SO=$(python3 -c "import mgclient; print(mgclient.__file__)")
ldd $MGCLIENT_SO | grep ssl
```

此时应该显示 `/x x x/openssl-3/lib64/libssl.so`



永久生效方法：

```bash
nano ~/.bashrc

# OpenSSL 3 环境变量
export LDFLAGS="-L$HOME/openssl-3/lib64"
export CPPFLAGS="-I$HOME/openssl-3/include"
export PKG_CONFIG_PATH="$HOME/openssl-3/lib64/pkgconfig"
export LD_LIBRARY_PATH="$HOME/openssl-3/lib64:$LD_LIBRARY_PATH"

# 保存后退出（Nano 中按 Ctrl+O 保存，Ctrl+X 退出）。

# 然后立即生效
source ~/.bashrc
```



# venv

激活虚拟环境命令

```bash
source .venv/bin/activate
```

切换回全局python环境

```bash
deactivate
```





# uv

更新 python 环境

```bash
uv python pin 3.11.12
```







流程图

```bash
@startuml
title 基于知识图谱的 RAG 反模式代码修复流程（保留语义扩展判断）

start

:输入反模式代码文件;

:解析代码，构建全项目实体依赖图;
note right: 提取函数、类、模块、调用关系等

:识别反模式相关的关键实体;
note right: 从输入文件中抽取与反模式相关的函数/类等节点

:基于依赖图查找直接相关文件;
note right: 得到第一层受影响文件集合 (结构层)

if (是否需要语义扩展分析?) then (是)
    :对关键实体生成分层语义描述;
    note right: 到 file 粒度
    - 子实体层: 函数/类语义描述
    - 文件层: 聚合子实体描述

    :LLM 评判潜在修复文件;
    note right: 根据分层语义描述判断哪些文件涉及反模式修复
    :得到潜在修复文件集合;
else (否)
    :直接使用结构依赖文件集合;
    note right: 不使用语义描述和 LLM 判断
endif

:基于文件集合生成修复上下文;
note right: 为 RAG 阶段准备输入，包括相关代码片段、依赖关系等

:调用 LLM 进行修复生成与验证;
note right: 使用候选文件上下文进行多文件修复生成

stop

@enduml
```

