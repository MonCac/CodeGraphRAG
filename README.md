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



