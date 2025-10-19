#!/bin/bash

# 用法: ./check_openssl_func.sh SSL_get1_peer_certificate

FUNC_NAME=$1

if [ -z "$FUNC_NAME" ]; then
    echo "请提供函数名，例如: $0 SSL_get1_peer_certificate"
    exit 1
fi

# 创建临时 C 文件
TMP_C=$(mktemp /tmp/check_func.XXXX.c)
TMP_BIN=$(mktemp /tmp/check_func.XXXX.out)

cat > $TMP_C <<EOF
#include <openssl/ssl.h>

int main() {
    SSL *ssl = NULL;
    void *p = (void*) $FUNC_NAME;
    return 0;
}
EOF

# 尝试编译
gcc $TMP_C -o $TMP_BIN -lssl -lcrypto &> /dev/null
RESULT=$?

# 删除临时文件
rm -f $TMP_C $TMP_BIN

if [ $RESULT -eq 0 ]; then
    echo "✅ 函数 $FUNC_NAME 可用！"
else
    echo "❌ 函数 $FUNC_NAME 不可用"
fi


