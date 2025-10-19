import chardet

with open("D:/智能重构/CodeGraphRAG/.tmp/kafka-out.json", "rb") as f:
    data = f.read(2048)  # 读取部分内容即可
    print(chardet.detect(data))