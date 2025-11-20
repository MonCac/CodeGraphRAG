# 给你修改建议和代码

1. 在 `LeaderAndIsrResponse` 类里增加一个静态代码块（`static { ... }`），里面调用 `AbstractResponse.registerParser`，把它自己的解析器注册进去。
2. 解析器实现可以用匿名内部类或 Lambda（Java 8+）写法。

------

# 修改示例

```
public class LeaderAndIsrResponse extends AbstractResponse {
    // ... 你已有的代码 ...

    // 注册解析器，静态块保证类加载时完成注册
    static {
        AbstractResponse.registerParser(ApiKeys.LEADER_AND_ISR, LeaderAndIsrResponse::parse);
    }

    public static LeaderAndIsrResponse parse(ByteBuffer buffer, short version) {
        return new LeaderAndIsrResponse(new LeaderAndIsrResponseData(new ByteBufferAccessor(buffer), version), version);
    }

    // ... 你已有的代码 ...
}
```

------

# 解释

- `LeaderAndIsrResponse::parse` 是方法引用，符合 `ResponseParser` 接口的 `parse(ByteBuffer, short)` 签名。
- 静态块会在类第一次加载时执行，完成注册。
- 之后 `AbstractResponse.parseResponse` 调用时直接调用注册的解析器，无需修改 `AbstractResponse` 之外的代码。