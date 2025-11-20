# 修复方案核心思路

- 新增一个 **静态注册表**：`Map<ApiKeys, ResponseParser>`，`ResponseParser` 是一个接口，定义 `AbstractResponse parse(ByteBuffer, short)` 方法。
- 每个具体响应类（比如 `LeaderAndIsrResponse`）实现自己的解析器，并在静态代码块中注册到这个表。
- `parseResponse(ApiKeys, ByteBuffer, short)` 里直接查表调用对应的解析器，去除switch。

这样能减少 `AbstractResponse` 对具体响应类的依赖，也方便后续新增响应只需注册即可。

------

# 修改后的 `AbstractResponse.java`

```
public abstract class AbstractResponse implements AbstractRequestResponse {
    public static final int DEFAULT_THROTTLE_TIME = 0;

    private final ApiKeys apiKey;

    protected AbstractResponse(ApiKeys apiKey) {
        this.apiKey = apiKey;
    }

    public final Send toSend(ResponseHeader header, short version) {
        return SendBuilder.buildResponseSend(header, data(), version);
    }

    final ByteBuffer serializeWithHeader(ResponseHeader header, short version) {
        return RequestUtils.serialize(header.data(), header.headerVersion(), data(), version);
    }

    final ByteBuffer serialize(short version) {
        return MessageUtil.toByteBuffer(data(), version);
    }

    public abstract Map<Errors, Integer> errorCounts();

    protected Map<Errors, Integer> errorCounts(Errors error) {
        return Collections.singletonMap(error, 1);
    }

    protected Map<Errors, Integer> errorCounts(Stream<Errors> errors) {
        return errors.collect(Collectors.groupingBy(e -> e, Collectors.summingInt(e -> 1)));
    }

    protected Map<Errors, Integer> errorCounts(Collection<Errors> errors) {
        Map<Errors, Integer> errorCounts = new HashMap<>();
        for (Errors error : errors)
            updateErrorCounts(errorCounts, error);
        return errorCounts;
    }

    protected Map<Errors, Integer> apiErrorCounts(Map<?, ApiError> errors) {
        Map<Errors, Integer> errorCounts = new HashMap<>();
        for (ApiError apiError : errors.values())
            updateErrorCounts(errorCounts, apiError.error());
        return errorCounts;
    }

    protected void updateErrorCounts(Map<Errors, Integer> errorCounts, Errors error) {
        Integer count = errorCounts.getOrDefault(error, 0);
        errorCounts.put(error, count + 1);
    }

    // 新增接口定义，解析器接口
    public interface ResponseParser {
        AbstractResponse parse(ByteBuffer buffer, short version);
    }

    // 维护注册表，ApiKeys -> 解析器
    private static final Map<ApiKeys, ResponseParser> parsers = new EnumMap<>(ApiKeys.class);

    // 注册解析器方法，供各响应类调用
    public static void registerParser(ApiKeys apiKey, ResponseParser parser) {
        parsers.put(apiKey, parser);
    }

    /**
     * Parse a response from the provided buffer.
     */
    public static AbstractResponse parseResponse(ByteBuffer buffer, RequestHeader requestHeader) {
        ApiKeys apiKey = requestHeader.apiKey();
        short apiVersion = requestHeader.apiVersion();

        ResponseHeader responseHeader = ResponseHeader.parse(buffer, apiKey.responseHeaderVersion(apiVersion));

        if (requestHeader.correlationId() != responseHeader.correlationId()) {
            throw new CorrelationIdMismatchException("Correlation id for response ("
                + responseHeader.correlationId() + ") does not match request ("
                + requestHeader.correlationId() + "), request header: " + requestHeader,
                requestHeader.correlationId(), responseHeader.correlationId());
        }

        return parseResponse(apiKey, buffer, apiVersion);
    }

    // 去掉switch，改用注册表调用
    public static AbstractResponse parseResponse(ApiKeys apiKey, ByteBuffer responseBuffer, short version) {
        ResponseParser parser = parsers.get(apiKey);
        if (parser == null) {
            throw new AssertionError(String.format("ApiKey %s is not currently handled in `parseResponse`, the code should be updated to do so.", apiKey));
        }
        return parser.parse(responseBuffer, version);
    }

    public boolean shouldClientThrottle(short version) {
        return false;
    }

    public ApiKeys apiKey() {
        return apiKey;
    }

    public abstract int throttleTimeMs();

    public abstract void maybeSetThrottleTimeMs(int throttleTimeMs);

    public String toString() {
        return data().toString();
    }
}
```

------

# 说明

- 新增了 `ResponseParser` 接口和静态注册表 `parsers`。
- 替换了原来的 `switch`，改用 `parsers.get(apiKey)` 调用。
- 各具体响应类负责在自己的类静态初始化时调用 `AbstractResponse.registerParser` 注册自己对应的解析器。