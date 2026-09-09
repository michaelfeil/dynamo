# baseten-mm-client

Rust-native HTTP multimodal encoder client. Uses
`baseten_performance_client_core` 0.1.13-post.1 for batching, retries, timeouts, proxying,
and cancellation. No Python runtime is needed.

```rust,no_run
use baseten_mm_client::{EncodeRequest, HttpEncoderConfig, Media, MultiModalClient};

# async fn example() -> anyhow::Result<()> {
let mut config = HttpEncoderConfig::new("http://encoder:8000/encode");
config.api_key = "secret".into();
let client = MultiModalClient::new(config)?;
let request = EncodeRequest::new(
    Media::Image {
        image_url: [("url".into(), "https://example.com/image.jpg".into())].into(),
    },
    "served-model",
);
let responses = client.encode(vec![request]).await?;
let response = &responses[0];
let packed_embeddings = response.mm_kwargs.bytes()?;
# Ok(())
# }
```

`encode` validates success envelopes and returns hash, positive token count, and
opaque embedding payload. Encoder rejections retain their status in
`EncoderRejection`; HTTP status failures can be inspected with `http_error_status`.
`call_batch` returns raw envelopes and HTTP timing/cache metrics for integrations
that own their error mapping and metrics registration.

`cache_urls` routes through the first cache and forwards the remaining hops via
the transparent-route header. `proxy` enables cacheable PUT requests. When unset,
the client reads `BDN_PROXY` unless `DYNAMO_DISABLE_BDN_PROXY` is `1` or `true`
(case-insensitive). BDN URLs without credentials get the standard BDN credentials;
existing credentials are preserved. An explicit `proxy` overrides the environment.
Response order matches input order. Dropping the batch future cancels pending work.

`ReplayRequest` serializes the original encoder input together with its expected
hash and length. `replay` re-executes those inputs through the same transport and
rejects metadata mismatches before returning full payloads. Credentials and cache
routes stay in the receiving worker's client configuration, not in the request.
