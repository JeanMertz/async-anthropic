use async_anthropic::errors::{AnthropicError, CreateMessagesError};
use async_anthropic::types::{
    CreateMessagesRequestBuilder, MessageBuilder, MessageContent, MessageRole,
};
use async_anthropic::Client;
use async_trait::async_trait;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// Helper trait for setting up and tearing down mock server
#[async_trait]
pub trait MockApp {
    async fn setup() -> MockServer;
}

struct TestSetup;

#[async_trait]
impl MockApp for TestSetup {
    async fn setup() -> MockServer {
        MockServer::start().await
    }
}

#[tokio::test]
async fn test_client_build_request() {
    let secret_key = "test_secret";

    let request = Client::builder().api_key(secret_key).build();

    assert!(request.is_ok());
}

#[test_log::test(tokio::test)]
async fn test_successful_request_execution() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    // Mock successful response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "mocked response"}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::builder()
        .base_url(server.uri())
        .api_key(secret_key)
        .build()
        .unwrap();

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await.unwrap();

    if let Some(content) = result.content {
        if let MessageContent::Text { text } = &content[0] {
            assert_eq!(text, "mocked response");
        }
    }
}

#[tokio::test]
#[ignore = "streaming not implemented"]
async fn test_streaming_response() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    // Mock streaming response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string("event: content_block_start\ndata: {\"type\": \"text\", \"text\": \"streamed chunk\"}\n\n"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::builder()
        .base_url(server.uri())
        .api_key(secret_key)
        .build()
        .unwrap();

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await.unwrap();

    if let Some(content) = result.content {
        if let MessageContent::Text { text } = &content[0] {
            assert_eq!(text, "streamed chunk");
        }
    }
}

#[tokio::test]
async fn test_error_handling_bad_request() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    // Mock 400 Bad Request response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_string("Bad request"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::builder()
        .base_url(server.uri())
        .api_key(secret_key)
        .build()
        .unwrap();

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await;

    assert!(result.is_err());
    assert!(
        matches!(
            result.as_ref().unwrap_err(),
            CreateMessagesError::AnthropicError(AnthropicError::BadRequest(_))
        ),
        "actual: {:?}",
        &result
    )
}

#[tokio::test]
async fn test_error_handling_unauthorized() {
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    // Mock 401 Unauthorized response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::builder()
        .base_url(server.uri())
        .api_key(secret_key)
        .build()
        .unwrap();

    let request = CreateMessagesRequestBuilder::default()
        .model("test-model".to_string())
        .stream(true)
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello world!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let result = client.messages().create(request).await;

    assert!(result.is_err());
    assert!(
        matches!(
            result.as_ref().unwrap_err(),
            CreateMessagesError::AnthropicError(AnthropicError::Unauthorized)
        ),
        "actual: {:?}",
        &result
    )
}
