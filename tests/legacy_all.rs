use anthropic_sdk::{Client, Request};
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
    let server = TestSetup::setup().await;
    let secret_key = "test_secret";

    let request = Client::new()
        .auth(secret_key)
        .model("test-model")
        .messages(&json!([]))
        .max_tokens(5)
        .build();

    assert!(request.is_ok());
}

#[tokio::test]
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

    let request = Client::new()
        .auth(secret_key)
        .model("test-model")
        .messages(&json!([]))
        .max_tokens(5)
        .base_url(&server.uri())
        .build()
        .unwrap();

    let result = request
        .execute(|text| async move {
            assert_eq!(text, "mocked response");
        })
        .await;

    assert!(result.is_ok(), "{result:?}");
}

#[tokio::test]
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

    let request = Client::new()
        .auth(secret_key)
        .model("test-model")
        .messages(&json!([]))
        .stream(true)
        .max_tokens(5)
        .base_url(&server.uri())
        .build()
        .unwrap();

    let result = request
        .execute(|text| async move {
            assert_eq!(text, "streamed chunk");
        })
        .await;

    assert!(result.is_ok(), "{result:?}");
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

    let request = Client::new()
        .auth(secret_key)
        .model("test-model")
        .messages(&json!([]))
        .max_tokens(5)
        .base_url(&server.uri())
        .build()
        .unwrap();

    let result = request.execute(|_| async move {}).await;

    assert!(result.is_err());
    assert_eq!(
        format!("{}", result.unwrap_err()),
        "Bad request. Check your request parameters. Bad request"
    );
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

    let request = Client::new()
        .auth(secret_key)
        .model("test-model")
        .messages(&json!([]))
        .max_tokens(5)
        .base_url(&server.uri())
        .build()
        .unwrap();

    dbg!(&request);

    let result = request.execute(|_| async move {}).await;

    assert!(result.is_err());
    assert_eq!(
        format!("{}", result.unwrap_err()),
        "Unauthorized. Check your authorization key."
    );
}
