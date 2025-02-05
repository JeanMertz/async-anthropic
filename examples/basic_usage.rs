// examples/basic_usage.rs

use anthropic_sdk::{
    types::{CreateMessagesRequestBuilder, MessageBuilder, MessageRole},
    Client,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::default();

    let request = CreateMessagesRequestBuilder::default()
        .model("claude-3-5-sonnet-20241022")
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("Hello claude!!")
            .build()
            .unwrap()])
        .build()
        .unwrap();

    let response = client.messages().create(request).await?;

    Ok(())
}
