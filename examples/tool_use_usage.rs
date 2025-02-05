// examples/tool_use_usage.rs

// use anthropic_sdk::{Client, ToolChoice};
// use serde_json::json;

use anthropic_sdk::{
    types::{CreateMessagesRequestBuilder, MessageBuilder, MessageRole, ToolChoice},
    Client,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::default();

    let request = CreateMessagesRequestBuilder::default()
        .model("claude-3-5-sonnet-20241022")
        .messages(vec![MessageBuilder::default()
            .role(MessageRole::User)
            .content("What is the weather like in San Francisco?")
            .build()
            .unwrap()])
        // TODO: Type the tool spec so we can skip the shenanigans
        .tools([json!({
          "name": "get_weather",
          "description": "Get the current weather in a given location",
          "input_schema": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              }
            },
            "required": ["location"]
          }
        })
        .as_object()
        .unwrap()
        .to_owned()])
        .tool_choice(ToolChoice::Auto)
        .build()
        .unwrap();

    let response = client.messages().create(request).await?;

    println!("{:?}", response);

    Ok(())
}
