// examples/basic_usage.rs

use async_anthropic::{
    types::{
        CreateMessagesRequestBuilder, CustomTool, MessageBuilder, MessageRole, Tool,
        ToolInputSchema, ToolInputSchemaKind,
    },
    Client,
};
use tokio_stream::StreamExt as _;

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
        .tools(vec![Tool::Custom(CustomTool {
            name: "get_weather".to_string(),
            description: Some("Get the current weather in a given location".to_string()),
            input_schema: ToolInputSchema {
                kind: ToolInputSchemaKind::Object,
                properties: serde_json::Map::from_iter(vec![(
                    "location".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }),
                )]),
                required: vec!["location".to_string()],
            },
            cache_control: None,
        })])
        .build()
        .unwrap();

    let mut stream = client.messages().create_stream(request).await;

    while let Some(response) = stream.next().await {
        match response {
            Ok(msg) => println!("{msg:?}"),
            Err(e) => eprintln!("Error: {e:?}"),
        }
    }

    Ok(())
}
