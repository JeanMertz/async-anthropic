use async_anthropic::{
    types::{
        CreateMessagesRequestBuilder, CustomTool, MessageBuilder, MessageRole, Tool, ToolChoice,
        ToolInputSchema, ToolInputSchemaKind, ToolResultBuilder,
    },
    Client,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::default();

    let mut messages = vec![];

    messages.push(
        MessageBuilder::default()
            .role(MessageRole::User)
            .content("What is the weather like in San Francisco?")
            .build()
            .unwrap(),
    );

    let request = CreateMessagesRequestBuilder::default()
        .model("claude-3-5-sonnet-20241022")
        .messages(messages.as_slice())
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
        .tool_choice(ToolChoice::auto())
        .build()
        .unwrap();

    let response = client.messages().create(request).await?;

    println!("1. ---");
    println!("{response:?}");

    for message in response.messages() {
        messages.push(message.clone());

        for tool_use in message.tool_uses() {
            println!("Tool use: {tool_use:?}");
            let location: String =
                serde_json::from_value(tool_use.input["location"].clone()).unwrap();

            messages.push(
                MessageBuilder::default()
                    .role(MessageRole::User)
                    .content(
                        ToolResultBuilder::default()
                            .tool_use_id(&tool_use.id)
                            .content(format!("Pretty warm in {location}"))
                            .build()
                            .unwrap(),
                    )
                    .build()
                    .unwrap(),
            );
        }
    }

    let request = CreateMessagesRequestBuilder::default()
        .model("claude-3-5-sonnet-20241022")
        .messages(messages.as_slice())
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

    let response = client.messages().create(request).await?;

    println!("2. ---");
    println!("{response:?}");

    // 2. ---
    // CreateMessagesResponse { id: Some("msg_019EVre2rdkCFwusZpGPgzDp"), content: Some([Text(Text { text: "According to the weather report, it's pretty warm in San Francisco right now." })]), model: Some("claude-3-5-sonnet-20241022"), stop_reason: Some("end_turn"), stop_sequence: None, usage: Some(Usage { input_tokens: Some(516), output_tokens: Some(20) }) }
    Ok(())
}
