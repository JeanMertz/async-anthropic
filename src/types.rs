use derive_builder::Builder;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use crate::messages;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub enum ToolChoice {
    Auto,
    Any,
    Tool(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(setter(into, strip_option))]
pub struct Message {
    role: MessageRole,
    content: MessageContentList<MessageContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageContentList<T>(Vec<T>);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(setter(into, strip_option))]
pub struct CreateMessagesRequest {
    messages: Vec<Message>,
    model: String,
    #[builder(default = messages::DEFAULT_MAX_TOKENS)]
    max_tokens: i32,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    stop_sequences: Option<Vec<String>>,
    #[builder(default = "false")]
    stream: bool, // Optional default false
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    temperature: Option<f32>, // 0 < x < 1
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    tool_choice: Option<ToolChoice>,
    // TODO: Type this
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    tools: Option<Vec<serde_json::Map<String, Value>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    top_k: Option<u32>, // > 0
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    top_p: Option<f32>, // 0 < x < 1
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    system: Option<String>, // 0 < x < 1
}

// {"id":"msg_01KkaCASJuaAgTWD2wqdbwC8",
// "type":"message",
// "role":"assistant",
// "model":"claude-3-5-sonnet-20241022",
// "content":[{"type":"text",
// "text":"Hi! How can I help you today?"}],
// "stop_reason":"end_turn",
// "stop_sequence":null,
// "usage":{"input_tokens":10,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":12}}
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(setter(into, strip_option))]
pub struct CreateMessagesResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<MessageContent>>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    ToolUse {
        id: String,
        input: Value,
        name: String,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<String>,
        is_error: bool,
    },
    Text {
        text: String,
    },
    // #[serde(untagged)]
    // Text(String),
    // TODO: Implement images and documents
}

impl<S: AsRef<str>> From<S> for MessageContent {
    fn from(s: S) -> Self {
        MessageContent::Text {
            text: s.as_ref().to_string(),
        }
    }
}

// Implement for any IntoIterator where item is Into<MessageContent>
// TODO: Uncomment when we have specialization :')
// impl<I, T> From<I> for MessageContentList<MessageContent>
// where
//     I: IntoIterator<Item = T>,
//     T: Into<MessageContent>,
// {
//     fn from(iter: I) -> Self {
//         MessageContentList(
//             iter.into_iter()
//                 .map(|item| item.into())
//                 .collect::<Vec<MessageContent>>(),
//         )
//     }
// }

// Any single AsRef<str> can be converted to a MessageContent, in a list as a single item
impl<S: AsRef<str>> From<S> for MessageContentList<MessageContent> {
    fn from(s: S) -> Self {
        MessageContentList(vec![s.as_ref().into()])
    }
}

// TODO: check if needed
impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ToolChoice::Auto => {
                serde::Serialize::serialize(&serde_json::json!({"type": "auto"}), serializer)
            }
            ToolChoice::Any => {
                serde::Serialize::serialize(&serde_json::json!({"type": "any"}), serializer)
            }
            ToolChoice::Tool(name) => serde::Serialize::serialize(
                &serde_json::json!({"type": "tool", "name": name}),
                serializer,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test_log::test(tokio::test)]
    async fn test_deserialize_response() {
        let response = json!({
        "id":"msg_01KkaCASJuaAgTWD2wqdbwC8",
        "type":"message",
        "role":"assistant",
        "model":"claude-3-5-sonnet-20241022",
        "content":[
            {"type":"text",
        "text":"Hi! How can I help you today?"}],
        "stop_reason":"end_turn",
        "stop_sequence":null,
        "usage":{"input_tokens":10,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":12}}).to_string();

        assert!(serde_json::from_str::<CreateMessagesResponse>(&response).is_ok());
    }
}
