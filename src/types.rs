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
    role: String,
    content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(setter(into, strip_option))]
pub struct CreateMessagesRequest {
    #[builder(default = messages::DEFAULT_MAX_TOKENS)]
    max_tokens: i32,
    messages: Vec<Message>,
    model: String,
    #[builder(default)]
    metadata: Option<serde_json::Map<String, Value>>,
    #[builder(default)]
    stop_sequences: Option<Vec<String>>,
    #[builder(default = "false")]
    stream: bool, // Optional default false
    #[builder(default)]
    temperature: Option<f32>, // 0 < x < 1
    #[builder(default)]
    tool_choice: Option<ToolChoice>,
    // TODO: Type this
    #[builder(default)]
    tools: Option<Vec<serde_json::Map<String, Value>>>,
    #[builder(default)]
    top_k: Option<u32>, // > 0
    #[builder(default)]
    top_p: Option<f32>, // 0 < x < 1
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
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
    Text {
        text: String,
    },
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
    // TODO: Implement images and documents
}

impl<S: AsRef<str>> From<S> for MessageContent {
    fn from(s: S) -> Self {
        MessageContent::Text {
            text: s.as_ref().to_string(),
        }
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
