use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{errors::CreateMessagesError, types::ToolChoice, types::Usage, Client};

pub const DEFAULT_MAX_TOKENS: i32 = 2048;

#[derive(Debug, Clone)]
pub struct Messages<'c> {
    client: &'c Client,
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
    #[builder(default = DEFAULT_MAX_TOKENS)]
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

impl Messages<'_> {
    pub fn new(client: &Client) -> Messages {
        Messages { client }
    }

    #[tracing::instrument(skip_all)]
    pub async fn create(
        &self,
        request: impl Into<CreateMessagesRequest>,
    ) -> Result<CreateMessagesResponse, CreateMessagesError> {
        // TODO: Handle streams like a champ
        //
        self.client
            .post("/v1/messages", request.into())
            .await
            .map_err(CreateMessagesError::AnthropicError)
    }
}
