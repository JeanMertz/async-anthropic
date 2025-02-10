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

#[derive(Debug, Clone, Serialize, Deserialize, Builder, PartialEq)]
#[builder(setter(into, strip_option))]
pub struct Message {
    role: MessageRole,
    content: MessageContentList<MessageContent>,
}

impl Message {
    /// Returns all the tool uses in the message
    pub fn tool_uses(&self) -> Vec<ToolUse> {
        self.content
            .0
            .iter()
            .filter(|c| matches!(c, MessageContent::ToolUse(_)))
            .map(|c| match c {
                MessageContent::ToolUse(tool_use) => tool_use.clone(),
                _ => unreachable!(),
            })
            .collect()
    }

    /// Returns the first text content in the message
    pub fn text(&self) -> Option<String> {
        self.content
            .0
            .iter()
            .filter_map(|c| match c {
                MessageContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .next()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageContentList<T>(Vec<T>);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

impl CreateMessagesResponse {
    /// Returns the content as Messages so they are more easily reusable
    pub fn messages(&self) -> Vec<Message> {
        let Some(content) = &self.content else {
            return vec![];
        };
        content
            .iter()
            .map(|c| Message {
                role: MessageRole::Assistant,
                content: c.clone().into(),
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    ToolUse(ToolUse),
    ToolResult(ToolResult),
    Text(Text),
    // TODO: Implement images and documents
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Builder)]
#[builder(setter(into, strip_option), default)]
pub struct ToolUse {
    pub id: String,
    pub input: Value,
    pub name: String,
}

impl From<ToolUse> for MessageContent {
    fn from(tool_use: ToolUse) -> Self {
        MessageContent::ToolUse(tool_use)
    }
}

impl From<ToolUse> for MessageContentList<MessageContent> {
    fn from(tool_use: ToolUse) -> Self {
        MessageContentList(vec![tool_use.into()])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Builder)]
#[builder(setter(into, strip_option), default)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: Option<String>,
    pub is_error: bool,
}

impl From<ToolResult> for MessageContent {
    fn from(tool_result: ToolResult) -> Self {
        MessageContent::ToolResult(tool_result)
    }
}

impl From<ToolResult> for MessageContentList<MessageContent> {
    fn from(tool_result: ToolResult) -> Self {
        MessageContentList(vec![tool_result.into()])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Builder)]
#[builder(setter(into, strip_option), default)]
pub struct Text {
    pub text: String,
}

impl From<Text> for MessageContent {
    fn from(text: Text) -> Self {
        MessageContent::Text(text)
    }
}

impl From<Text> for MessageContentList<MessageContent> {
    fn from(text: Text) -> Self {
        MessageContentList(vec![text.into()])
    }
}

impl<S: AsRef<str>> From<S> for MessageContent {
    fn from(s: S) -> Self {
        MessageContent::Text(Text {
            text: s.as_ref().to_string(),
        })
    }
}

impl<S: AsRef<str>> From<S> for Message {
    fn from(s: S) -> Self {
        MessageBuilder::default()
            .role(MessageRole::User)
            .content(s.as_ref().to_string())
            .build()
            .expect("infallible")
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

impl From<MessageContent> for MessageContentList<MessageContent> {
    fn from(content: MessageContent) -> Self {
        MessageContentList(vec![content])
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

    #[test_log::test(tokio::test)]
    async fn test_from_str() {
        let message: Message = "Hello world!".into();

        assert_eq!(
            message,
            Message {
                role: MessageRole::User,
                content: MessageContentList(vec![MessageContent::Text(Text {
                    text: "Hello world!".to_string()
                })])
            }
        );

        assert_eq!(message.text(), Some("Hello world!".to_string()));
    }
}
