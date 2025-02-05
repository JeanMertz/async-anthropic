use std::collections::HashMap;

use derive_builder::Builder;
use reqwest::StatusCode;
use secrecy::ExposeSecret;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

use crate::{errors::AnthropicError, messages::Messages};

const BASE_URL: &str = "https://api.anthropic.com";

#[derive(Clone, Debug, Builder)]
#[builder(setter(into, strip_option))]
pub struct Client {
    #[builder(default)]
    http_client: reqwest::Client,
    #[builder(default)]
    base_url: String,
    #[builder(default)]
    api_key: secrecy::SecretString,
    #[builder(default)]
    version: String,
    #[builder(default)]
    beta: Option<String>,
}

impl Default for Client {
    fn default() -> Self {
        Self {
            http_client: reqwest::Client::new(),
            api_key: String::new().into(), // Default env?
            version: "2023-06-01".to_string(),
            beta: None,
            base_url: BASE_URL.to_string(),
        }
    }
}

impl Client {
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    pub fn messages(&self) -> Messages {
        Messages::new(self)
    }

    pub async fn post<I, O>(&self, path: &str, request: I) -> Result<O, AnthropicError>
    where
        I: Serialize,
        O: DeserializeOwned,
    {
        let mut request = self
            .http_client
            .post(format!(
                "{}/{}",
                &self.base_url.trim_end_matches('/'),
                &path.trim_start_matches('/')
            ))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", &self.version)
            .header("content-type", "application/json")
            .json(&request);

        if let Some(beta_value) = &self.beta {
            request = request.header("anthropic-beta", beta_value);
        }

        // TODO: Better handling deserialization errors
        // TODO: Handle status codes
        let response = request.send().await?;

        match response.status() {
            StatusCode::OK => {
                let response = response.json::<O>().await?;
                Ok(response)
            }
            StatusCode::BAD_REQUEST => {
                let text = response.text().await?;
                Err(AnthropicError::BadRequest(text))
            }
            StatusCode::UNAUTHORIZED => Err(AnthropicError::Unauthorized),
            _ => {
                let text = response.text().await?;
                Err(AnthropicError::Unknown(text))
            }
        }
    }

    // pub fn build(self) -> Result<Request, ReqwestError> {
    //     let mut body_map: HashMap<&str, Value> = HashMap::new();
    //     body_map.insert("model", json!(self.model));
    //     body_map.insert("max_tokens", json!(self.max_tokens));
    //     body_map.insert("messages", json!(self.messages));
    //     body_map.insert("stream", json!(self.stream));
    //     body_map.insert("temperature", json!(self.temperature));
    //     body_map.insert("system", json!(self.system));
    //
    //     if self.tools != Value::Null {
    //         body_map.insert("tools", self.tools.clone());
    //     }
    //     if let Some(tool_choice) = self.tool_choice {
    //         body_map.insert("tool_choice", json!(tool_choice));
    //     }
    //
    //     if self.metadata != Value::Null {
    //         body_map.insert("metadata", self.metadata.clone());
    //     }
    //
    //     if self.stop_sequences.len() > 0 {
    //         body_map.insert("stop_sequences", json!(self.stop_sequences));
    //     }
    //
    //     if let Some(top_k) = self.top_k {
    //         body_map.insert("top_k", json!(top_k));
    //     }
    //
    //     if let Some(top_p) = self.top_p {
    //         body_map.insert("top_p", json!(top_p));
    //     }
    //
    //     let mut request_builder = self
    //         .client
    //         .post(format!("{}/{MESSAGES_PATH}", self.base_url))
    //         .header("x-api-key", self.secret_key)
    //         .header("anthropic-version", self.version)
    //         .header("content-type", "application/json")
    //         .json(&body_map);
    //
    //     if let Some(beta_value) = self.beta {
    //         request_builder = request_builder.header("anthropic-beta", beta_value);
    //     }
    //
    //     Ok(Request {
    //         request_builder,
    //         stream: self.stream,
    //         verbose: self.verbose,
    //         tools: self.tools,
    //     })
    // }
    //
    // pub fn builder(self) -> Result<RequestBuilder, ReqwestError> {
    //     let mut body_map: HashMap<&str, Value> = HashMap::new();
    //     body_map.insert("model", json!(self.model));
    //     body_map.insert("max_tokens", json!(self.max_tokens));
    //     body_map.insert("messages", json!(self.messages));
    //     body_map.insert("stream", json!(self.stream));
    //     body_map.insert("temperature", json!(self.temperature));
    //     body_map.insert("system", json!(self.system));
    //
    //     if self.tools != Value::Null {
    //         body_map.insert("tools", self.tools.clone());
    //     }
    //
    //     if self.metadata != Value::Null {
    //         body_map.insert("metadata", self.metadata.clone());
    //     }
    //
    //     if self.stop_sequences.len() > 0 {
    //         body_map.insert("stop_sequences", json!(self.stop_sequences));
    //     }
    //
    //     if let Some(top_k) = self.top_k {
    //         body_map.insert("top_k", json!(top_k));
    //     }
    //
    //     if let Some(top_p) = self.top_p {
    //         body_map.insert("top_p", json!(top_p));
    //     }
    //
    //     let mut request_builder = self
    //         .client
    //         .post(self.base_url)
    //         .header("x-api-key", self.secret_key)
    //         .header("anthropic-version", self.version)
    //         .header("content-type", "application/json")
    //         .json(&body_map);
    //
    //     if let Some(beta_value) = self.beta {
    //         request_builder = request_builder.header("anthropic-beta", beta_value);
    //     }
    //
    //     Ok(request_builder)
    // }
}
