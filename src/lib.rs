use reqwest::{Client as ReqwestClient, Error as ReqwestError, RequestBuilder, StatusCode};
use serde::Deserialize;
use serde_json::{json, Value};
mod types;
use std::collections::HashMap;

mod client;
pub mod errors;
pub mod messages;
pub use client::Client;

// #[derive(Deserialize)]
// struct JsonResponse {
//     content: Vec<Content>,
// }
//
// #[derive(Deserialize)]
// struct Content {
//     #[serde(rename = "type")]
//     content_type: String,
//     text: String,
// }
//
// #[derive(Debug)]
// pub struct Request {
//     request_builder: RequestBuilder,
//     stream: bool,
//     verbose: bool,
//     tools: Value,
// }
//
// impl Request {
//     pub async fn execute<F, Fut>(self, mut callback: F) -> Result<()>
//     where
//         F: FnMut(String) -> Fut,
//         Fut: std::future::Future<Output = ()> + Send,
//     {
//         let mut response = self
//             .request_builder
//             .send()
//             .await
//             .context("Failed to send request")?;
//
//         dbg!(&response);
//         match response.status() {
//             StatusCode::OK => {
//                 if self.stream {
//                     let mut buffer = String::new();
//                     while let Some(chunk) = response.chunk().await? {
//                         let s = match std::str::from_utf8(&chunk) {
//                             Ok(v) => v,
//                             Err(e) => panic!("Invalid UTF-8 sequence: {}", e),
//                         };
//                         buffer.push_str(s);
//                         loop {
//                             if let Some(index) = buffer.find("\n\n") {
//                                 let chunk = buffer[..index].to_string();
//                                 buffer.drain(..=index + 1);
//
//                                 if self.verbose {
//                                     callback(chunk.clone()).await;
//                                 } else {
//                                     if chunk == "data: [DONE]" {
//                                         break;
//                                     }
//                                     let processed_chunk = chunk
//                                         .trim_start_matches("event: message_start")
//                                         .trim_start_matches("event: content_block_start")
//                                         .trim_start_matches("event: ping")
//                                         .trim_start_matches("event: content_block_delta")
//                                         .trim_start_matches("event: content_block_stop")
//                                         .trim_start_matches("event: message_delta")
//                                         .trim_start_matches("event: message_stop")
//                                         .to_string();
//                                     let cleaned_string = &processed_chunk
//                                         .trim_start()
//                                         .strip_prefix("data: ")
//                                         .unwrap_or(&processed_chunk);
//                                     match serde_json::from_str::<AnthropicChatCompletionChunk>(
//                                         &cleaned_string,
//                                     ) {
//                                         Ok(d) => {
//                                             if let Some(delta) = d.delta {
//                                                 if let Some(content) = delta.text {
//                                                     callback(content).await;
//                                                 }
//                                             }
//                                         }
//                                         Err(_) => {
//                                             let processed_chunk = cleaned_string
//                                                 .trim_start_matches("event: error")
//                                                 .to_string();
//                                             let cleaned_string = &processed_chunk
//                                                 .trim_start()
//                                                 .strip_prefix("data: ")
//                                                 .unwrap_or(&processed_chunk);
//                                             match serde_json::from_str::<AnthropicErrorMessage>(
//                                                 &cleaned_string,
//                                             ) {
//                                                 Ok(error_message) => {
//                                                     return Err(anyhow!(
//                                                         "{}: {}",
//                                                         error_message.error.error_type,
//                                                         error_message.error.message
//                                                     ));
//                                                 }
//                                                 Err(_) => {
//                                                     eprintln!(
//                                                         "Couldn't parse AnthropicChatCompletionChunk or AnthropicErrorMessage: {}",
//                                                         &cleaned_string
//                                                     );
//                                                 }
//                                             }
//                                         }
//                                     }
//                                 }
//                             } else {
//                                 break;
//                             }
//                         }
//                     }
//                 } else {
//                     let json_text = response
//                         .text()
//                         .await
//                         .context("Failed to read response text")?;
//                     if self.tools == Value::Null && !self.verbose {
//                         match serde_json::from_str::<JsonResponse>(&json_text) {
//                             Ok(parsed_json) => {
//                                 if let Some(content) = parsed_json
//                                     .content
//                                     .iter()
//                                     .find(|c| c.content_type == "text")
//                                 {
//                                     callback(content.text.clone()).await;
//                                 }
//                             }
//                             Err(_) => return Err(anyhow!("Unable to parse JSON")),
//                         }
//                     } else {
//                         callback(json_text).await;
//                     }
//                 }
//                 Ok(())
//             }
//             StatusCode::BAD_REQUEST => Err(anyhow!(
//                 "Bad request. Check your request parameters. {}",
//                 response.text().await?
//             )),
//             StatusCode::UNAUTHORIZED => Err(anyhow!("Unauthorized. Check your authorization key.")),
//             _ => {
//                 let error_message = format!("Unexpected status code: {:?}", response.text().await?);
//                 Err(anyhow!(error_message))
//             }
//         }
//     }
// }
