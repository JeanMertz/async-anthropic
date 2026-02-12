use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AnthropicError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("api error: {0}")]
    Api(#[from] ApiError),

    #[error("rate limited (retry after {} seconds)", retry_after.unwrap_or_default())]
    RateLimit { retry_after: Option<u64> },

    #[error("failed to deserialize response: {0}")]
    Deserialization(#[from] serde_json::Error),

    #[error("stream transport error: {0}")]
    StreamTransport(String),

    #[error("unknown error: {0}")]
    Unknown(String),
}

/// The wire-format envelope for Anthropic API errors.
///
/// ```json
/// {
///     "type": "error",
///     "error": {
///         "type": "overloaded_error",
///         "message": "Overloaded"
///     }
/// }
/// ```
///
/// The top-level `type` is always `"error"` and is discarded during
/// deserialization. Only the inner [`ApiError`] is kept.
#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorEnvelope {
    pub error: ApiError,
}

/// An error returned by the Anthropic API.
///
/// Represents the inner `error` object from the standard error envelope.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq, Serialize)]
pub struct ApiError {
    /// The error type, e.g. `"overloaded_error"`, `"rate_limit_error"`,
    /// `"invalid_request_error"`.
    #[serde(rename = "type")]
    pub error_type: String,

    /// Human-readable error message.
    pub message: Option<String>,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {}",
            self.error_type,
            self.message.as_deref().unwrap_or("(no message)")
        )
    }
}

impl std::error::Error for ApiError {}

pub(crate) fn map_deserialization_error(e: serde_json::Error, _bytes: &[u8]) -> AnthropicError {
    AnthropicError::Deserialization(e)
}
