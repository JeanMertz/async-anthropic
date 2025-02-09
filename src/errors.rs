use thiserror::Error;

#[derive(Debug, Error)]
pub enum CreateMessagesError {
    #[error(transparent)]
    AnthropicError(#[from] AnthropicError),
}

#[derive(Debug, Error)]
pub enum AnthropicError {
    #[error("network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("malformed request: {0}")]
    BadRequest(String),

    #[error("unauthorized; check your API key")]
    Unauthorized,

    #[error("unknown error: {0}")]
    Unknown(String),
}
