use crate::{
    errors::CreateMessagesError,
    types::{CreateMessagesRequest, CreateMessagesResponse},
    Client,
};

pub const DEFAULT_MAX_TOKENS: i32 = 2048;

#[derive(Debug, Clone)]
pub struct Messages<'c> {
    client: &'c Client,
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
