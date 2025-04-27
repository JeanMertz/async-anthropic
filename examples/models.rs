// examples/models.rs

use async_anthropic::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::default();

    let response = client.models().list().await?;

    println!("{:?}", response);

    let response = client.models().get("claude-3-7-sonnet-20250219").await?;

    println!("{:?}", response);

    Ok(())
}
