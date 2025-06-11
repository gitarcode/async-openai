use std::pin::Pin;

use bytes::Bytes;
use futures::{stream::StreamExt, Stream};
use reqwest::multipart::Form;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use serde::{de::DeserializeOwned, Serialize};
use tracing::info;

use crate::{
    config::{Config, OpenAIConfig},
    error::{map_deserialization_error, ApiError, OpenAIError, WrappedError},
    file::Files,
    image::Images,
    moderation::Moderations,
    traits::AsyncTryFrom,
    Assistants, Audio, AuditLogs, Batches, Chat, Completions, Embeddings, FineTuning, Invites,
    Models, Projects, Threads, Uploads, Users, VectorStores,
};

#[derive(Debug, Clone, Default)]
/// Client is a container for config, backoff and http_client
/// used to make API calls.
pub struct Client<C: Config> {
    http_client: reqwest::Client,
    config: C,
    backoff: backoff::ExponentialBackoff,
}

impl Client<OpenAIConfig> {
    /// Client with default [OpenAIConfig]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<C: Config> Client<C> {
    /// Create client with a custom HTTP client, OpenAI config, and backoff.
    pub fn build(
        http_client: reqwest::Client,
        config: C,
        backoff: backoff::ExponentialBackoff,
    ) -> Self {
        Self {
            http_client,
            config,
            backoff,
        }
    }

    /// Create client with [OpenAIConfig] or [crate::config::AzureConfig]
    pub fn with_config(config: C) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            config,
            backoff: Default::default(),
        }
    }

    /// Provide your own [client] to make HTTP requests with.
    ///
    /// [client]: reqwest::Client
    pub fn with_http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = http_client;
        self
    }

    /// Exponential backoff for retrying [rate limited](https://platform.openai.com/docs/guides/rate-limits) requests.
    pub fn with_backoff(mut self, backoff: backoff::ExponentialBackoff) -> Self {
        self.backoff = backoff;
        self
    }

    // API groups

    /// To call [Models] group related APIs using this client.
    pub fn models(&self) -> Models<C> {
        Models::new(self)
    }

    /// To call [Completions] group related APIs using this client.
    pub fn completions(&self) -> Completions<C> {
        Completions::new(self)
    }

    /// To call [Chat] group related APIs using this client.
    pub fn chat(&self) -> Chat<C> {
        Chat::new(self)
    }

    /// To call [Images] group related APIs using this client.
    pub fn images(&self) -> Images<C> {
        Images::new(self)
    }

    /// To call [Moderations] group related APIs using this client.
    pub fn moderations(&self) -> Moderations<C> {
        Moderations::new(self)
    }

    /// To call [Files] group related APIs using this client.
    pub fn files(&self) -> Files<C> {
        Files::new(self)
    }

    /// To call [Uploads] group related APIs using this client.
    pub fn uploads(&self) -> Uploads<C> {
        Uploads::new(self)
    }

    /// To call [FineTuning] group related APIs using this client.
    pub fn fine_tuning(&self) -> FineTuning<C> {
        FineTuning::new(self)
    }

    /// To call [Embeddings] group related APIs using this client.
    pub fn embeddings(&self) -> Embeddings<C> {
        Embeddings::new(self)
    }

    /// To call [Audio] group related APIs using this client.
    pub fn audio(&self) -> Audio<C> {
        Audio::new(self)
    }

    /// To call [Assistants] group related APIs using this client.
    pub fn assistants(&self) -> Assistants<C> {
        Assistants::new(self)
    }

    /// To call [Threads] group related APIs using this client.
    pub fn threads(&self) -> Threads<C> {
        Threads::new(self)
    }

    /// To call [VectorStores] group related APIs using this client.
    pub fn vector_stores(&self) -> VectorStores<C> {
        VectorStores::new(self)
    }

    /// To call [Batches] group related APIs using this client.
    pub fn batches(&self) -> Batches<C> {
        Batches::new(self)
    }

    /// To call [AuditLogs] group related APIs using this client.
    pub fn audit_logs(&self) -> AuditLogs<C> {
        AuditLogs::new(self)
    }

    /// To call [Invites] group related APIs using this client.
    pub fn invites(&self) -> Invites<C> {
        Invites::new(self)
    }

    /// To call [Users] group related APIs using this client.
    pub fn users(&self) -> Users<C> {
        Users::new(self)
    }

    /// To call [Projects] group related APIs using this client.
    pub fn projects(&self) -> Projects<C> {
        Projects::new(self)
    }

    pub fn config(&self) -> &C {
        &self.config
    }

    /// Make a GET request to {path} and deserialize the response body
    pub(crate) async fn get<O>(&self, path: &str) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
    {
        let request_maker = || async {
            Ok(self
                .http_client
                .get(self.config.url(path))
                .query(&self.config.query())
                .headers(self.config.headers())
                .build()?)
        };

        self.execute(request_maker).await
    }

    /// Make a GET request to {path} with given Query and deserialize the response body
    pub(crate) async fn get_with_query<Q, O>(&self, path: &str, query: &Q) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
        Q: Serialize + ?Sized,
    {
        let request_maker = || async {
            Ok(self
                .http_client
                .get(self.config.url(path))
                .query(&self.config.query())
                .query(query)
                .headers(self.config.headers())
                .build()?)
        };

        self.execute(request_maker).await
    }

    /// Make a DELETE request to {path} and deserialize the response body
    pub(crate) async fn delete<O>(&self, path: &str) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
    {
        let request_maker = || async {
            Ok(self
                .http_client
                .delete(self.config.url(path))
                .query(&self.config.query())
                .headers(self.config.headers())
                .build()?)
        };

        self.execute(request_maker).await
    }

    /// Make a GET request to {path} and return the response body
    pub(crate) async fn get_raw(&self, path: &str) -> Result<Bytes, OpenAIError> {
        let request_maker = || async {
            Ok(self
                .http_client
                .get(self.config.url(path))
                .query(&self.config.query())
                .headers(self.config.headers())
                .build()?)
        };

        self.execute_raw(request_maker).await
    }

    /// Make a POST request to {path} and return the response body
    pub(crate) async fn post_raw<I>(&self, path: &str, request: I) -> Result<Bytes, OpenAIError>
    where
        I: Serialize,
    {
        let request_maker = || async {
            Ok(self
                .http_client
                .post(self.config.url(path))
                .query(&self.config.query())
                .headers(self.config.headers())
                .json(&request)
                .build()?)
        };

        self.execute_raw(request_maker).await
    }

    /// Make a POST request to {path} and deserialize the response body
    pub(crate) async fn post<I, O>(&self, path: &str, request: I) -> Result<O, OpenAIError>
    where
        I: Serialize,
        O: DeserializeOwned,
    {
        let request_maker = || async {
            Ok(self
                .http_client
                .post(self.config.url(path))
                .query(&self.config.query())
                .headers(self.config.headers())
                .json(&request)
                .build()?)
        };

        self.execute(request_maker).await
    }

    /// POST a form at {path} and return the response body
    pub(crate) async fn post_form_raw<F>(&self, path: &str, form: F) -> Result<Bytes, OpenAIError>
    where
        Form: AsyncTryFrom<F, Error = OpenAIError>,
        F: Clone,
    {
        let request_maker = || async {
            Ok(self
                .http_client
                .post(self.config.url(path))
                .query(&self.config.query())
                .headers(self.config.headers())
                .multipart(<Form as AsyncTryFrom<F>>::try_from(form.clone()).await?)
                .build()?)
        };

        self.execute_raw(request_maker).await
    }

    /// POST a form at {path} and deserialize the response body
    pub(crate) async fn post_form<O, F>(&self, path: &str, form: F) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
        Form: AsyncTryFrom<F, Error = OpenAIError>,
        F: Clone,
    {
        let request_maker = || async {
            Ok(self
                .http_client
                .post(self.config.url(path))
                .query(&self.config.query())
                .headers(self.config.headers())
                .multipart(<Form as AsyncTryFrom<F>>::try_from(form.clone()).await?)
                .build()?)
        };

        self.execute(request_maker).await
    }

    /// Execute a HTTP request and retry on rate limit
    ///
    /// request_maker serves one purpose: to be able to create request again
    /// to retry API call after getting rate limited. request_maker is async because
    /// reqwest::multipart::Form is created by async calls to read files for uploads.
    async fn execute_raw<M, Fut>(&self, request_maker: M) -> Result<Bytes, OpenAIError>
    where
        M: Fn() -> Fut,
        Fut: core::future::Future<Output = Result<reqwest::Request, OpenAIError>>,
    {
        let client = self.http_client.clone();

        backoff::future::retry(self.backoff.clone(), || async {
            let request = request_maker().await.map_err(backoff::Error::Permanent)?;
            let response = client
                .execute(request)
                .await
                .map_err(OpenAIError::Reqwest)
                .map_err(backoff::Error::Permanent)?;

            let status = response.status();
            let bytes = response
                .bytes()
                .await
                .map_err(OpenAIError::Reqwest)
                .map_err(backoff::Error::Permanent)?;

            if status.is_server_error() {
                // OpenAI does not guarantee server errors are returned as JSON so we cannot deserialize them.
                let message: String = String::from_utf8_lossy(&bytes).into_owned();
                tracing::warn!("Server error: {status} - {message}");
                return Err(backoff::Error::Transient {
                    err: OpenAIError::ApiError(ApiError {
                        message,
                        r#type: None,
                        param: None,
                        code: None,
                    }),
                    retry_after: None,
                });
            }

            // Deserialize response body from either error object or actual response object
            if !status.is_success() {
                let wrapped_error: WrappedError = serde_json::from_slice(bytes.as_ref())
                    .map_err(|e| map_deserialization_error(e, bytes.as_ref()))
                    .map_err(backoff::Error::Permanent)?;

                if status.as_u16() == 429
                    // API returns 429 also when:
                    // "You exceeded your current quota, please check your plan and billing details."
                    && wrapped_error.error.r#type != Some("insufficient_quota".to_string())
                {
                    // Rate limited retry...
                    tracing::warn!("Rate limited: {}", wrapped_error.error.message);
                    return Err(backoff::Error::Transient {
                        err: OpenAIError::ApiError(wrapped_error.error),
                        retry_after: None,
                    });
                } else {
                    return Err(backoff::Error::Permanent(OpenAIError::ApiError(
                        wrapped_error.error,
                    )));
                }
            }

            Ok(bytes)
        })
        .await
    }

    /// Execute a HTTP request and retry on rate limit
    ///
    /// request_maker serves one purpose: to be able to create request again
    /// to retry API call after getting rate limited. request_maker is async because
    /// reqwest::multipart::Form is created by async calls to read files for uploads.
    async fn execute<O, M, Fut>(&self, request_maker: M) -> Result<O, OpenAIError>
    where
        O: DeserializeOwned,
        M: Fn() -> Fut,
        Fut: core::future::Future<Output = Result<reqwest::Request, OpenAIError>>,
    {
        let bytes = self.execute_raw(request_maker).await?;

        let response: O = serde_json::from_slice(bytes.as_ref())
            .map_err(|e| map_deserialization_error(e, bytes.as_ref()))?;

        Ok(response)
    }

    /// Make HTTP POST request to receive SSE
    pub(crate) async fn post_stream<I, O>(
        &self,
        path: &str,
        request: I,
    ) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIError>> + Send>>
    where
        I: Serialize,
        O: DeserializeOwned + std::marker::Send + 'static,
    {
        // Clone the request for potential retry as non-streaming
        let request_clone = serde_json::to_value(&request).unwrap();
        
        let event_source_result = self
            .http_client
            .post(self.config.url(path))
            .query(&self.config.query())
            .headers(self.config.headers())
            .json(&request)
            .eventsource();

        match event_source_result {
            Ok(event_source) => stream_enhanced_error_handling(event_source).await,
            Err(e) => {
                // EventSource creation failed, return error immediately
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                let _ = tx.send(Err(OpenAIError::StreamError(format!("Failed to create EventSource: {}", e))));
                Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
            }
        }
    }

    pub(crate) async fn post_stream_mapped_raw_events<I, O>(
        &self,
        path: &str,
        request: I,
        event_mapper: impl Fn(eventsource_stream::Event) -> Result<O, OpenAIError> + Send + 'static,
    ) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIError>> + Send>>
    where
        I: Serialize,
        O: DeserializeOwned + std::marker::Send + 'static,
    {
        let event_source = self
            .http_client
            .post(self.config.url(path))
            .query(&self.config.query())
            .headers(self.config.headers())
            .json(&request)
            .eventsource()
            .unwrap();

        stream_mapped_raw_events(event_source, event_mapper).await
    }

    /// Make HTTP GET request to receive SSE
    pub(crate) async fn _get_stream<Q, O>(
        &self,
        path: &str,
        query: &Q,
    ) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIError>> + Send>>
    where
        Q: Serialize + ?Sized,
        O: DeserializeOwned + std::marker::Send + 'static,
    {
        let event_source = self
            .http_client
            .get(self.config.url(path))
            .query(query)
            .query(&self.config.query())
            .headers(self.config.headers())
            .eventsource()
            .unwrap();

        stream(event_source).await
    }
}

/// Enhanced stream function with better error handling
pub(crate) async fn stream_enhanced_error_handling<O>(
    mut event_source: EventSource,
) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIError>> + Send>>
where
    O: DeserializeOwned + std::marker::Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    info!("(async) We got the error: {:?}", e);
                    
                    // Enhanced error handling with better messages
                    let pretty_err = match e {
                        reqwest_eventsource::Error::InvalidStatusCode(status_code, response) => {
                            let status_str = status_code.to_string();
                            match response.text().await {
                                Ok(response_text) => {
                                    if response_text.trim().is_empty() {
                                        format!("Invalid status code: {}", status_str)
                                    } else {
                                        // Try to parse as JSON error response first
                                        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&response_text) {
                                            if let Some(error_obj) = json_val.get("error") {
                                                if let Some(message) = error_obj.get("message").and_then(|m| m.as_str()) {
                                                    format!("Bad request: {}", message)
                                                } else {
                                                    format!("Error response: {}", response_text)
                                                }
                                            } else {
                                                format!("Error response: {}", response_text)
                                            }
                                        } else {
                                            // Not JSON, return raw response
                                            format!("Error: {}", response_text)
                                        }
                                    }
                                }
                                Err(_) => format!("Invalid status code: {}", status_str)
                            }
                        }
                        reqwest_eventsource::Error::InvalidContentType(header_value, response) => {
                            let header_str = header_value.to_str().unwrap_or("unknown");
                            match response.text().await {
                                Ok(response_text) => {
                                    if response_text.trim().is_empty() {
                                        format!("Invalid content type: {}", header_str)
                                    } else {
                                        format!("Invalid content type: {}\nResponse: {}", header_str, response_text)
                                    }
                                }
                                Err(_) => format!("Invalid content type: {}", header_str)
                            }
                        }
                        reqwest_eventsource::Error::Transport(ref transport_err) => {
                            // Handle transport errors which might contain HTTP error responses
                            if let Some(status) = transport_err.status() {
                                if status.is_client_error() || status.is_server_error() {
                                    format!("HTTP Error {}: {}", status.as_u16(), transport_err)
                                } else {
                                    format!("Transport error: {}", transport_err)
                                }
                            } else {
                                // Check if this is a decode error that might indicate a non-SSE response
                                let err_msg = transport_err.to_string();
                                if err_msg.contains("unexpected EOF") || err_msg.contains("chunk size") || err_msg.contains("decoding response body") {
                                    format!("Server returned non-streaming response. This usually indicates an HTTP error response (like max_tokens exceeded) was sent instead of streaming data. Check your request parameters and try a non-streaming request to see the specific error.")
                                } else {
                                    format!("Transport error: {}", transport_err)
                                }
                            }
                        }
                        _ => e.to_string(),
                    };

                    if let Err(_e) = tx.send(Err(OpenAIError::StreamError(pretty_err))) {
                        // rx dropped
                        break;
                    }
                }
                Ok(event) => match event {
                    Event::Message(message) => {
                        info!("(async) New message: {:?}", &message);
                        if message.data == "[DONE]" {
                            break;
                        }

                        // Check if this is an error response before attempting deserialization
                        let response = if message.data.contains("\"error\":{") {
                            // This is an error response, parse it and convert to OpenAIError
                            match serde_json::from_str::<serde_json::Value>(&message.data) {
                                Ok(json_val) => {
                                    if let Some(error_obj) = json_val.get("error") {
                                        let error_message = error_obj.get("message")
                                            .and_then(|m| m.as_str())
                                            .unwrap_or("Unknown error");
                                        Err(OpenAIError::StreamError(error_message.to_string()))
                                    } else {
                                        Err(OpenAIError::StreamError("Unknown error format".to_string()))
                                    }
                                },
                                Err(_) => Err(OpenAIError::StreamError("Failed to parse error response".to_string()))
                            }
                        } else {
                            // Normal response, deserialize as expected type
                            match serde_json::from_str::<O>(&message.data) {
                                Err(e) => Err(map_deserialization_error(e, message.data.as_bytes())),
                                Ok(output) => Ok(output),
                            }
                        };

                        if let Err(_e) = tx.send(response) {
                            // rx dropped
                            break;
                        }
                    }
                    Event::Open => continue,
                },
            }
        }

        event_source.close();
    });

    Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}

/// Request which responds with SSE.
/// [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format)
pub(crate) async fn stream<O>(
    mut event_source: EventSource,
) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIError>> + Send>>
where
    O: DeserializeOwned + std::marker::Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    info!("(async) We got the error: {:?}", e);
                    if let Err(_e) = tx.send(Err(OpenAIError::StreamError(e.to_string()))) {
                        break;
                    }
                }
                Ok(event) => match event {
                    Event::Message(message) => {
                        info!("(async) New message: {:?}", &message);
                        if message.data == "[DONE]" {
                            break;
                        }

                        let response = match serde_json::from_str::<O>(&message.data) {
                            Err(e) => Err(map_deserialization_error(e, message.data.as_bytes())),
                            Ok(output) => Ok(output),
                        };

                        if let Err(_e) = tx.send(response) {
                            break;
                        }
                    }
                    Event::Open => continue,
                },
            }
        }

        event_source.close();
    });

    Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}

pub(crate) async fn stream_mapped_raw_events<O>(
    mut event_source: EventSource,
    event_mapper: impl Fn(eventsource_stream::Event) -> Result<O, OpenAIError> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIError>> + Send>>
where
    O: DeserializeOwned + std::marker::Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    if let Err(_e) = tx.send(Err(OpenAIError::StreamError(e.to_string()))) {
                        // rx dropped
                        break;
                    }
                }
                Ok(event) => match event {
                    Event::Message(message) => {
                        let mut done = false;

                        if message.data == "[DONE]" {
                            done = true;
                        }

                        let response = event_mapper(message);

                        if let Err(_e) = tx.send(response) {
                            // rx dropped
                            break;
                        }

                        if done {
                            break;
                        }
                    }
                    Event::Open => continue,
                },
            }
        }

        event_source.close();
    });

    Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}
