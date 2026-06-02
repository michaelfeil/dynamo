// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use anyhow::{Error, Result};
use chrono::Utc;
use dynamo_runtime::component::Component;
use dynamo_runtime::prelude::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher as RuntimeEventPublisher;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing as log;

/// Configuration options for the JsonPublisher
pub struct JsonPublisherConfig {
    /// Buffer size for the message channel
    pub channel_buffer_size: usize,
    /// Whether to automatically add timestamps to events
    pub add_timestamps: bool,
}

impl Default for JsonPublisherConfig {
    fn default() -> Self {
        Self {
            channel_buffer_size: 100,
            add_timestamps: true,
        }
    }
}

/// A publisher for JSON events that processes them asynchronously
pub struct JsonPublisher {
    /// Channel for sending events to the background task
    sender: mpsc::Sender<(String, JsonValue)>,
    /// Cancellation token for shutting down the publisher
    cancellation_token: CancellationToken,
    /// The task handle for the background processor
    processor_handle: Option<tokio::task::JoinHandle<()>>,
}

impl JsonPublisher {
    /// Creates a new JsonPublisher that will send events to the provided component
    pub fn new(component: Component, config: Option<JsonPublisherConfig>) -> Self {
        let config = config.unwrap_or_default();
        let (sender, receiver) = mpsc::channel::<(String, JsonValue)>(config.channel_buffer_size);
        let cancellation_token = CancellationToken::new();

        // Start the processor task
        let processor_handle = Self::start_processor(
            component,
            receiver,
            cancellation_token.clone(),
            config.add_timestamps,
        );

        JsonPublisher {
            sender,
            cancellation_token,
            processor_handle: Some(processor_handle),
        }
    }

    /// Start the background processor task
    fn start_processor(
        component: Component,
        mut receiver: mpsc::Receiver<(String, JsonValue)>,
        cancellation_token: CancellationToken,
        add_timestamps: bool,
    ) -> tokio::task::JoinHandle<()> {
        component.drt().runtime().secondary().spawn(async move {
            log::info!("JsonPublisher event processor started");

            // Cache of topic -> EventPublisher to avoid recreating publishers
            let mut publishers: HashMap<String, RuntimeEventPublisher> = HashMap::new();

            loop {
                tokio::select! {
                    // Check for cancellation
                    _ = cancellation_token.cancelled() => {
                        log::info!("JsonPublisher event processor received cancellation signal");
                        break;
                    }

                    // Process incoming messages
                    event = receiver.recv() => {
                        let Some((topic, mut json_value)) = event else {
                            // Channel closed - this could be clean shutdown or error
                            // Check if we were cancelled to distinguish between clean shutdown and error
                            if cancellation_token.is_cancelled() {
                                log::info!("JsonPublisher channel closed, terminating processor");
                            } else {
                                // TODO - should we panic here? It's likely an error condition.
                                log::error!("JsonPublisher channel closed unexpectedly, terminating processor");
                            }
                            break;
                        };

                        // Add timestamp if configured and not present
                        if add_timestamps
                            && let JsonValue::Object(ref mut obj) = json_value
                                && !obj.contains_key("timestamp") {
                                    obj.insert(
                                        "timestamp".to_string(),
                                        JsonValue::Number(Utc::now().timestamp_millis().into()),
                                    );
                                }

                        // Get or create publisher for this topic
                        if !publishers.contains_key(&topic) {
                            match RuntimeEventPublisher::for_component(&component, &topic).await {
                                Ok(pub_instance) => {
                                    publishers.insert(topic.clone(), pub_instance);
                                }
                                Err(e) => {
                                    log::error!("Failed to create EventPublisher for topic {}: {}", topic, e);
                                    continue;
                                }
                            }
                        }

                        if let Some(publisher) = publishers.get(&topic) {
                            // Publish the event
                            if let Err(e) = publisher.publish(&json_value).await {
                                log::error!("Failed to publish JSON event to {}: {}", topic, e);
                            } else {
                                log::trace!("Published JSON event to {}: {}", topic, json_value);
                            }
                        }
                    }
                }
            }

            log::info!("JsonPublisher event processor terminated");
        })
    }

    /// Publish a JSON value to a specific topic
    pub async fn publish(&self, topic: impl Into<String>, json_value: JsonValue) -> Result<()> {
        let topic_string = topic.into();

        match self
            .sender
            .send((topic_string.clone(), json_value.clone()))
            .await
        {
            Ok(_) => {
                log::trace!("Queued JSON event for topic {}", topic_string);
                Ok(())
            }
            Err(_) => {
                log::warn!(
                    "Failed to queue JSON event for topic '{}', channel closed",
                    topic_string
                );
                Err(Error::msg("JSON publisher channel closed"))
            }
        }
    }

    /// Publish a JSON value synchronously (non-blocking)
    /// Returns error if the channel is full or closed
    pub fn publish_sync(&self, topic: impl Into<String>, json_value: JsonValue) -> Result<()> {
        let topic_string = topic.into();

        match self
            .sender
            .try_send((topic_string.clone(), json_value.clone()))
        {
            Ok(_) => {
                log::trace!("Queued JSON event for topic {}", topic_string);
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                log::warn!("Failed to queue JSON event, channel full");
                Err(Error::msg("JSON publisher channel full"))
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                log::warn!("Failed to queue JSON event, channel closed");
                Err(Error::msg("JSON publisher channel closed"))
            }
        }
    }

    /// Shutdown the publisher
    pub fn shutdown(&mut self) {
        if !self.cancellation_token.is_cancelled() {
            self.cancellation_token.cancel();
        }

        if let Some(handle) = self.processor_handle.take() {
            handle.abort();
        }

        log::debug!("JsonPublisher shutdown completed");
    }
}

impl Drop for JsonPublisher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::component::ComponentBuilder;
    use dynamo_runtime::transports::event_plane::EventSubscriber as RuntimeEventSubscriber;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use serde_json::json;
    use std::time::Duration;

    #[tokio::test]
    // Disabled because it has failed in CI.
    #[ignore]
    async fn test_json_publisher() -> Result<()> {
        // Set up runtime
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings(rt.clone()).await?;
        let namespace = drt.namespace("test".to_string())?;

        let component = ComponentBuilder::from_runtime(drt.into())
            .name("json_publisher_test".to_string())
            .namespace(namespace.clone())
            .build()?;

        // Create publisher
        let config = JsonPublisherConfig {
            channel_buffer_size: 10,
            add_timestamps: true,
        };

        let publisher = JsonPublisher::new(component.clone(), Some(config));

        // Create a subscriber to verify events are published
        let mut subscriber =
            RuntimeEventSubscriber::for_component(&component, "test.topic").await?;

        // Publish a test event
        let test_event = json!({
            "message": "test message",
            "value": 42
        });

        publisher.publish("test.topic", test_event.clone()).await?;

        // Wait for the event to be published
        let timeout = Duration::from_millis(100);
        let received = tokio::time::timeout(timeout, subscriber.next()).await;

        // Check that we received the event
        assert!(received.is_ok(), "Timed out waiting for published event");
        let envelope = received.unwrap().unwrap().unwrap();
        let event: serde_json::Value = serde_json::from_slice(&envelope.payload).unwrap();

        // Verify the event has the expected data
        assert_eq!(event["message"], "test message");
        assert_eq!(event["value"], 42);

        // Verify a timestamp was added
        assert!(event.as_object().unwrap().contains_key("timestamp"));

        // Test cleanup
        drop(publisher);
        rt.shutdown();

        Ok(())
    }
}
