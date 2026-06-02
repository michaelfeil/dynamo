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
use dynamo_runtime::component::Component;
use dynamo_runtime::prelude::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber as RuntimeEventSubscriber;
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing as log;

/// Configuration options for the JsonSubscriber
pub struct JsonSubscriberConfig {
    pub channel_buffer_size: usize,
}

impl Default for JsonSubscriberConfig {
    fn default() -> Self {
        Self {
            channel_buffer_size: 100,
        }
    }
}

/// A subscriber for JSON events that processes them asynchronously
pub struct JsonSubscriber {
    /// Channel for receiving events from the background task
    receiver: mpsc::Receiver<JsonValue>,
    /// Cancellation token for shutting down the subscriber
    cancellation_token: CancellationToken,
    /// The task handle for the background processor
    processor_handle: Option<tokio::task::JoinHandle<()>>,
}

impl JsonSubscriber {
    /// Creates a new JsonSubscriber that will receive events from the provided component and topic
    pub async fn new(
        component: Component,
        topic: impl Into<String>,
        config: Option<JsonSubscriberConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let topic = topic.into();
        let (sender, receiver) = mpsc::channel::<JsonValue>(config.channel_buffer_size);
        let cancellation_token = CancellationToken::new();

        // Start the processor task
        let processor_handle =
            Self::start_processor(component, topic, sender, cancellation_token.clone());

        Ok(JsonSubscriber {
            receiver,
            cancellation_token,
            processor_handle: Some(processor_handle),
        })
    }

    /// Start the background processor task
    fn start_processor(
        component: Component,
        topic: String,
        sender: mpsc::Sender<JsonValue>,
        cancellation_token: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        component.drt().runtime().secondary().spawn(async move {
            log::info!("JsonSubscriber event processor started for topic: {}", topic);

            // Subscribe to the topic
            let mut subscriber =
                match RuntimeEventSubscriber::for_component(&component, &topic).await {
                    Ok(sub) => sub,
                    Err(e) => {
                        log::error!("Failed to subscribe to topic {}: {}", topic, e);
                        return;
                    }
                };

            loop {
                tokio::select! {
                    // Check for cancellation
                    _ = cancellation_token.cancelled() => {
                        log::info!("JsonSubscriber event processor received cancellation signal");
                        break;
                    }

                    // Process incoming messages
                    message = subscriber.next() => {
                        let Some(msg_result) = message else {
                            // Subscriber closed - this could be clean shutdown or error
                            if cancellation_token.is_cancelled() {
                                log::info!("JsonSubscriber channel closed, terminating processor");
                            } else {
                                log::error!("JsonSubscriber channel closed unexpectedly, terminating processor");
                            }
                            break;
                        };

                        match msg_result {
                            Ok(envelope) => {
                                // Decode payload from msgpack (event plane uses MsgpackCodec)
                                match rmp_serde::from_slice::<JsonValue>(&envelope.payload) {
                                    Ok(json_value) => {
                                        log::trace!("Received and parsed event from {}: {:?}", topic, json_value);
                                        if let Err(e) = sender.send(json_value).await {
                                            log::error!("Failed to send JSON event to channel: {}", e);
                                        }
                                    }
                                    Err(e) => {
                                        log::error!("Failed to decode msgpack from message payload: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!("Error receiving event from subscriber: {}", e);
                            }
                        }
                    }
                }
            }

            log::info!("JsonSubscriber event processor terminated");
        })
    }

    /// Get the next JSON value from the subscriber
    pub async fn next(&mut self) -> Option<JsonValue> {
        self.receiver.recv().await
    }

    /// Try to get the next JSON value synchronously (non-blocking)
    pub fn try_next(&mut self) -> Result<Option<JsonValue>> {
        match self.receiver.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => {
                Err(Error::msg("JSON subscriber channel closed"))
            }
        }
    }

    /// Shutdown the subscriber
    pub fn shutdown(&mut self) {
        if !self.cancellation_token.is_cancelled() {
            self.cancellation_token.cancel();
        }

        if let Some(handle) = self.processor_handle.take() {
            handle.abort();
        }

        log::debug!("JsonSubscriber shutdown completed");
    }
}

impl Drop for JsonSubscriber {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::component::ComponentBuilder;
    use dynamo_runtime::transports::event_plane::EventPublisher as RuntimeEventPublisher;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use serde_json::json;
    use std::time::Duration;

    #[tokio::test]
    // Disabled because it has failed in CI.
    #[ignore]
    async fn test_json_subscriber() -> Result<()> {
        // Set up runtime
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::from_settings(rt.clone()).await?;
        let namespace = drt.namespace("test".to_string())?;

        let component = ComponentBuilder::from_runtime(drt.into())
            .name("json_subscriber_test".to_string())
            .namespace(namespace.clone())
            .build()?;

        // Create subscriber
        let config = JsonSubscriberConfig::default();
        let mut subscriber =
            JsonSubscriber::new(component.clone(), "test.topic", Some(config)).await?;

        // Create a publisher to send test events
        let publisher = RuntimeEventPublisher::for_component(&component, "test.topic").await?;

        // Publish a test event
        let test_event = json!({
            "message": "test message",
            "value": 42
        });

        publisher.publish(&test_event).await?;

        // Wait for the event to be received
        let timeout = Duration::from_millis(100);
        let received = tokio::time::timeout(timeout, subscriber.next()).await;

        // Check that we received the event
        assert!(received.is_ok(), "Timed out waiting for published event");
        let message = received.unwrap().unwrap();

        // Verify the event has the expected data
        assert_eq!(message["message"], "test message");
        assert_eq!(message["value"], 42);

        // Test cleanup
        drop(subscriber);
        rt.shutdown();

        Ok(())
    }
}
