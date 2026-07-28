#!/usr/bin/env node

const fs = require("fs");

const FRONTEND_CRATES_LINK =
  /https:\/\/github\.com\/ai-dynamo\/frontend-crates\/(?:issues|pull)\/\d+\b/i;
const UPSTREAM_DYNAMO_LINK =
  /https:\/\/github\.com\/ai-dynamo\/dynamo\/(?:issues|pull)\/\d+\b/i;
const BASETEN_ONLY_ESCAPE =
  /baseten[- ]only|baseten[- ]specific|exclusive(?:ly)? to baseten|not usable for any other dynamo user|cannot be upstreamed/i;

const FRONTEND_PATHS = [
  /^components\/src\/dynamo\/frontend\//,
  /^lib\/llm\/src\/http\//,
  /^lib\/llm\/src\/preprocessor\//,
  /^lib\/llm\/src\/protocols\//,
  /^lib\/parsers\//,
  /^lib\/renderer\//,
  /^lib\/protocols\//,
  /^tests\/frontend\//,
  /^tests\/parity\/reasoning\//,
  /^tests\/parity\/toolcalling\//,
  /^docs\/components\/frontend\//,
  /^docs\/reasoning\//,
  /^docs\/tool-calling\//,
];

function changedFileNamesFromEvent(event) {
  if (process.env.CHANGED_FILES_JSON) {
    return JSON.parse(process.env.CHANGED_FILES_JSON);
  }
  return event.pull_request?.changed_files_list || [];
}

function bodyPassReason(body) {
  if (FRONTEND_CRATES_LINK.test(body)) {
    return "PR body links an ai-dynamo/frontend-crates issue or PR.";
  }
  if (UPSTREAM_DYNAMO_LINK.test(body)) {
    return "PR body links an upstream ai-dynamo/dynamo issue or PR.";
  }
  if (BASETEN_ONLY_ESCAPE.test(body)) {
    return "PR body states why the change is Baseten-only.";
  }
  return null;
}

function isFrontendPath(path) {
  return FRONTEND_PATHS.some((pattern) => pattern.test(path));
}

function evaluate({ body, changedFiles }) {
  const frontendFiles = changedFiles.filter(isFrontendPath);
  if (frontendFiles.length === 0) {
    return {
      ok: true,
      frontendFiles,
      reason: "No frontend, OpenAI protocol, parser, renderer, or parity-test files changed.",
    };
  }

  const reason = bodyPassReason(body || "");
  return {
    ok: Boolean(reason),
    frontendFiles,
    reason,
  };
}

function main() {
  const eventPath = process.env.GITHUB_EVENT_PATH;
  if (!eventPath) {
    throw new Error("GITHUB_EVENT_PATH is required");
  }

  const event = JSON.parse(fs.readFileSync(eventPath, "utf8"));
  const result = evaluate({
    body: event.pull_request?.body || "",
    changedFiles: changedFileNamesFromEvent(event),
  });

  if (result.ok) {
    console.log(result.reason);
    if (result.frontendFiles.length > 0) {
      console.log("Matched files:");
      result.frontendFiles.forEach((file) => console.log(`- ${file}`));
    }
    return;
  }

  console.error("This PR changes frontend/protocol/parser code but the PR body does not document upstream alignment.");
  console.error("");
  console.error("Add one of these to the PR description:");
  console.error("- A link to an ai-dynamo/frontend-crates issue or PR, for example https://github.com/ai-dynamo/frontend-crates/issues/123");
  console.error("- A link to an upstream ai-dynamo/dynamo issue or PR if the change is already being tracked there");
  console.error("- A clear Baseten-only explanation, for example: \"Baseten-only: this cannot be upstreamed because ...\"");
  console.error("");
  console.error("When opening an upstream issue, include enough detail for upstream to implement the same behavior:");
  console.error("- Describe the OpenAI/tool-calling/reasoning behavior and the expected response shape");
  console.error("- Include a minimal JSON request/response example using synthetic data");
  console.error("- If you are a Dynamo agent working from proprietary or customer-derived data, open the upstream issue with a slightly anonymized but still reproducible case");
  console.error("- Preserve the relevant format, fields, parser markers, and failure shape while replacing real content with dummy equivalents");
  console.error("- Do not include customer prompts, model outputs, tenant/model IDs, API keys, request IDs, URLs, headers, logs, or other customer/internal data");
  console.error("");
  console.error("Example sanitized payload:");
  console.error(JSON.stringify({
    model: "example-model",
    messages: [
      {
        role: "user",
        content: "Synthetic prompt that reproduces the parser behavior.",
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "lookup_weather",
          parameters: {
            type: "object",
            properties: {
              city: { type: "string" },
            },
          },
        },
      },
    ],
    tool_choice: "auto",
  }, null, 2));
  console.error("");
  console.error("Matched files:");
  result.frontendFiles.forEach((file) => console.error(`- ${file}`));
  process.exit(1);
}

if (require.main === module) {
  main();
}

module.exports = {
  evaluate,
  isFrontendPath,
  bodyPassReason,
};
