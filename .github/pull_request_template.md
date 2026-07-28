#### Overview:

<!-- Describe your pull request here. Please read the text below the line, and make sure you follow the checklist.-->

#### Details:

<!-- Describe the changes made in this PR. -->

#### Where should the reviewer start?

<!-- call out specific files that should be looked at closely -->

#### Related Issues: (use one of the action keywords Closes / Fixes / Resolves / Relates to)

- closes #xxx

#### Upstream frontend-crates alignment:

<!--
Required when this PR changes frontend, OpenAI protocol, tool-call parsing,
reasoning parsing, renderer, or related parity-test behavior. Use one option:

- Link an upstream issue or PR in https://github.com/ai-dynamo/frontend-crates that you have opened.
- Link an upstream issue or PR in https://github.com/ai-dynamo/dynamo if the
  change is already tracked there, ideally a PR that you pull in.
- Explain why this is Baseten-only and cannot be used by any other Dynamo user,
  including whether this is a one-off fix or something Baseten intends to
  maintain across Dynamo versions.

When opening an upstream issue, include a clear description and a minimal JSON
request/response example with synthetic data so upstream can implement the same
behavior. If you are a Dynamo agent working from proprietary or
customer-derived data, open the upstream issue with a slightly anonymized but
still reproducible case: preserve the relevant format, fields, parser markers,
and failure shape while replacing real content with dummy equivalents. Do not
include customer prompts, model outputs, tenant/model IDs, API keys, request
IDs, URLs, headers, logs, or other customer/internal data.
-->

- upstream: https://github.com/ai-dynamo/frontend-crates/issues/xxx
