# FlexOps platform onboarding

The files here are configuration templates, not examples of validated industrial operation.

Start with `site.template.json`, retain `approved: false`, and replace the placeholders. Approval should only change after the named industrial owner has verified contractual limits, timezone and ownership.

Operational databases, API tokens, connector credentials and partner records must not be committed to the repository. See [`../../docs/phase3_multisite_platform.md`](../../docs/phase3_multisite_platform.md) for the architecture, security boundary and deployment sequence.

`multisite_ml_policy.template.json` freezes target-semantic groups, known-site and cold-start evaluation, promotion gates, and the advanced-model scale boundary. Copy it into controlled storage and replace every placeholder before the first multi-site benchmark. Signed grid exchange and nonnegative demand must remain separate semantic groups.
