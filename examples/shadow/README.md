# Shadow scheduler onboarding

This directory intentionally contains no synthetic industrial issue. Phase 2 is useful only after a partner supplies and approves its real physical, timing, operational, and commercial inputs.

Copy `partner-readiness.template.yaml` outside the repository, replace every placeholder, and keep each item unapproved until the industrial owner has reviewed the referenced evidence. Then run:

```powershell
shadow-scheduler readiness --partner path\to\partner-readiness.yaml
```

The required `shadow.issue.v1` structure and operating workflow are documented in [`../../docs/phase2_shadow_scheduler.md`](../../docs/phase2_shadow_scheduler.md). Issue files and operational records may contain commercially sensitive information and should normally be stored in a controlled data system, not committed to this public repository.

Before the first eligible prospective issue, copy `phase2_prospective_protocol.template.json` into the partner's controlled workspace, replace every placeholder, obtain the required approvals, and freeze its hash:

```powershell
shadow-scheduler protocol --protocol path\to\partner_protocol_frozen.json
```

Do not enable switchback assignment unless both policies are already approved and the partner has accepted the experimental design.
