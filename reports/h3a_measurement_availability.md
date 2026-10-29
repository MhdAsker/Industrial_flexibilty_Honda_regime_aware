# H3A measurement availability at forecast issuance

## Candidate issuance

The H3A candidate forecast is issued at 13:00 Europe/Berlin on D-1 for every physical 15-minute interval of local calendar day D. The time is aligned to the 15-minute cadence and is never ambiguous during European DST transitions because clock changes occur during the early morning, not at 13:00. It is 12:00 UTC in winter and 11:00 UTC in summer.

The issuance is operationally compatible with a later post-Day-Ahead-auction analysis. H3B must still verify and record the exact market, auction gate closure, result-publication convention, and historical rule applicable to the intended decision process. H3A does not freeze a market rule.

## Left-labelled measurement semantics

Honda non-cumulative 15-minute values are left-labelled arithmetic means. A value labelled 12:45 represents the physical interval `[12:45, 13:00)`. It is complete only at 13:00. A value labelled 13:00 represents `[13:00, 13:15)` and is not available at a 13:00 issuance.

The candidate historical-measurement cutoff is therefore:

`interval_end_utc <= forecast_issue_time_utc`

Equivalently for 15-minute left-labelled data:

`interval_start_utc + 15 minutes <= forecast_issue_time_utc`

The rule is evaluated in UTC. Local timestamps are retained for interpretation, but UTC is the unique physical key.

## Concrete winter example

For target day 2023-01-02, issuance is 2023-01-01 13:00 CET, equal to 12:00 UTC.

- label 12:45 CET: interval ends 13:00 CET and is available;
- label 13:00 CET: interval ends 13:15 CET and is unavailable;
- latest usable grid measurement starts 11:45 UTC and ends exactly at 12:00 UTC.

## Concrete summer example

For target day 2023-07-02, issuance is 2023-07-01 13:00 CEST, equal to 11:00 UTC.

- label 12:45 CEST: interval ends 13:00 CEST and is available;
- label 13:00 CEST: interval ends 13:15 CEST and is unavailable;
- latest usable grid measurement starts 10:45 UTC and ends exactly at 11:00 UTC.

## Context-window construction

Every context window is a fixed physical duration ending at issuance, with the issuance endpoint excluded from interval starts. A 24-hour context therefore contains 96 intervals from `issuance - 24 h` through `issuance - 15 min`. No context window crosses into an unfinished or future interval. H3A performs no filling: observed-original and corrected-nonimputed counts are separate, and imputed, unknown, unresolved, or absent intervals remain unavailable.

## Target-day lead range

At 13:00 D-1, the first target interval starts 11 physical hours after issuance. The last target interval starts:

- 34.75 hours after issuance on an ordinary 96-interval day;
- 33.75 hours after issuance on a 92-interval spring DST day;
- 35.75 hours after issuance on a 100-interval autumn DST day.

These are interval-start lead times. H3B may later define lead-time bands, but H3A does not.
