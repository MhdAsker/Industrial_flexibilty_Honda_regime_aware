# Honda baseline DST audit

The protocol index contains four spring days with 92 physical intervals and four autumn days with 100 physical intervals across 2020–2023. UTC is the unique key.

B1 searches earlier local calendar dates by local quarter-hour and fold. B2 maps exactly seven local dates earlier using the same key. B3 applies the same mapping for 7k-day lags. The second autumn occurrence (fold=1) is distinct and is never collapsed onto fold=0. If an exact fold analogue does not exist, the prediction is unavailable rather than fabricated.

The 2023 spring and autumn days retain 92 and 100 unique UTC targets respectively. Leakage checks confirm every used historical interval completed by issuance.
