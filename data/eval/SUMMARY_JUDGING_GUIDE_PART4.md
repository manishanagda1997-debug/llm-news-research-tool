# Summary judging guide, Part 4: industry-context check

This guide applies only to the summaries made with prompt version 4. Parts 1 to 3 are in `SUMMARY_JUDGING_GUIDE.md`; they stay unchanged and still apply.

## What changes in the summaries

An industry-context point needs an article that itself reports on the wider industry, the market or other companies. A fact about Tata Motors alone is not industry context. When no selected article reports on the wider industry, the industry-context section is empty and the app states: "The selected articles do not report on the wider industry."

## Files

- `summary_v2_p4_judging.csv`: the points and briefs that are new or changed. An item with exactly the same section, text and cited articles as a direct item in the locked sheet keeps its locked judgement and is not in this sheet. The same claim with different wording, different citations or in a different section is judged again.
- `summary_v2_p4_events.csv`: event coverage for the three new summaries, judged as in Part 3.
- `summary_v2_p4_industry.csv`: one row per summary.
- `data/fulltext/summary_v2_p4_judging.html`: the reading page.

## Order of judging

1. Industry evidence: section 1 of the reading page shows the selected articles of each summary, without the summary. The `industry_evidence` column is filled from these articles alone.
2. Items: section 2, judged as in Part 2.
3. Event coverage: section 3, judged as in Part 3.

## Industry evidence (`industry_evidence` column)

The question for each summary: does any selected article contain evidence about the wider industry, the market or other companies?

- yes: at least one article reports a fact about the sector, the market, or a company other than Tata Motors in its own right (for example, a rival's sales or a market-wide demand figure).
- no: every article reports only on Tata Motors, or on other companies only through their part in a Tata Motors deal (for example, Iveco's shareholders in the takeover, Stellantis in the partnership).

The note names the article and the fact for a yes, and gives the reason for a no.

## What the results report

- Industry points: the number produced, and the counts of supported, partly supported and not supported.
- Key points and briefs: supported, partly supported and not supported for each section, and the combined supported share with its numerator and denominator, next to the locked result (29 of 38, 76.3%).
- Items that keep their locked judgement: their number. They count in the results with that judgement.
- Event coverage, as in Part 3.
- Every figure for each run, as well as in total.

## Pass rule

1. No industry point is not supported.
2. An industry section with points passes only when `industry_evidence` for that summary is yes. An empty industry section passes only when `industry_evidence` for that summary is no. Across the three summaries, there is at least one supported industry point, or every section is empty with `industry_evidence` no.
3. Key points and briefs together have no not-supported point and a supported share no lower than the locked share (29 of 38) minus 5 percentage points, calculated without rounding (0.7132).
4. Event coverage is at least 80%.
5. Each of the three development runs (14, 23 and 24 October) has exactly one direct summary, and every one is produced.
