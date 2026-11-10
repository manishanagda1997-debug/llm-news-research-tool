# Summary judging guide, Part 6: fresh-week test

This part applies to the summary of the fresh week (24 to 30 October) made with the frozen summary design (`summary_freeze.json`: direct, k = 5, prompt version 4, configuration 7, pacer version 3). Parts 1 to 4 are in `SUMMARY_JUDGING_GUIDE.md` and `SUMMARY_JUDGING_GUIDE_PART4.md`; they stay unchanged and still apply.

## What is fresh

This is a fresh-week test of the frozen summary design. The week is not used to select or tune the summary prompt, configuration or passage rules. Stage 3 evaluates the ranking method on this same week separately. Its labels and event sheet come from Stage 3 and are not changed here.

## Files

- `summary_v2_fresh_k5_c7_p4.json`: the run record.
- `summary_v2_fresh_judging.csv`: every item of the summary (company points, industry points and article briefs), shuffled. No item keeps a judgement from an earlier sheet.
- `summary_v2_fresh_events.csv`: one row per relevant event among the five selected articles, from the fresh week's event sheet.
- `summary_v2_fresh_industry.csv`: one row.
- `data/fulltext/summary_v2_fresh_judging.html`: the reading page.

## Order of judging

1. Industry evidence, from the five selected articles alone (Part 4).
2. Items (Part 2).
3. Event coverage (Part 3).

The sheets are locked before the score is made.

## Pass rule

1. A summary is produced.
2. Event coverage is at least 80%.
3. Contradicted items are at most 5% of all judged items, calculated without rounding. With fewer than 20 items, a single contradicted item is above 5%.
4. No contradicted item is central.
5. No industry point is not supported.
6. An industry section with points passes only when `industry_evidence` is yes; an empty industry section passes only when it is no.

## What the results report

The supported share of key points and briefs together, as a count and a percentage, next to the development result for prompt version 4 (32 of 40, 80.0%). One summary gives few items, so this comparison has no pass level and is not a reliable statistical comparison.

## After the result

The frozen design stays, whatever the result. A failed check is reported with its diagnosis. A change to the design belongs to a new development cycle and needs a new fresh test.
