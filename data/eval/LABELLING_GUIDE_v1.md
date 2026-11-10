# Labelling guide: relevance of news articles (version 1)

This guide defines the version 1 labels, used in `dev_tata_motors_20261023.csv` and `dev_tata_motors_20261014.csv`. The current rules are in `LABELLING_GUIDE.md`.

## The question

For each article: **would an equity analyst researching the company want this article in a summary of the week's news?**

Version 1 labels use only the short NewsAPI text: the title, the description and the first 200 characters of content. Full article pages are not opened during version 1 labelling.

## Labels

| Label | Rule | Examples (Tata Motors) |
|---|---|---|
| `direct` | The article is mainly about the company or one of its businesses. Without the company, there would be no story. | Results, deals (the Iveco takeover), launches, management changes, share price moves, Jaguar Land Rover news |
| `indirect` | The article is not mainly about the company, but it can affect the company's business or share price. | Sales of the whole car industry, a competitor's new model, a supplier's or partner's problem, a stock list that includes TMPV or TMCV |
| `not` | The article is not useful for research on the company. | Crime, entertainment, politics with no business link, other Tata Group companies (Tata Steel, Tata Elxsi), general market news that does not mention the auto sector |

Order of questions for each article:

1. Is it mainly about the company or one of its businesses? Then `direct`.
2. If not: is it about the company's industry, competitors, suppliers, partners, or a stock list that includes the company, in a way that matters for its business? Then `indirect`.
3. Otherwise: `not`.

## Company scope: Tata Motors

Since the demerger (effective 1 October 2025) there are two listed companies: Tata Motors Passenger Vehicles (TMPV: passenger cars, electric vehicles, Jaguar Land Rover) and Tata Motors Limited (TMCV: trucks, buses, commercial vehicles). News about either company counts as news about Tata Motors.

## Relevant or not, for scoring

`direct` and `indirect` both count as relevant. `not` counts as not relevant.

## Borderline case: sector-wide moves

A story about a move across the whole sector (for example, all auto stocks falling after an RBI decision) that names the company among others is `indirect`, marked `unsure = yes`, even when the company is named first.

## Unsure cases

The `unsure` column is marked `yes` when the choice between two labels is close. The label column still holds the best choice, and the `note` column gives the reason. Unsure cases are reported separately in the results.

## Fairness rules

- Labels are given without seeing what the tool kept or dropped.
- Rows are shuffled, so position gives no hint.
- The same rule is used for similar articles across all files.
- Labels are not changed after seeing the tool's results.

## Label history

- `dev_tata_motors_20261023.csv` (20 articles): labelled blind. Checked against these rules; no changes.
- `dev_tata_motors_20261014.csv` (38 articles): labelled blind. The version 1 record shows a check against these rules before any evaluation results. Nine rows contain corrections to follow the rules: row 30 (stock list including TMPV) to `indirect`; rows 4, 9, 14, 23, 27, 29 and 36 (other groups, earnings calendars, general market news, farm equipment, an in-car gaming deal) to `not`; row 5 (sector-wide fall in auto stocks) to `indirect` with `unsure = yes`.
