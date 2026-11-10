# Labelling guide: relevance of news articles (version 2)

This guide defines the version 2 labels, used in `v2_dev_tata_motors.csv` and `recheck_v2_dev_tata_motors.csv`. The same rules apply to development data and test data. The current rules are in `LABELLING_GUIDE.md`, and version 1 keeps its original rules in `LABELLING_GUIDE_v1.md`.

## The question

For each article: **would an equity analyst researching the company want this article in a summary of the week's news?**

Version 2 labels use only the short NewsAPI text: the title, the description and the first 200 characters of content. Full article pages are not opened during version 2 labelling. The label does not guess what the rest of the article says.

## Labels

| Label | Rule |
|---|---|
| `direct` | The article mainly discusses the company, one of its subsidiaries, its current products, operations, financial results, share price or major business decisions. |
| `indirect` | The article is not mainly about the company, but it has a clear connection to the company's business. At least one of the conditions below is true. |
| `not` | The article has no clear business connection to the company, or it mentions the company only in passing or in unrelated website text. |

An article is `indirect` when at least one of these is true:

1. It discusses a competitor, supplier or partner in a business segment where the company operates, and the development can reasonably affect that segment or the company.
2. It reports demand, sales, prices or regulations for a market **and** a business segment where the company operates.
3. It reports a cost, technology or supply-chain development that can reasonably affect the company's business.
4. It names the company as part of a real industry comparison or development (for example, market shares, or a list of stocks that moved).

A mention of the company alone is not enough. The same country, the same technology or a similar company name is not enough either.

Order of questions for each article:

1. Is it mainly about the company, a subsidiary or a current product? Then `direct`.
2. If not: is at least one `indirect` condition true? Then `indirect`.
3. Otherwise: `not`.

## Company scope: Tata Motors

Since the demerger (effective 1 October 2025) there are two listed companies:

- Tata Motors Passenger Vehicles (TMPV): passenger cars, electric vehicles, and its subsidiary Jaguar Land Rover (Jaguar, Range Rover, Defender, Discovery).
- Tata Motors Limited (TMCV): trucks, buses and other commercial vehicles.

News about either company or about Jaguar Land Rover counts as news about Tata Motors. Other Tata Group companies (Tata Steel, Tata Elxsi, Tata Consultancy Services) are separate businesses.

## Examples (Tata Motors)

| Article | Label | Reason |
|---|---|---|
| Quarterly results of TMPV or TMCV | `direct` | Company's own results |
| Jaguar Land Rover sales or a new showroom | `direct` | Subsidiary's business |
| Review or road test of a model sold today (Nexon EV, current Range Rover) | `direct` | Current product |
| Recall, price change or major feature of a current model | `direct` | Current product |
| Car sales across India during a festival season | `indirect` | Condition 2: demand in its market and segment |
| A competitor's new car or plant in India | `indirect` | Condition 1 |
| Chip shortage for carmakers | `indirect` | Condition 3 |
| List of stocks at 52-week lows that includes TMPV | `indirect` | Condition 4 |
| Auction or private sale of an old or classic Jaguar | `not` | No effect on the business today |
| Review of a model no longer sold | `not` | Not a current product, unless the article links it to current products or decisions |
| General electric-vehicle news with no link to the company's markets or segments | `not` | Same technology only |
| Robotaxi service in a city where the company does not operate | `not` | Unrelated market and business |
| Biography that mentions a past job at Tata Motors | `not` | Passing mention |
| Crime story with Tata Motors only in website text | `not` | Unrelated website text |
| Tata Steel or Tata Elxsi news | `not` | Separate company |

## Borderline cases

**Sector-wide moves.** A story about a move across the whole sector (for example, all auto stocks falling after an RBI decision) that names the company among others is `indirect`, with `unsure = yes`, even when the company is named first.

**Subsidiaries and brands.** News about a subsidiary inside the research scope, or about its current products, is `direct`. News about a separate company with a similar name is `not`, unless the article shows a business link. A similar name does not mean the same business.

**Vehicles of the company's brands.** Only current models count as the company's products. When it is unclear whether a model is still sold, the best-supported label is chosen, with `unsure = yes` and a note.

## Unsure cases

The `unsure` column is `yes` only when the visible text does not support a clear choice between two labels. The label column still holds the best choice, and the `note` column gives the reason. Unsure cases are reported separately in the results.

## Fairness rules

- Labels are given without seeing what the tool kept or dropped.
- Labels are given without knowing which search found the article.
- Rows are shuffled, so position gives no hint.
- The same rule is used for similar articles across all files.
- Labels are not changed to improve a score. A rule change applies to every file, and every changed label is recorded.
- When only the labels change, a change in the scores comes from the labels, not from the tool.

## Label history

- Version 1: `dev_tata_motors_20261023.csv` (20 articles) and `dev_tata_motors_20261014.csv` (38 articles), labelled blind. In the second file, nine rows carry corrections to follow version 1. The corrections come before any results on that file. Details are in `LABELLING_GUIDE_v1.md`.
- Version 2: the rule for `indirect` now needs a clear business connection (conditions 1 to 4), and rules for vehicle reviews, old vehicles and subsidiaries are added. All 58 earlier articles are labelled again, blind, under version 2. The version 1 files stay unchanged, and every difference between the two versions is recorded with its reason.
- Result of the version 2 relabelling (58 articles, file `recheck_v2_dev_tata_motors.csv`):
  - First blind pass: 57 of 58 labels equal to version 1 (agreement 0.983, Cohen's kappa 0.974). The first pass has one different label: Schaeffler India (a supplier named as serving Tata Motors, with its CEO in custody) receives `not`.
  - Check against the guide: row 15 (a supplier joint venture) is `indirect` under condition 1, so the Schaeffler row, where the text names Tata Motors as a customer, is changed to `indirect` with `unsure = yes` for consistency. After this check, all 58 labels equal version 1 (agreement 1.000).
  - `unsure = yes` is added to three rows: the Schaeffler row and the two copies of the global used-truck market report (geographic link unclear).
  - The same person labels both versions within one week, including the 20-article file on the same day. The agreement measures consistency, not independent confirmation.
  - No label changes under the stricter `indirect` rule, so the version 1 and version 2 scores are the same.
