# Labelling guide: relevance of news articles (version 3)

This guide defines the labels used in the files in `data/eval/`. The same rules apply to development data and test data. Earlier versions keep their original rules in `LABELLING_GUIDE_v1.md` and `LABELLING_GUIDE_v2.md`.

## The question

For each article: **would an equity analyst researching the company want this article in a summary of the week's news?**

## The text that is judged

The label is based on the text the tool uses:

- **Full text**, when the article page can be fetched. This is the cleaned main text of the page, shown on the reading page with the status "complete". Page junk (subscription boxes, "Also Read" links, lists of other stories) is already removed and does not count if any is left.
- **The short NewsAPI text** (title, description and the first 200 characters), when the page cannot be fetched or its status says "older text, page cleaning unavailable". The status is shown for every article.
- **The short NewsAPI text**, also when the fetched page holds less text than the short NewsAPI text (a teaser or a login page). The reading page shows the reason.

The labeller does not open links or use outside knowledge. The label does not guess what is not in the text.

**The article is labelled as a whole, by its research value.** The number of times the company is named does not decide the label. A long article about something else, with one paragraph that gives real information about the company (for example, its sales figure during a festival week), can be relevant. An article that names the company once without any information about its business is not.

## Labels

| Label | Rule |
|---|---|
| `direct` | The article mainly discusses the company, one of its subsidiaries, its current products, operations, financial results, share price or major business decisions. |
| `indirect` | The article is not mainly about the company, but it gives real information connected to the company's business. At least one of the conditions below is true. |
| `not` | The article has no clear business connection to the company, or it names the company only in passing. |

An article is `indirect` when at least one of these is true:

1. It discusses a competitor, supplier or partner in a business segment where the company operates, and the development can reasonably affect that segment or the company.
2. It reports demand, sales, prices or regulations for a market **and** a business segment where the company operates.
3. It reports a cost, technology or supply-chain development that can reasonably affect the company's business.
4. It names the company as part of a real industry comparison (for example, market shares or sales figures of several carmakers), or it connects the company's share movement to a development in its sector (see "Share prices and market reports").

A mention of the company alone is not enough. The same country, the same technology or a similar company name is not enough either.

Order of questions for each article:

1. Is it mainly about the company, a subsidiary or a current product? Then `direct`.
2. If not: is at least one `indirect` condition true? Then `indirect`.
3. Otherwise: `not`.

## Company scope: Tata Motors

Since the demerger (effective 1 October 2025) there are two listed companies:

- Tata Motors Passenger Vehicles (TMPV): passenger cars, electric vehicles, and its subsidiary Jaguar Land Rover (Jaguar, Range Rover, Defender, Discovery).
- Tata Motors Limited (TMCV): trucks, buses and other commercial vehicles.

News about either company or about Jaguar Land Rover counts as news about Tata Motors. Other Tata Group companies (Tata Steel, Tata Elxsi, Tata Consultancy Services, Tata Sons) are separate businesses.

## Company scope: Reliance Industries

Reliance Industries Limited (RIL) and the companies it controls count as the company. Control decides, not a separate listing: a subsidiary with its own shares on the stock market still counts while RIL controls it.

- Reliance Industries itself: the oil-to-chemicals business (refining, fuels, petrochemicals) and the oil and gas business.
- Jio Platforms and its subsidiary Reliance Jio Infocomm (telecom and digital services). Jio Platforms lists its shares on 28 October 2026, and RIL keeps control.
- Reliance Retail Ventures and its businesses, such as Reliance Retail, JioMart and Reliance Consumer Products.
- Reliance New Energy (solar, batteries and other new energy businesses).

Jio Financial Services (a separate listed company since 2023) and the Anil Ambani group companies (Reliance Power, Reliance Infrastructure, Reliance Communications) are separate companies. News about them does not count as news about RIL. The usual rules still apply: such an article is `indirect` when it meets one of the conditions 1 to 4, for example a real connection to RIL's business, and `not` otherwise. A similar name does not mean the same business.

When an article names a business outside this list, the article's evidence decides whether RIL controls it. A minority investment or a business partnership alone does not make it part of RIL. When control is unclear, the best-supported label is chosen, with `unsure = yes` and the reason in the note.

| Article | Label | Reason |
|---|---|---|
| RIL quarterly results | `direct` | Company's own results |
| Jio Platforms share listing, or a Jio tariff change | `direct` | Subsidiary's business |
| Reliance Retail store openings or JioMart sales | `direct` | Subsidiary's business |
| Bharti Airtel raises mobile tariffs, and the article discusses the effect on the telecom market or on Jio | `indirect` | Condition 1: a competitor's development with a stated effect on the telecom market or Jio's business |
| A Bharti Airtel announcement with no effect on the telecom market, such as a new office | `not` | The same industry alone is not enough |
| Refining margins for Indian refiners fall, with the reasons and their effect on refiners' earnings | `indirect` | Condition 3: a cost development that can affect RIL's refining business |
| A daily fuel-price table with no development or explanation | `not` | No development that affects the business |
| Sensex round-up that only lists RIL's share price | `not` | No company or sector development |
| Jio Financial Services results | `not` | Separate company, no condition met |
| Reliance Power or Reliance Infrastructure news | `not` | Separate company, no condition met |

## Share prices and market reports

| Label | Rule |
|---|---|
| `direct` | The article mainly discusses the company's own share movement, stock performance or valuation, or the reasons for it. |
| `indirect` | The article discusses a development in the company's sector and itself connects that development to the company's share movement. The company does not need to be the main subject. |
| `not` | A general market report only lists the company's share price or percentage change, without a company or sector development that explains it. |
| `not` | A general market explanation and the company's share movement appear together, but the article does not connect them. |

The connection must be written in the article. A connection that only the reader infers does not count. When the text suggests a connection but does not state it clearly, the best-supported label is chosen with `unsure = yes`.

| Article | Label |
|---|---|
| Tata Motors shares fall after weak quarterly results | `direct` |
| Auto stocks fall after an RBI rate decision; TMPV falls 2% with other carmakers | `indirect` |
| Sensex round-up: "TMPV fell 2.42 per cent to ₹279.90", no reason given | `not` |
| Sensex falls on global worries; TMPV appears in a list of losers | `not` |
| A list of stocks at 52-week lows that includes TMPV, with no reason for its fall | `not` |

## Examples (Tata Motors)

| Article | Label | Reason |
|---|---|---|
| Quarterly results of TMPV or TMCV | `direct` | Company's own results |
| Jaguar Land Rover sales, job cuts or a new showroom | `direct` | Subsidiary's business |
| Review or road test of a model sold today (Nexon EV, current Range Rover) | `direct` | Current product |
| Recall, price change or major feature of a current model | `direct` | Current product |
| Festival-season car sales in India, with Tata Motors' figures in the text | `indirect` | Conditions 2 and 4 |
| A competitor's new car or plant in India | `indirect` | Condition 1 |
| A supplier to Tata Motors faces a problem that can affect its supplies | `indirect` | Condition 1 |
| Chip shortage for carmakers | `indirect` | Condition 3 |
| Auction or private sale of an old or classic Jaguar | `not` | No effect on the business today |
| Review of a model no longer sold | `not` | Not a current product, unless the article links it to current products or decisions |
| General electric-vehicle news with no link to the company's markets or segments | `not` | Same technology only |
| Robotaxi service in a city where the company does not operate | `not` | Unrelated market and business |
| Biography that mentions a past job at Tata Motors | `not` | Passing mention |
| An essay that mentions a past event in Tata Motors' history | `not` | Passing mention, no current business information |
| A start-up or supplier profile that lists Tata Motors among its customers, about something that does not affect Tata Motors | `not` | Passing mention |
| A results calendar that only lists Tata Motors among companies reporting that day | `not` | No information about the company's business; `indirect` if it gives Tata Motors' expected or actual figures |
| Tata Steel, Tata Elxsi or Tata Sons news | `not` | Separate company |

## Borderline cases

**Sector-wide moves.** A story about a move across the whole sector (for example, all auto stocks falling after an RBI decision) that names the company among others is `indirect`, with `unsure = yes`, even when the company is named first.

**Subsidiaries and brands.** News about a subsidiary inside the research scope, or about its current products, is `direct`. News about a separate company with a similar name is `not`, unless the article shows a business link. A similar name does not mean the same business.

**Vehicles of the company's brands.** Only current models count as the company's products. When it is unclear whether a model is still sold, the best-supported label is chosen, with `unsure = yes` and a note.

**Quotes about the company.** A quote about the company in an article on another subject (for example, a minister's remark on Tata Motors' past) counts when it gives information relevant to the company's business today, such as policy plans that affect its market. A remark only about the past is a passing mention.

## Unsure cases

The `unsure` column is `yes` only when the text does not support a clear choice between two labels. The label column still holds the best choice, and the `note` column gives the reason. Unsure cases are reported separately in the results.

## Fairness rules

- Labels are given without seeing what the tool kept or dropped.
- Labels are given without knowing which search found the article.
- Labels are given without looking at earlier label files.
- Rows are shuffled, so position gives no hint.
- The same rule is used for similar articles across all files.
- Labels are not changed to improve a score. A rule change applies to every file, and every changed label is recorded with its reason and guide version.
- When only the labels change, a change in the scores comes from the labels, not from the tool.

## Label history

- Version 1: `dev_tata_motors_20261023.csv` (20 articles) and `dev_tata_motors_20261014.csv` (38 articles), labelled blind. In the second file, nine rows carry corrections to follow version 1. The corrections come before any results on that file. Details are in `LABELLING_GUIDE_v1.md`.
- Version 2: the rule for `indirect` needs a clear business connection (conditions 1 to 4), with rules for vehicle reviews, old vehicles and subsidiaries. All 58 earlier articles receive new blind labels under version 2. The first-pass agreement with version 1 is 0.983 (Cohen's kappa: 0.974). After the guide check, agreement with version 1 reaches 1.000. Details are in `LABELLING_GUIDE_v2.md`.
- Version 3: labels are based on the cleaned full text when it is available. New rules for share prices and market reports, passing mentions, results calendars and quotes, and the rule that an article is labelled as a whole by its research value. Every article with full text is labelled again on its full text; the 3 articles whose pages cannot be fetched (blocked by the site's robots.txt) keep their version 2 label, based on the short text. The labeller already knows the version 2 labels, so version 3 labelling is not fully independent. Each difference from version 2 has a recorded reason based on this guide. On the 71 articles with full text, agreement with version 2 is 0.944 (Cohen's kappa: 0.914): 4 labels change, each with its reason in `differences_v2_v3_20261024.csv`. The same person gives both versions, so this agreement shows consistency, not independent confirmation. The final version 3 labels for all 104 articles are in `v3_dev_tata_motors.csv`.
- Version 3 adds the company scope for Reliance Industries, for the transfer test. The rules are unchanged, so no Tata Motors label changes.
