# Summary judging guide

This guide defines how summaries are judged in `data/eval/`. It covers two tasks, done in this order:

1. **Event grouping** (`events_dev_tata_motors.csv`, and `events.csv` in each fresh run folder), before any summary or ranking result is read.
2. **Judging** (`summary_judging_<date>.csv`), after the summaries are made.

## Part 1: Event grouping

An **event** is one development in the world, such as Tata Motors raising its offer for Iveco. Several articles can report the same event. The event column groups them, so that a summary that covers an event through one article is not counted as missing the others. The same groups measure how many different events reach the articles chosen for the summary.

Rules:

- Each row gets one short event name in lower case, for example `iveco_offer_raised` or `navratri_sales_fada`.
- Articles report the same event when they describe the same development at the same stage. Use the same name for them.
- A new stage is a new step in the story: a new decision, action or outcome, such as an offer, a raised offer, a shareholder vote or the final result of the offer. A later stage is a separate event. A number that only reports progress within the same step, such as a new acceptance figure while the offer is still open, keeps the same event name. A number that completes a step, such as acceptance reaching the level the offer needs, is an outcome and starts a new event.
- An article with no partner gets its own event name.
- The title, the description and the first 200 characters give the first view of each article. When they do not show the main development clearly, the full article text decides the event, and the `notes` column gives the reason. When the publisher blocks the full text, the available text decides, and `notes` says so.
- The grouping is finished and saved before any summary file or ranking result is opened.

The `notes` column:

- `notes` gives the reason for a decision that is not obvious: a decision from the full text, a round-up, or a main subject that differs from the headline. Rows with an obvious event leave it empty.
- `notes` explains a decision. It never adds a second event to a row: each row counts under its one event name only.
- `notes` does not replace the record of label corrections. Once a sheet is committed, a changed event name goes into the corrections file with the old name and the reason.

Edge cases:

- **Copies and reposts** of one article get the same name as the original.
- **A time gap alone** does not make a new event. Two reports of the same development a week apart share one name.
- **Analysis or comment** whose main subject is one development gets that development's name.
- **A forecast and the actual result** are separate events, for example `tata_motors_cv_q2_preview` and `tata_motors_q2_actual_results`.
- **The same report with different comparison periods** is one event. For example, articles that use the same FADA Navratri figures share `navratri_sales_fada`, even when one compares them with 10 days of 2024 and another with 11 days of 2025.
- **An article that repeats known data to discuss what comes next** gets an event for what comes next, when that is its main subject. For example, an article on Diwali sales hopes that repeats the Navratri figures is `diwali_sales_outlook`.
- **A detail decided as part of a larger step** belongs to that step. For example, board appointments approved at the shareholders' meeting are `iveco_shareholders_approve`.
- **A round-up** of many companies (such as "Stocks to watch") gets the event name of its Tata Motors development. When it reports several separate Tata Motors developments, the main one gives the event name, and the `notes` column lists the others.
- **The same topic but different news** gets different names.
- **When unsure**, choose the best name and write the reason in `notes`.

Examples (Tata Motors):

| Articles | Event |
|---|---|
| "Tata Motors raises Iveco takeover offer to €14.40 per share" and "Tata Motors raises Iveco takeover offer price to 14.40 euro per share" | same event: `iveco_offer_raised` |
| "Iveco shareholders approve Tata Motors' takeover bid" | different stage: `iveco_shareholders_approve` |
| "Chandrasekaran, P.B. Balaji to join Iveco board…" (approved at the same meeting) | same event: `iveco_shareholders_approve` |
| Seven reports of the Tata Motors–Stellantis agreement, including a "Stocks to watch" round-up | same event: `tata_stellantis_partnership` |
| "Jaguar unveils $130,500 electric vehicle" and "The long road to the Type 01: Jaguar's EV timeline" | same event: `jaguar_type_01_unveil` (the timeline article is about the same launch) |
| "Tata Motors CV Q2 Results Preview…" and "Q2 Results Today Live…" | different events: a forecast and the actual result |

## Part 2: Judging summaries

### What is judged

Each row of the judging sheet is one item:

- `company point`: one point of the combined summary about the company
- `industry point`: one point of the industry context
- `article brief`: the short summary of one article

The `evidence` column shows exactly what the model had for that item: the cited articles' IDs, titles and the article text available to the summary model. **Judging uses only this evidence**, not the full article online and not outside knowledge. A true fact that is not in the evidence is still unsupported.

### Judgement (`judgement` column)

| Value | Rule |
|---|---|
| `supported` | Every factual claim in the item is stated in the evidence. Wording can differ; meaning cannot. |
| `partly_supported` | The main claim is in the evidence, but at least one detail is not (an extra number, a name, a cause, a date). The `note` names the unsupported detail. |
| `not_supported` | The main claim is not in the evidence, or the evidence says something different. |

### Contradicted (`contradicted` column)

`yes` when the evidence says the opposite of a claim, for example the item says profit **rose** and the evidence says it **fell**. A contradiction is more serious than missing evidence, so it is recorded on its own. An item with `contradicted = yes` is never `supported`.

### Numbers

Each number in an item is checked against the evidence for four things:

1. **Metric**: the number belongs to the same measure (revenue is not profit, registrations are not sales).
2. **Company**: the number belongs to the same company (a Maruti figure is not a Tata Motors figure).
3. **Period**: the number belongs to the same time (Q2 is not the full year; this Navratri is not last Navratri).
4. **Direction**: rise or fall matches the evidence.

A number that fails any of these makes the item `partly_supported` (when the number is a detail) or `not_supported` (when the number is the main claim). A wrong direction is also `contradicted = yes`. The `note` names the number and what is wrong with it.

Examples:

| Evidence | Item | Judgement | Contradicted |
|---|---|---|---|
| "Tata Motors raised its offer to 14.40 euro per share" | "Tata Motors raised its Iveco offer to €14.40 per share." | `supported` | |
| "Tata Motors raised its offer to 14.40 euro per share" | "Tata Motors raised its offer to €14.40 per share, its final offer, after shareholder pressure." | `partly_supported` (no evidence of shareholder pressure) | |
| "Car registrations fell 22 per cent this Navratri" | "Car sales rose 22% this Navratri." | `not_supported` | `yes` |
| "Revenue rose 12% while sales fell 5%" | "Revenue fell 12% and sales rose 5%." | `not_supported` | `yes` |
| "Tata Motors overtook Mahindra as No. 2 by domestic dispatches in April–September" | "Tata Motors became India's No. 2 car company in April–September." | `supported` | |

### Briefs that say there is not enough text

When the evidence is too short and the brief says so (for example "The excerpt gives no details"), the brief is `supported`: it states what the evidence shows. A brief that fills the gap with details not in the evidence is `partly_supported` or `not_supported`.

### Fairness rules

- The sheet does not show which run (k or search) produced an item, and the rows are shuffled.
- The key file (`summary_judging_key_<date>.json`) is not opened until every row is judged and saved.
- The same rule is used for similar items across the whole sheet.
- Judgements are not changed to improve a score. A genuine mistake is corrected using a rule from this guide. Every correction has a recorded reason, and both judgement versions remain available. If a correction follows inspection of model or summary results, that data is no longer treated as an untouched evaluation set.

### What the results report

Structural checks, number consistency, citation coverage, event coverage and human factual support are reported as separate measures. A summary that passes every automatic check is structurally valid, not necessarily correct; the judgement in this sheet measures the difference.

## Part 3: Comparing summary designs

### Files

- `summary_v2_judging.csv`: one row per item from every summary in the comparison, shuffled. Columns: `row`, `kind`, `text`, `evidence`, `judgement`, `contradicted`, `central`, `note`. The `evidence` column holds article IDs only. The reading page (`data/fulltext/summary_v2_judging.html`, outside Git) shows the text the model had for those articles.
- `summary_v2_events.csv`: one row per relevant event in each summary. Columns: `row`, `summary`, `event`, `articles`, `covered`, `note`.
- `summary_v2_judging_key.json` links each row to its design and run. It is kept in Git so the result can be repeated. Blinding is a procedure: the key can be opened, so it is not opened until both sheets are judged and locked.

### Items

Items are judged with the rules of Part 2, with these additions:

- Every row needs `judgement` and `contradicted` (`yes` or `no`). An empty field is never read as `no`.
- `judgement` and `contradicted` are separate. An item can hold supported facts and a contradicted claim at the same time; such an item is `partly_supported` or `not_supported`, never `supported`.
- `central` is filled only when `contradicted = yes`. It is `yes` when any contradicted claim in the item is a central fact: the main development, or a fact an analyst needs to understand it, such as an offer price or a vote result. It is `no` when every contradicted claim is a detail. A contradiction counts even when the rest of the item is correct.
- The `note` names the unsupported detail of a `partly_supported` item and the contradicted claim of a contradicted item.

### Event coverage

Each summary has a letter, given in random order. For each relevant event among its articles, `covered` is:

- `yes`: a company point or an industry point states the event's main development. A point counts even when it cites another article about the same event.
- `no`: no point states it. Naming the company, using a word from the event or citing one of its articles is not enough. A brief alone does not count, because every article has a brief.

The `articles` column lists the titles of the event's articles. The reading page shows each summary's points.

### Locking and scoring

When both sheets are complete, `python -m src.summary_judging lock` checks every row and saves fingerprints of the sheets, this guide, the summary record and the scoring code. `python -m src.summary_judging score` opens the key and refuses to run if a sheet changed after the lock. A genuine mistake found later goes into a separate adjudication record with its reason; the locked sheets stay as they are.

### Consistency check

A few days after the lock, `python -m src.summary_judging rejudge` makes a second sheet (`summary_v2_rejudge.csv`) with 20% of the items, in a new order with new row numbers. The sheet holds only the kind, the text, the evidence IDs and empty judgement columns. It never shows the first judgements or the design, and the command never opens the key, even after scoring. The sheet is judged with the same rules, without opening the first sheet, the key or `summary_v2_rejudge_key.json`. The share of matching judgements and a table of first against second judgements are reported. The first judgements are the ones used for the results.
