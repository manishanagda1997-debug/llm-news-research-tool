# Summary judging guide, Part 5: prompt version 5

Prompt version 5 keeps the industry-context rule of prompt version 4 and adds one instruction: "When an article reports a relevant fact about the company's sector or market, or about another company operating in that same sector, include that fact as an "industry_context" point and cite the article. Examples include a rival's sales, a sector-wide figure, or a supplier's results. Do not include companies from other sectors merely because an article lists them next to the company, for example in a "stocks in news" list or a results roundup."

The summaries made with prompt version 5 are judged and scored with Part 4 (`SUMMARY_JUDGING_GUIDE_PART4.md`), unchanged, with these differences:

- Files: `summary_v2_p5_judging.csv`, `summary_v2_p5_events.csv` and the reading page `data/fulltext/summary_v2_p5_judging.html`, which shows the items and the event coverage.
- Industry evidence is not judged again. The selected articles are the same as in Part 4, so each run keeps its locked Part 4 decision: 14 October no, 23 October yes, 24 October yes.
- An item keeps a locked judgement only when its section, exact text and cited articles match a direct item in the locked Part 3 or Part 4 sheet. When the locked sheets contain conflicting judgements for an item, either within one sheet or between the two sheets, it is judged again. Every other item is judged again.
- The threshold for key points and briefs is (29 of 38) minus 0.05, calculated and compared without rounding.
