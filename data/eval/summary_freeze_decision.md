# Stage 4 summary freeze: decision

Design: direct. Articles per summary (k): 5. Prompt version 4. Configuration 7 (targeted correction, number check 2, summary passage 3). Pacer version 3.

**Design.** Direct and brief-first are compared on the three development runs with blind judging. Both meet every acceptance rule. Direct has the higher supported share (31 of 45, 68.9%, against 18 of 30, 60.0%), full event coverage and fewer calls per summary, and it is higher on two of the three runs.

**Industry context.** In that comparison, all four unsupported direct points are industry points that present a Tata Motors fact as a sector trend. Prompt version 4 adds a rule against this; prompt version 5 also tells the model to include facts about the sector and about other companies in it. Neither passes every check of the industry-context test (Parts 4 and 5 of the judging guide). Prompt 4 leaves out one relevant fact about another automotive company (one case in the three runs). Prompt 5 adds industry points in a run whose articles hold no wider-industry evidence. Neither version has an unsupported industry point, key point or brief.

**Selection.** Prompt 4 is selected because it has fewer industry points that are not fully supported (0 of 1, against 3 of 5 for prompt 5). This rule is stated after the prompt 4 and prompt 5 results, not in advance. The selection does not show that prompt 4 is better on unseen news; the fresh-week test checks that.

**Articles per summary.** k = 10 does not qualify under the project's free-plan headroom rule. The condition is at least 10% headroom under the limit of 8,000 tokens a minute, so at most 7,200 tokens per request. On 24 October, Groq reports 6,900 input tokens for the k = 10 request. Earlier Groq refusal messages show 276–551 requested tokens above the input count. Using the largest observed difference gives a conservative estimate of 7,451 tokens, above 7,200; using the smallest gives 7,176. The 23 October request has a longer input and is not sent because the project's own estimate exceeds the limit; Groq does not refuse it. The quality of k = 10 summaries is not judged.

**Judging consistency.** A second judgement of 15 items by the same judge matches the first in 13 (86.7%; approximate 95% interval 62% to 96%), and all 15 contradiction decisions match.
