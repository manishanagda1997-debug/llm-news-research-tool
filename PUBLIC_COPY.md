# Public copy

This repository is the public copy of the News Research Tool project. It holds the code, the tests, the notebooks with their saved outputs, the labelling and judging guides, and every evaluation plan, record, label, judgement, lock and score.

Article content returned by NewsAPI and text from publishers' pages is not included. NewsAPI's terms do not allow publishing the data it returns, and article text belongs to its publishers. The evaluation results in this repository are the project's own measurements and judgements; they are not NewsAPI data or statistics about the NewsAPI service.

## Not included

- The 16 saved NewsAPI responses in `data/raw/`.
- Two columns of article text from NewsAPI, `description` and `content_first_200_chars`, in 11 sheets:
  `data/eval/dev_tata_motors_20261014.csv`, `data/eval/dev_tata_motors_20261023.csv`,
  `data/eval/dev_tata_motors_fresh_20261024.csv`, `data/eval/dev_tata_motors_retrieval_20261023.csv`,
  `data/eval/events_dev_tata_motors.csv`, `data/eval/recheck_v2_dev_tata_motors.csv`, `data/eval/v2_dev_tata_motors.csv`,
  and the `events.csv` sheets of `fresh_2026-10-24_30`, `transfer_reliance_2026-10-26_11-01`,
  `week2_tata_2026-10-31_11-06` and `week2_reliance_2026-11-02_11-08`.
- One column of sentences quoted from articles, `added_text`, in `data/eval/fact_check_passage2_added.csv`.
- Publisher page text (`data/fulltext/`), the app's saved summaries and the `.env` file, which are never part of the repository.

Article titles and links stay in the sheets: they identify each article that is labelled and judged.

## Checks

- Every other value of the 12 shortened sheets is identical to the original sheet, and every other file is identical to the original file.
- Publisher text: every 60-character run of the NewsAPI descriptions and excerpts and of the publisher page text is searched for in every file. The remaining matches are short phrases inside the summaries the model writes, which are the results under evaluation, in test examples, and in four short phrases of the project's own notes (the fact lists and one notebook sentence). The longest is 214 characters, one sentence.
- No API key and no `.env` file is in any file.
- Tests: 1,738 pass. 32 are skipped because they read the saved NewsAPI responses: 29 run the Round 3 pair command, which rebuilds each day's five articles from them; 2 check the listing filter against labelled articles; 1 checks each company's earlier data.
- The freeze check passes: `data/eval/summary_freeze.json` matches the tag `freeze-summary`, and the frozen summary design is accepted.

## History and running the notebooks

This copy is one snapshot of the final project. The ten freeze tags point to it. The date of each plan, seal and freeze is in its record (`created`, `sealed_at`, `source_commit`); the commits those records name are in the full development history, 231 commits with the original tags, which is kept privately.

The notebooks show their saved outputs. Running them again needs the saved NewsAPI responses and publisher page texts, which are not included; the app and the tests run with your own free NewsAPI and Groq keys (see `.env.example` and `check_setup.py`).
