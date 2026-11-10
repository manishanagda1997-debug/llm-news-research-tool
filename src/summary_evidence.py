"""
The summary evidence: what the summary model reads for each relevant article
(Stage 4.1).

When the cleaned full text fits in MAX_CHARS, the model reads the whole
article. When it does not, sentences are chosen so that the model still sees
what happens and why it matters:

    whole sentences  a sentence is kept whole or left out, never cut, so a
                     negation, a condition or an attribution survives
    the lead         the opening paragraph comes first
    signals          status and timing, causes, impact and risks, quantities
                     and the company raise a sentence; a table-like paragraph
                     (many numbers and names, few words) ranks last
    qualifications   a kept sentence brings the next one when it starts with
                     "However", "But", "Although" and similar words; if both do
                     not fit, neither is kept
    order            kept sentences stay in article order; GAP marks left-out text

The signals only order the sentences. The choice between this evidence, the
relevance passage and the NewsAPI short text rests on a hand check of
important facts on a sample of long dev articles.

cut_sentences() counts the places where a passage cuts a sentence (the
relevance passage trims long paragraphs at spaces), for the comparison.
"""

import re

from src.query import matches, parse_query
from src.relevance_check import BUSINESS, CUT, GAP, MAX_PASSAGE_CHARS, _paragraphs

MAX_CHARS = MAX_PASSAGE_CHARS          # the same size as the relevance passage, for a fair comparison

ABBREVIATIONS = {"rs", "mr", "mrs", "ms", "dr", "no", "nos", "st", "vs", "inc", "ltd", "co", "corp", "jr", "sr",
                 "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "approx",
                 "est", "fig", "govt", "dept", "e.g", "i.e", "u.s", "u.k", "bn", "mn", "cr"}
_END = re.compile(r"[.!?][\"'”’)\]]*\s+")
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)$")

STATUS = re.compile(
    r"\b(?:announc\w*|approv\w*|reject\w*|complet\w*|sign\w*|agree\w*|launch\w*|unveil\w*|delay\w*|postpon\w*|"
    r"withdr[ae]w\w*|cancel\w*|den(?:y|ies|ied)|deny\w*|confirm\w*|expect\w*|plan\w*|propos\w*|pending|"
    r"subject to|raise[sd]?|cut[s]?|accept\w*|acquir\w*|tender\w*|clos(?:e|es|ed|ing)|will|first|final)\b", re.I)
CAUSE = re.compile(r"\b(?:because|due to|as a result|after|amid|following|driven by|owing to|since)\b", re.I)
IMPACT = re.compile(r"\b(?:risk\w*|impact\w*|concern\w*|pressure|margin\w*|demand|could|may|might|threat\w*|"
                    r"hit|boost\w*|weigh\w*|uncertain\w*|warn\w*|outlook|guidance)\b", re.I)
QUALIFIER = re.compile(r"^(?:However|But|Although|Though|Still|Yet|That said|Even so|Meanwhile|Nevertheless|"
                       r"Instead|Despite)\b")
NUMBER = re.compile(r"\d[\d,.]*")
WEIGHTS = {"company": 3, "status": 3, "quantity": 2, "cause": 2, "impact": 2}
LEAD_SCORE = 100
TABLE_SCORE = -10


def split_sentences(text: str) -> list[str]:
    """Sentences, ended by . ! or ? before a capital letter, a digit or a quote; abbreviations do not end one."""
    sentences, start = [], 0
    for match in _END.finditer(text):
        after = text[match.end():match.end() + 1]
        if not after or not (after.isupper() or after.isdigit() or after in "\"'“‘("):
            continue
        if text[match.start()] == ".":
            word = _WORD_BEFORE.search(text[start:match.start()])
            if word and (word.group(1).lower() in ABBREVIATIONS or len(word.group(1)) == 1):
                continue
        sentences.append(text[start:match.end()].strip())
        start = match.end()
    rest = text[start:].strip()
    if rest:
        sentences.append(rest)
    return sentences


def _table_like(paragraph: str) -> bool:
    words = paragraph.split()
    numbers = NUMBER.findall(paragraph)
    return len(numbers) >= 8 and len(numbers) / max(len(words), 1) > 0.25


def _score(sentence: str, rule) -> int:
    signals = {"company": rule is not None and matches(sentence, rule), "status": bool(STATUS.search(sentence)),
               "quantity": bool(BUSINESS.search(sentence)), "cause": bool(CAUSE.search(sentence)),
               "impact": bool(IMPACT.search(sentence))}
    return sum(WEIGHTS[name] for name, found in signals.items() if found)


def _render(title: str, units: list[tuple[int, int, str]], chosen: set[int]) -> str:
    """Kept sentences in article order: same paragraph and next to each other -> one line; a gap -> GAP."""
    lines, previous = [title], None
    for index in sorted(chosen):
        paragraph, _, sentence = units[index]
        if previous is not None and index == previous + 1 and units[previous][0] == paragraph:
            lines[-1] += " " + sentence
        else:
            if previous is not None and index != previous + 1:
                lines.append(GAP)
            lines.append(sentence)
        previous = index
    return "\n".join(lines)


def build_summary_evidence(title: str, text: str, company: str, aliases: dict | None,
                           max_chars: int = MAX_CHARS, short_text: str = "") -> dict:
    paragraphs = _paragraphs(text) or _paragraphs(short_text)
    title = (title or "").strip()[:max_chars]
    units = [(p, s, sentence) for p, paragraph in enumerate(paragraphs)
             for s, sentence in enumerate(split_sentences(paragraph))]
    whole = "\n".join([title, *paragraphs])
    if len(whole) <= max_chars:
        return {"text": whole, "complete": True, "sentences_kept": len(units), "sentences_total": len(units)}

    rule = parse_query(company, aliases=aliases)
    tables = {p for p, paragraph in enumerate(paragraphs) if _table_like(paragraph)}
    scores = []
    for index, (p, _, sentence) in enumerate(units):
        if p in tables:
            score = TABLE_SCORE
        elif p == 0:
            score = LEAD_SCORE
        else:
            score = _score(sentence, rule)
        scores.append(score)
    order = sorted(range(len(units)), key=lambda i: (-scores[i], i))

    chosen: set[int] = set()
    for index in order:
        if index in chosen or scores[index] < 0:
            continue
        group = {index}
        following = index + 1
        if following < len(units) and units[following][0] == units[index][0] \
                and QUALIFIER.match(units[following][2]):
            group.add(following)                   # the qualification travels with its sentence
        if len(_render(title, units, chosen | group)) <= max_chars:
            chosen |= group
    return {"text": _render(title, units, chosen), "complete": False,
            "sentences_kept": len(chosen), "sentences_total": len(units)}


def cut_sentences(passage: str, full_text: str) -> int:
    """Pieces of a passage that start or end inside a sentence (marked with CUT by the relevance passage)."""
    count = 0
    for line in passage.split("\n")[1:]:
        if line == GAP:
            continue
        count += int(line.startswith(CUT) or line.endswith(CUT))
    return count
