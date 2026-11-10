"""
Press-release boilerplate, removed from the summary input (summary passage version 3, src/summary_passage.py).
The relevance passage (Stages 2 and 3) does not use this module.

The rules work on the structure of the text, never on single words:

    leading notice  in the first two paragraphs only: a paragraph in capital letters with a distribution notice
                    ("NOT FOR RELEASE", "NOT FOR DISTRIBUTION", "NOT FOR PUBLICATION", "FOR IMMEDIATE RELEASE").
                    When the capital-letter notice opens a paragraph that then goes on in normal case, only the
                    notice is cut.
    boilerplate tail  from a boilerplate heading to the end of the text, only when everything after the heading is
                    boilerplate too:
                    - another boilerplate heading;
                    - a contact line: an email address, a phone number, or a short line that is not a sentence
                      (a name, a role, an agency);
                    - text under a legal or footer heading;
                    - at most ABOUT_PARAGRAPHS paragraphs under an "About <Name>" heading.
                    Anything else, for example a news paragraph under a contacts heading, stops the rule there.
                    At least one paragraph of the text always stays.

A heading is a line of at most HEADING_CHARS characters that does not end a sentence and is not a list item.
"""

import re

HEADING_CHARS = 90
CONTACT_LINE_CHARS = 60
ABOUT_PARAGRAPHS = 3
CAPITALS_SHARE = 0.8

NOTICE = re.compile(r"\bNOT FOR (?:RELEASE|DISTRIBUTION|PUBLICATION)\b|\bFOR IMMEDIATE RELEASE\b")
HEADINGS = {
    "contacts": re.compile(r"^(?:.{0,60}\s)?(?:contacts?|(?:media|press|investor|news)\s+"
                           r"(?:relations|enquiries|inquiries|contacts?)(?:\s+contacts?)?)$", re.I),
    "about": re.compile(r"^About\s+(?!the\b|this\b|these\b|our\b|a\b|an\b)[A-Z0-9].*$"),
    "legal": re.compile(r"^(?:general restrictions|important (?:notice|information)|disclaimers?|legal notice|"
                        r"forward[- ]looking (?:statements?|information)|cautionary (?:statements?|notes?).*|"
                        r"safe harbou?r(?: statements?)?|additional information)$", re.I),
    "footer": re.compile(r"^(?:comments?|attachments?|related (?:articles|stories|news))$", re.I),
}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\b(?:tel|phone|mobile|fax)\b|\+\s?\d[\d\s().-]{6,}\d", re.I)


def _paragraphs(text: str) -> list[str]:
    return [line.strip() for line in (text or "").split("\n") if line.strip()]


def heading_kind(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > HEADING_CHARS or line[0] in "-•*" or line.endswith((".", "!", "?")):
        return None
    line = line.rstrip(":").strip()
    for kind, pattern in HEADINGS.items():
        if pattern.match(line):
            return kind
    return None


def _contact_line(line: str) -> bool:
    if EMAIL.search(line) or PHONE.search(line):
        return len(line) <= 2 * CONTACT_LINE_CHARS
    return len(line) <= CONTACT_LINE_CHARS and not line.endswith((".", "!", "?"))


def _capitals_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def _notice_prefix(paragraph: str) -> str | None:
    """The capital-letter notice at the start of a paragraph, or None."""
    words, prefix = paragraph.split(" "), []
    for word in words:
        if any(c.isalpha() for c in word) and _capitals_share(word) < 1.0:
            break
        prefix.append(word)
    text = " ".join(prefix).strip()
    if len(text) >= 20 and NOTICE.search(text) and _capitals_share(text) >= CAPITALS_SHARE:
        return text
    return None


def _tail_fits(paragraphs: list[str], start: int) -> bool:
    section, under_about = heading_kind(paragraphs[start]), 0
    for line in paragraphs[start + 1:]:
        kind = heading_kind(line)
        if kind:
            section, under_about = kind, 0
            continue
        if section == "about":
            under_about += 1
            if under_about > ABOUT_PARAGRAPHS:
                return False
            continue
        if section in ("legal", "footer") or _contact_line(line):
            continue
        return False
    return True


def clean(text: str) -> dict:
    """{"text", "removed": [{"kind", "first", "last", "chars", "heading"}], "removed_chars"}; paragraphs are
    counted from 0 in the original text."""
    paragraphs = _paragraphs(text)
    kept = list(enumerate(paragraphs))
    removed = []
    for position, (index, paragraph) in enumerate(kept[:2]):
        notice = _notice_prefix(paragraph)
        if notice is None:
            continue
        rest = paragraph[len(notice):].strip()
        removed.append({"kind": "leading_notice", "first": index, "last": index, "heading": "",
                        "chars": len(paragraph) - len(rest)})
        kept[position] = (index, rest)
        break
    kept = [(index, p) for index, p in kept if p]
    for position, (index, paragraph) in enumerate(kept):
        if position == 0 or not heading_kind(paragraph):
            continue
        if not any(heading_kind(p) is None for _, p in kept[:position]):
            continue
        lines = [p for _, p in kept[position:]]
        if _tail_fits(lines, 0):
            removed.append({"kind": "tail", "first": index, "last": kept[-1][0], "heading": paragraph,
                            "chars": sum(len(p) for p in lines)})
            kept = kept[:position]
            break
    return {"text": "\n".join(p for _, p in kept), "removed": removed,
            "removed_chars": sum(r["chars"] for r in removed)}
