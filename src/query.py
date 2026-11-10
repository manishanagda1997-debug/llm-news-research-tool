"""
Search rules shared by the news fetcher and the cleaning step.

One place decides what a query means, so that the query sent to NewsAPI
and the check on the returned articles always agree.

Supported (operators in capitals only):
    Infosys                    one word
    Tata Motors                plain words: exact phrase, or all words
                               in any order (exact_phrase=False)
    "Tata Motors"              exact phrase
    Microsoft OR Google        any of the terms
    AI AND healthcare          all of the terms
    +Infosys results           + means "must appear" (same as a plain term)
    Tata Motors -Jaguar        - or NOT means "must not appear"
    Tata Motors NOT Jaguar

Not supported by the local check (it is skipped, never guessed):
    brackets, AND mixed with OR, NOT/- mixed with OR, an operator with
    nothing after it, or a query with only excluded words.

The check only looks at the text NewsAPI returns (title, description and
the first 200 characters of content), not the full article.

Optional company aliases (off unless an alias file is passed in):
    A search for one company name, such as "Tata Motors", can also accept
    the verified names listed for it in data/company_aliases.json
    (for example TMPV and TMCV). Excluded words still win: with
    "Tata Motors" -TMPV, an article that only mentions TMPV is not kept.
    Aliases are added only when the query searches for one name or one
    phrase; queries with several terms, OR or brackets are left as they are.
    Some names are ambiguous (Jaguar: the car brand, an animal, a fighter
    jet); they count only when the same text also has a context word
    listed for them in the alias file (car, carmaker, EV ...).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

OPERATORS = {"AND", "OR", "NOT"}
BRACKETS = {"(", ")"}

# A token is a "quoted phrase", a bracket, or a run of other characters
TOKEN = re.compile(r'"[^"]*"|[()]|[^\s()"]+')


class QueryError(ValueError):
    """The query cannot be used at all (empty, or a quote mark is not closed)."""


@dataclass
class QueryRule:
    include: list[str] = field(default_factory=list)  # terms that must (or may, for OR) appear
    exclude: list[str] = field(default_factory=list)  # terms that must not appear
    mode: str = "all"                                 # "all" = every include term, "any" = at least one
    supported: bool = True                            # False = local check is skipped
    reason: str = ""                                  # why it is skipped, shown in the app
    aliases_used: bool = False                        # True = the rule includes verified alias names
    alias_group: str = ""                             # the company group the aliases came from
    # names that count only next to a context word, e.g. ("jaguar", ["car", "ev", ...])
    include_with_context: list[tuple[str, list[str]]] = field(default_factory=list)


def _tokens(query: str) -> list[str]:
    query = (query or "").strip()
    if not query:
        raise QueryError("Please type a search query.")
    if query.count('"') % 2 != 0:
        raise QueryError("A quote mark (\") is not closed. Add the missing quote and try again.")
    return TOKEN.findall(query)


def _is_plain(tokens: list[str]) -> bool:
    """Plain words only: no quotes, no operators, no brackets, no +word or -word."""
    for token in tokens:
        if token.startswith('"') or token in OPERATORS or token in BRACKETS:
            return False
        if len(token) > 1 and token[0] in "+-":
            return False
    return True


def build_search_query(query: str, exact_phrase: bool = True) -> str:
    """
    The text sent to NewsAPI.

    Plain words follow the "Exact phrase" setting:
        exact_phrase=True   Tata Motors  ->  "Tata Motors"
        exact_phrase=False  Tata Motors  ->  Tata AND Motors
    Anything with quotes, operators, + or - is sent exactly as typed.
    """
    tokens = _tokens(query)
    if not _is_plain(tokens) or len(tokens) == 1:
        return query.strip()
    if exact_phrase:
        return '"' + " ".join(tokens) + '"'
    return " AND ".join(tokens)


MAX_SEARCH_LENGTH = 500   # NewsAPI limit for the q parameter


def _quote(term: str) -> str:
    return f'"{term}"' if " " in term else term


def _display_terms(tokens: list[str]) -> tuple[list[str], list[str]]:
    """The user's own include and exclude terms, with their original spelling."""
    include, exclude, next_is_excluded = [], [], False
    for token in tokens:
        if token in ("AND", "OR"):
            continue
        if token == "NOT":
            next_is_excluded = True
            continue
        excluded, term = next_is_excluded, token
        next_is_excluded = False
        if len(term) > 1 and term[0] == "-":
            excluded, term = True, term[1:]
        elif len(term) > 1 and term[0] == "+":
            term = term[1:]
        term = " ".join(term.strip('"').split())
        (exclude if excluded else include).append(term)
    return include, exclude


SEARCH_PARTS = ("all", "own", "subsidiaries")


def build_search_plan(query: str, exact_phrase: bool = True, aliases: dict | None = None,
                      part: str = "all") -> dict:
    """
    The search sent to NewsAPI, plus a record of how it is built.

    part - which alias names to search for:
           "all"           the name, its own names and its subsidiaries' names
           "own"           the name and its own names (TMPV, TMCV ...)
           "subsidiaries"  only the subsidiaries' names (Jaguar Land Rover ...);
                           empty when the company has none, so no request is made

    Without aliases, or when the query is not "one name or phrase with
    optional excluded words", the search is the same as build_search_query().
    With aliases, the name becomes a group and excluded words apply to the
    whole group:  ("Tata Motors" OR TMPV OR TMCV) NOT Jaguar
    """
    if part not in SEARCH_PARTS:
        raise QueryError(f"part must be one of {SEARCH_PARTS}.")
    plan = {
        "original_query": query.strip(),
        "search_query": build_search_query(query, exact_phrase=exact_phrase),
        "aliases_added": False,
        "note": "",
    }
    if part == "subsidiaries":
        plan["search_query"] = ""
        plan["note"] = "No subsidiary names are known for this search."
    if not aliases:
        return plan

    rule = _parse(query, exact_phrase)
    if not rule.supported or rule.mode != "all" or len(rule.include) != 1:
        plan["note"] = "Aliases not added: they are only added to a search for one name or phrase."
        return plan

    names = [name for name in _alias_display_names(rule.include[0], aliases, part)
             if _normal(name) not in rule.exclude]
    with_context = [(name, words) for name, words in context_names(rule.include[0], aliases, part)
                    if _normal(name) not in rule.exclude]
    if not names and not with_context:
        return plan

    tokens = _tokens(query)
    if _is_plain(tokens):
        main, excluded = " ".join(tokens), []
    else:
        include_terms, excluded = _display_terms(tokens)
        main = include_terms[0]

    parts = [_quote(term) for term in ([] if part == "subsidiaries" else [main]) + names]
    for name, words in with_context:
        parts.append(f"({_quote(name)} AND ({' OR '.join(_quote(word) for word in words)}))")
    expanded = "(" + " OR ".join(parts) + ")"
    expanded += "".join(f" NOT {_quote(term)}" for term in excluded)
    if len(expanded) > MAX_SEARCH_LENGTH:
        plan["note"] = (f"Aliases not added: the search would be longer than NewsAPI's "
                        f"{MAX_SEARCH_LENGTH}-character limit.")
        return plan

    plan.update(search_query=expanded, aliases_added=True, note="")
    return plan


def _unsupported(reason: str) -> QueryRule:
    return QueryRule(supported=False, reason=f"Relevance check skipped: {reason}.")


def load_aliases(path) -> dict:
    """Read the alias file (see data/company_aliases.json)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normal(text: str) -> str:
    return " ".join(text.lower().split())


def _in_part(identity: dict, part: str) -> bool:
    """A subsidiary is an identity with a parent in the alias file."""
    if part == "own":
        return "parent" not in identity
    if part == "subsidiaries":
        return "parent" in identity
    return True


def _alias_display_names(company: str, aliases: dict | None, part: str = "all") -> list[str]:
    """Alias names with their original spelling, for the search sent to NewsAPI."""
    if not aliases:
        return []
    key = _normal(company)
    group = aliases.get("search_groups", {}).get(key)
    if not group:
        return []
    names: list[str] = []
    for identity in group["identities"]:
        if not _in_part(aliases["identities"][identity], part):
            continue
        for name in aliases["identities"][identity]["names"]:
            if _normal(name) != key and name not in names:
                names.append(name)
    return names


def _group_identities(company: str, aliases: dict | None) -> list[tuple[str, dict]]:
    if not aliases:
        return []
    group = aliases.get("search_groups", {}).get(_normal(company))
    if not group:
        return []
    return [(identity, aliases["identities"][identity]) for identity in group["identities"]]


def context_names(company: str, aliases: dict | None, part: str = "all") -> list[tuple[str, list[str]]]:
    """Names that only count next to a context word, with their original spelling."""
    result = []
    for _, identity in _group_identities(company, aliases):
        if not _in_part(identity, part):
            continue
        for entry in identity.get("names_needing_context", []):
            result.append((entry["name"], list(entry["context_words"])))
    return result


def matched_identities(text: str, company: str, aliases: dict | None) -> list[str]:
    """Which companies of the group the text names through their own names (not the group name)."""
    text = text.lower()
    found = []
    for identity_id, identity in _group_identities(company, aliases):
        plain = any(_contains(text, _normal(name)) for name in identity["names"] if _normal(name) != _normal(company))
        with_context = any(
            _contains(text, _normal(entry["name"])) and any(_contains(text, _normal(w)) for w in entry["context_words"])
            for entry in identity.get("names_needing_context", [])
        )
        if plain or with_context:
            found.append(identity_id)
    return found


def alias_names(company: str, aliases: dict | None) -> list[str]:
    """Verified extra names for a company search group, in lower case. Empty if none."""
    if not aliases:
        return []
    key = _normal(company)
    group = aliases.get("search_groups", {}).get(key)
    if not group:
        return []
    names: list[str] = []
    for identity in group["identities"]:
        for name in aliases["identities"][identity]["names"]:
            name = _normal(name)
            if name != key and name not in names:
                names.append(name)
    return names


def _add_aliases(rule: QueryRule, aliases: dict | None) -> QueryRule:
    """Add alias names when the rule searches for exactly one name or phrase."""
    if not aliases or not rule.supported or rule.mode != "all" or len(rule.include) != 1:
        return rule
    extra = [name for name in alias_names(rule.include[0], aliases) if name not in rule.exclude]
    with_context = [
        (_normal(name), [_normal(word) for word in words])
        for name, words in context_names(rule.include[0], aliases)
        if _normal(name) not in rule.exclude
    ]
    if not extra and not with_context:
        return rule
    return QueryRule(include=rule.include + extra, exclude=rule.exclude, mode="any", aliases_used=True,
                     alias_group=rule.include[0], include_with_context=with_context)


def parse_query(query: str, exact_phrase: bool = True, aliases: dict | None = None) -> QueryRule:
    """
    Turn the query into a QueryRule that matches() can check against article text.

    aliases - the loaded alias file, or None to use the query words only
    """
    return _add_aliases(_parse(query, exact_phrase), aliases)


def _parse(query: str, exact_phrase: bool) -> QueryRule:
    tokens = _tokens(query)

    # Plain words: one phrase, or every word on its own
    if _is_plain(tokens):
        words = [token.lower() for token in tokens]
        if exact_phrase and len(words) > 1:
            return QueryRule(include=[" ".join(words)])
        return QueryRule(include=words)

    if any(token in BRACKETS for token in tokens):
        return _unsupported("the query uses brackets")

    include: list[str] = []
    exclude: list[str] = []
    used_operators: set[str] = set()
    next_is_excluded = False
    last_was_operator = False

    for token in tokens:
        if token in ("AND", "OR"):
            if last_was_operator or not (include or exclude):
                return _unsupported(f"{token} is in the wrong place")
            used_operators.add(token)
            last_was_operator = True
            continue
        if token == "NOT":
            next_is_excluded = True
            used_operators.add("NOT")
            last_was_operator = True
            continue

        excluded = next_is_excluded
        next_is_excluded = False
        last_was_operator = False

        term = token
        if len(term) > 1 and term[0] == "-":
            excluded, term = True, term[1:]
        elif len(term) > 1 and term[0] == "+":
            term = term[1:]
        term = " ".join(term.strip('"').lower().split())
        if not term:
            return _unsupported("the query has an empty phrase")

        (exclude if excluded else include).append(term)

    if last_was_operator:
        return _unsupported("the query ends with an operator")
    if "AND" in used_operators and "OR" in used_operators:
        return _unsupported("the query mixes AND with OR")
    if "OR" in used_operators and exclude:
        return _unsupported("the query mixes NOT or - with OR")
    if not include:
        return _unsupported("the query has no word to search for, only excluded words")

    return QueryRule(include=include, exclude=exclude, mode="any" if "OR" in used_operators else "all")


def _contains(text: str, term: str) -> bool:
    """
    Whole-word check. The term must not have a letter or digit right
    before or after it, so "ai" matches "AI-powered" but not "said",
    and "c++" works even though "+" is not a letter.
    """
    pattern = re.escape(term).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None


def matches(text: str, rule: QueryRule) -> bool:
    """True when the text follows the rule. Only call this for supported rules."""
    if not rule.supported:
        raise ValueError("This query is not supported by the local check.")
    text = text.lower()
    found = [_contains(text, term) for term in rule.include]
    found += [
        _contains(text, name) and any(_contains(text, word) for word in words)
        for name, words in rule.include_with_context
    ]
    wanted = any(found) if rule.mode == "any" else all(found)
    blocked = any(_contains(text, term) for term in rule.exclude)
    return wanted and not blocked
