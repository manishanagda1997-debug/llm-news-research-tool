"""
Tests for the `notes` column of the event sheets (no network, no model calls).

`notes` is the last column of every event sheet. It gives the reason for a
grouping decision that is not obvious: a decision made from the full text, a
round-up, or a main subject that differs from the headline. It never changes
an event name, and it never replaces the separate record of label
corrections.

Readers work the same with or without the column, so older sheets still load
and every result built from the event names stays the same.
"""

import csv
from pathlib import Path

import pytest

from src import event_check as ec
from src import summary_run as sr

ROOT = Path(__file__).resolve().parent.parent
DEV_SHEET = ROOT / "data" / "eval" / "events_dev_tata_motors.csv"
OLD_COLUMNS = ["row", "event", "title", "source", "published", "description", "content_first_200_chars", "url"]

DEV_EVENTS = {
    1: "iveco_shareholder_acceptance", 2: "goyal_india_car_hub", 3: "goyal_india_car_hub",
    4: "iveco_offer_raised", 5: "tata_motors_cv_q2_preview", 6: "stellantis_india_development_target",
    7: "iveco_shareholder_acceptance", 8: "iveco_shareholder_acceptance", 9: "navratri_sales_fada",
    10: "jaguar_type_01_unveil", 11: "global_used_truck_market_report", 12: "jaguar_type_01_unveil",
    13: "navratri_sales_fada", 14: "jaguar_type_01_unveil", 15: "tata_stellantis_partnership",
    16: "jaguar_type_01_unveil", 17: "global_carmakers_india_investment", 18: "tata_stellantis_partnership",
    19: "iveco_offer_raised", 20: "iveco_investor_gains", 21: "diwali_sales_outlook",
    22: "goyal_india_car_hub", 23: "global_used_truck_market_report", 24: "uk_india_trade_deals",
    25: "iveco_shareholders_approve", 26: "auto_stocks_rbi_impact", 27: "tata_motors_market_ranking",
    28: "delhi_old_truck_replacement_curb", 29: "china_world_car_strategy", 30: "tata_stellantis_partnership",
    31: "jlr_delhi_gurugram_expansion", 32: "iveco_offer_raised", 33: "stellantis_india_development_target",
    34: "gabriel_forvia_jv", 35: "tata_stellantis_partnership", 36: "range_rover_electric_test",
    37: "iveco_shareholders_approve", 38: "china_european_car_cost_pressure", 39: "tata_stellantis_partnership",
    40: "jlr_rebrand_narrative", 41: "tata_stellantis_partnership", 42: "iveco_shareholders_approve",
    43: "navratri_sales_fada", 44: "multi_powertrain_model_trend", 45: "tata_stellantis_partnership",
    46: "jaguar_type_01_unveil", 47: "goyal_india_car_hub", 48: "tata_motors_q2_actual_results",
}

FADA = "Full text: the articles use the same FADA Navratri figures but compare them with different periods."
DEV_NOTES = {
    1: "Full text: the main news is the ongoing acceptance process. The article reports 31.19% of shares "
       "tendered against the 80% needed, not a new vote.",
    9: FADA, 13: FADA, 43: FADA,
    15: "Round-up; its Tata Motors item is the Stellantis agreement.",
    20: "The main subject is the gain for three investors, not the acceptance progress.",
    21: "Full text: repeats the FADA Navratri figures; the main subject is the hope for Diwali sales.",
    37: "Full text: the board appointments are part of the resolutions approved at the shareholders' meeting.",
}


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------- the column ----------

def test_notes_is_the_last_column_and_the_other_columns_stay_the_same():
    assert sr.EVENT_COLUMNS == OLD_COLUMNS + ["notes"]


def _raw(n):
    return {"source": {"id": None, "name": f"Source {n}"}, "author": "A", "title": f"Tata Motors item {n}",
            "description": f"Description {n}.", "content": "x" * 300, "url": f"https://x.com/{n}",
            "urlToImage": None, "publishedAt": f"2026-10-{10 + n:02d}T10:00:00Z"}


def test_a_new_sheet_has_an_empty_notes_column():
    rows = sr.relevant_event_rows([_raw(n) for n in range(1, 4)], {f"https://x.com/{n}": "direct" for n in range(1, 4)})
    assert all(list(row) == sr.EVENT_COLUMNS and row["notes"] == "" for row in rows)


# ---------- readers work with or without the column ----------

def test_load_events_gives_the_same_result_with_or_without_notes(tmp_path):
    rows = _read(DEV_SHEET)
    with_notes, without = tmp_path / "with.csv", tmp_path / "without.csv"
    _write(with_notes, rows, sr.EVENT_COLUMNS)
    _write(without, rows, OLD_COLUMNS)
    assert sr.load_events(with_notes) == sr.load_events(without)


def test_a_note_with_commas_quotes_and_line_breaks_does_not_disturb_the_events(tmp_path):
    path = tmp_path / "events.csv"
    rows = [{"row": "1", "event": "iveco_offer_raised", "url": "https://x.com/1",
             "notes": 'Full text: "best and final", per Reuters,\nsecond line'}]
    _write(path, rows, sr.EVENT_COLUMNS)
    assert sr.load_events(path) == {"https://x.com/1": "iveco_offer_raised"}


def test_the_event_check_gives_the_same_findings_with_or_without_notes():
    rows = _read(DEV_SHEET)
    without = [{k: v for k, v in row.items() if k != "notes"} for row in rows]
    assert ec.check_events(rows) == ec.check_events(without)


# ---------- the development sheet ----------

PUBLISHER_COLUMNS = ("description", "content_first_200_chars")      # article text from NewsAPI


def test_the_dev_sheet_has_every_column_in_order():
    """Every column in order. A copy without the saved NewsAPI responses also leaves out the two columns of article
    text from NewsAPI (the public copy); every other column is still required, in the same order."""
    from tests.raw_data import raw_data_present
    expected = sr.EVENT_COLUMNS if raw_data_present() else [c for c in sr.EVENT_COLUMNS if c not in PUBLISHER_COLUMNS]
    with open(DEV_SHEET, encoding="utf-8-sig", newline="") as handle:
        assert next(csv.reader(handle)) == expected


def test_the_dev_sheet_keeps_its_rows_and_event_names():
    rows = _read(DEV_SHEET)
    assert [int(r["row"]) for r in rows] == list(range(1, 49))
    assert {int(r["row"]): r["event"] for r in rows} == DEV_EVENTS
    assert len(set(DEV_EVENTS.values())) == 25


def test_the_dev_sheet_has_notes_only_where_a_decision_needs_a_reason():
    rows = _read(DEV_SHEET)
    assert {int(r["row"]): r["notes"] for r in rows if r["notes"]} == DEV_NOTES


@pytest.mark.parametrize("note", sorted(set(DEV_NOTES.values())))
def test_a_note_explains_and_never_names_an_event_of_its_own(note):
    assert not any(name in note for name in set(DEV_EVENTS.values()))
