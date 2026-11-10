"""Tests: every score uses one clear label version, and saved records work on any computer."""

import csv

import pytest

from src import evaluate, retrieval_check as rc


def _labels(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "label", "url"])
        writer.writeheader()
        for number, (url, label) in enumerate(rows, start=1):
            writer.writerow({"row": number, "label": label, "url": url})
    return path


# ---------- 1. the label versions are named ----------

def test_version_1_and_version_2_label_files_are_named_separately():
    assert evaluate.DEV_LABELS_V1 == ["dev_tata_motors_20261014.csv", "dev_tata_motors_20261023.csv"]
    assert evaluate.DEV_LABELS_V2 == ["v2_dev_tata_motors.csv"]
    assert evaluate.LABEL_VERSION == "2"


# ---------- 2. conflicting labels stop the run ----------

def test_label_set_accepts_the_same_label_twice(tmp_path):
    first = _labels(tmp_path / "a.csv", [("https://x.com/1", "direct")])
    second = _labels(tmp_path / "b.csv", [("https://www.x.com/1/", "direct")])
    assert evaluate.load_label_set([first, second]) == {"https://x.com/1": "direct"}


def test_label_set_stops_when_two_files_disagree(tmp_path):
    first = _labels(tmp_path / "a.csv", [("https://x.com/1", "direct")])
    second = _labels(tmp_path / "b.csv", [("https://x.com/1", "not")])
    with pytest.raises(evaluate.LabelConflictError):
        evaluate.load_label_set([first, second])


# ---------- 3. the retrieval score reads version 2 only ----------

def test_score_reads_version_2_and_the_retrieval_sheets_only(tmp_path):
    for name in ("dev_tata_motors_20261014.csv", "dev_tata_motors_20261023.csv",
                 "v2_dev_tata_motors.csv", "dev_tata_motors_retrieval_20261023.csv",
                 "recheck_v2_dev_tata_motors.csv"):
        (tmp_path / name).write_text("row,label,url\n", encoding="utf-8")
    names = [path.name for path in rc.score_label_files(tmp_path)]
    assert names == ["v2_dev_tata_motors.csv", "dev_tata_motors_retrieval_20261023.csv"]


def test_score_stops_without_the_version_2_file(tmp_path):
    (tmp_path / "dev_tata_motors_retrieval_20261023.csv").write_text("row,label,url\n", encoding="utf-8")
    with pytest.raises(rc.RetrievalCheckError):
        rc.score_label_files(tmp_path)


# ---------- 4. the pair record works on any computer ----------

@pytest.mark.parametrize("stored", [
    "C:\\Users\\someone\\news-research-tool\\data\\raw\\20261023_161430_tata-motors.json",
    "/home/someone/news-research-tool/data/raw/20261023_161430_tata-motors.json",
    "20261023_161430_tata-motors.json",
])
def test_raw_path_finds_the_file_in_the_raw_folder(tmp_path, stored):
    assert rc.raw_path(stored, tmp_path) == tmp_path / "20261023_161430_tata-motors.json"


def test_pair_record_stores_file_names_only():
    record = rc.pair_record("Tata Motors", {
        "without_aliases": "C:\\Users\\someone\\data\\raw\\a.json",
        "with_aliases": "/home/someone/data/raw/b.json",
    })
    assert record == {"query": "Tata Motors", "raw_files": {"without_aliases": "a.json", "with_aliases": "b.json"}}
