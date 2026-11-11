import pytest

from currency import possibly_outdated
from evaluate import calibrate_threshold, load_questions, parse_config


def test_parse_config_accepts_glossary_and_rejects_unknown():
    assert parse_config("hybrid+lookup+glossary") == {
        "mode": "hybrid", "lookup": True, "rerank": False, "glossary": True,
    }
    with pytest.raises(ValueError):
        parse_config("vector+lookup+lookup")
    with pytest.raises(ValueError):
        parse_config("hybrid+magic")
    with pytest.raises(ValueError):
        parse_config("semantic")


def test_load_questions_reads_type_column(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text(
        "question,expected_articles,notes,type\n"
        "Pergunta A,49,,lookup\nPergunta B,18;19,x,\n", encoding="utf-8")
    rows = load_questions(path)
    assert [r["type"] for r in rows] == ["lookup", "natural"]
    assert rows[1]["expected"] == {"18", "19"}


def test_load_questions_works_without_type_column(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text("question,expected_articles\nA,1\n", encoding="utf-8")
    assert load_questions(path)[0]["type"] == "natural"


def test_calibrate_threshold_separable():
    result = calibrate_threshold([0.10, 0.20], [0.40, 0.50])
    assert result["threshold"] == pytest.approx(0.30)
    assert result["false_refusals"] == 0 and result["false_answers"] == 0


def test_calibrate_threshold_overlapping_minimises_errors():
    result = calibrate_threshold([0.10, 0.20, 0.45], [0.40, 0.50, 0.60])
    assert result["false_refusals"] + result["false_answers"] == 1


def test_possibly_outdated_marker_articles():
    assert possibly_outdated(["1", "42-A", "119"]) is True      # PDF de 2013
    assert possibly_outdated(["1", "54-A", "104-A"]) is False   # após a Lei 14.181/2021
