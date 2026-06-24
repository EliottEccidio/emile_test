"""Tests des metriques GSM8K (logique pure, sans modele ni reseau)."""
from src.metrics import (accuracy, extract_gold_answer, extract_pred_answer,
                         is_correct)


def test_extract_gold_answer():
    assert extract_gold_answer("blabla raisonnement...\n#### 18") == "18"
    assert extract_gold_answer("#### 1,200") == "1200"
    assert extract_gold_answer("#### -5") == "-5"


def test_extract_pred_after_marker():
    assert extract_pred_answer("etapes...\n#### 42") == "42"
    assert extract_pred_answer("Donc #### 3.5 au final") == "3.5"


def test_extract_pred_last_number_fallback():
    assert extract_pred_answer("la reponse est 7 dollars") == "7"
    assert extract_pred_answer("il y a 12 puis 9 pommes") == "9"
    assert extract_pred_answer("aucun nombre ici") is None


def test_is_correct():
    assert is_correct("18", "18")
    assert is_correct("18.0", "18")        # egalite numerique
    assert not is_correct("19", "18")
    assert not is_correct(None, "18")


def test_accuracy():
    assert accuracy(["1", "2", "9"], ["1", "2", "3"]) == 2 / 3
    assert accuracy([], []) == 0.0
