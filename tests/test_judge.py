import pytest

from vault_rag.judge import (PAD, wilson, judge_prompt, judge_one, length_bias, pad, pair_prompt, parse_score,
                             parse_winner, position_bias, validate)


def test_parse_score_reads_json_and_ignores_scores_in_the_reason():
    assert parse_score('{"reason": "gives 2 of 2 facts", "score": 2}') == 2
    assert parse_score('Sure.\n```json\n{"reason": "wrong", "score": 0}\n```') == 0
    with pytest.raises(ValueError):
        parse_score("I think it is mostly right")
    with pytest.raises(ValueError):
        parse_score('{"score": 7}')


def test_parse_winner():
    assert parse_winner('{"winner": "B"}') == "B"
    assert parse_winner('{"winner": "tie"}') == "tie"
    with pytest.raises(ValueError):
        parse_winner('{"winner": "C"}')


def test_prompts_carry_question_gold_and_answers_but_judge_prompt_hides_labels():
    p = judge_prompt("Q?", "gold fact", "the answer")
    assert "Q?" in p and "gold fact" in p and "the answer" in p
    pp = pair_prompt("Q?", "gold fact", "first ans", "second ans")
    assert pp.index("first ans") < pp.index("second ans")


def test_judge_one_returns_parsed_score():
    assert judge_one(lambda p: '{"reason": "ok", "score": 1}', "q", "g", "a") == 1


def test_validate_reports_agreement_and_kappa():
    human = [2, 2, 2, 2, 0, 0, 0, 1]
    judge = [2, 2, 2, 1, 0, 0, 2, 1]
    v = validate(human, judge)
    assert v["n"] == 8 and v["agreement"] == 0.75
    assert v["kappa_3class"] == pytest.approx(0.6)
    assert v["binary_agreement"] == 0.75
    assert v["kappa_binary"] == pytest.approx(0.5)
    assert v["confusion"] == {"2": {"2": 3, "1": 1}, "0": {"0": 2, "2": 1}, "1": {"1": 1}}


def test_position_bias_detects_a_judge_that_always_says_a():
    pairs = [("q1", "g", "good1", "bad1"), ("q2", "g", "good2", "bad2")]
    r = position_bias(lambda p: '{"winner": "A"}', pairs)
    assert r == {"n": 2, "pick_first_rate": 1.0, "acc_good_first": 1.0, "acc_good_second": 0.0,
                 "flip_rate": 1.0, "acc_overall": 0.5}


def test_position_bias_is_clean_for_a_judge_that_reads_the_answer():
    def llm(p):
        a, b = p.split("Answer A:")[1].split("Answer B:")
        return '{"winner": "%s"}' % ("A" if "good" in a else "B")

    r = position_bias(llm, [("q", "g", "good answer", "bad answer")] * 3)
    assert r["acc_overall"] == 1.0 and r["flip_rate"] == 0.0 and r["pick_first_rate"] == 0.5


def test_pad_appends_filler_without_changing_the_start():
    assert pad("Two tonnes.").startswith("Two tonnes.") and len(pad("Two tonnes.").split()) > 40


def test_length_bias_flags_a_judge_that_scores_by_length():
    rows = [("q", "g", "short wrong", 0), ("q", "g", "short right", 2), ("q", "g", "also wrong", 0)]

    def long_is_good(p):
        ans = p.split("Answer to grade:")[1]
        return '{"reason": "", "score": %d}' % (2 if len(ans.split()) > 30 else 0)

    r = length_bias(long_is_good, rows)
    assert r["n"] == 3 and r["mean_delta"] == 2.0 and r["pct_raised"] == 1.0
    assert r["wrong_promoted"] == 1.0


def test_wilson_interval_for_perfect_agreement_on_forty():
    lo, hi = wilson(40, 40)
    assert lo == pytest.approx(0.9124, abs=1e-3) and hi == pytest.approx(1.0)
    lo, hi = wilson(30, 40)
    assert lo == pytest.approx(0.5976, abs=2e-3) and hi == pytest.approx(0.858, abs=2e-3)
