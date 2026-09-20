from travelrag.eval_check import tamper


def test_tamper_uses_the_first_phrasing_that_appears():
    case = {"replace": [["4:3", "3:2"], ["4 to 3", "3 to 2"]]}
    assert tamper(case, "The ratio drops to 4:3 on Oct 1.") == "The ratio drops to 3:2 on Oct 1."
    assert tamper(case, "The ratio drops from 1 to 1 to 4 to 3.") == "The ratio drops from 1 to 1 to 3 to 2."


def test_a_case_that_cannot_be_applied_is_skipped_not_fatal():
    """A model rewording an answer used to abort the whole validation run."""
    assert tamper({"replace": [["64 business", "24 business"]]}, "It has sixty-four seats.") is None


def test_a_single_pair_and_an_append_case_still_work():
    assert tamper({"replace": ["11.8%", "18.4%"]}, "Fell 11.8% in August.") == "Fell 18.4% in August."
    assert tamper({"append": " Extra."}, "Base.") == "Base. Extra."
