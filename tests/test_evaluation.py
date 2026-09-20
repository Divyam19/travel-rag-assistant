import pytest

from travelrag import evaluation
from travelrag.evaluation import (LEGAL_CLAIM, Tier1, drop_headings, evaluate_answer, split_sentences,
                                  stray_script, ungrounded_names, ungrounded_numbers)
from travelrag.rag import Source

CONTEXT = ("Qatar - Level 3: Reconsider Travel\n"
           "20 milestone qualifying nights earn 2,500 points. Fan diameter is 69.4 inches. "
           "The ratio drops from 1:1 to 4:3 on October 1. Frankfurt Airport reported the outbreak.")


def test_split_sentences_keeps_claims_only():
    answer = ("**Welcome Bonus**: New members earn 60,000 points [1].\n"
              "Ultimately, the decision depends on your plans.\n"
              "Is it worth it?\n"
              "Please confirm with the official source.\n"
              "The fee is $650 per year for this particular card.")
    # The question, the "ultimately" advice and the "please confirm" hedge make no claim of their own.
    assert split_sentences(answer) == ["Welcome Bonus: New members earn 60,000 points .",
                                       "The fee is $650 per year for this particular card."]


def test_changed_numbers_are_found():
    assert ungrounded_numbers("It is Level 3.", CONTEXT) == []
    assert ungrounded_numbers("It is Level 1.", CONTEXT) == ["1"]
    assert ungrounded_numbers("The fan is 69.4 inches.", CONTEXT) == []
    assert ungrounded_numbers("The fan is 72.0 inches.", CONTEXT) == ["72.0"]
    assert ungrounded_numbers("Ratio 4:3 from 1:1.", CONTEXT) == []
    assert ungrounded_numbers("Ratio drops to 3:2.", CONTEXT) == ["3:2"]
    assert ungrounded_numbers("Earns 2,500 points for 20 nights.", CONTEXT) == []
    assert ungrounded_numbers("Earns 5,000 points.", CONTEXT) == ["5,000"]


def test_small_integers_other_than_levels_are_ignored():
    assert ungrounded_numbers("There are 7 steps and 3 options.", CONTEXT) == []


def test_invented_names_are_found_but_sentence_starts_and_months_are_not():
    assert ungrounded_names("The outbreak reached Munich airport.", CONTEXT) == ["Munich"]
    assert ungrounded_names("Frankfurt Airport reported it in October.", CONTEXT) == []
    assert ungrounded_names("Additionally, Qatar is Level 3.", CONTEXT) == []


def make_sources():
    return [Source(1, "Qatar", "https://x.test", "x.test", None, 0.7, "content", "feed", 1)]


@pytest.fixture
def stub(monkeypatch):
    """Replace embeddings and the judge so verdict routing can be tested without API calls."""
    state = {"tier1": Tier1(0.9, [], [], [], 3), "judge": ({}, 0, 0), "judge_calls": 0}
    monkeypatch.setattr(evaluation, "tier1", lambda *a, **k: state["tier1"])

    def fake_judge(*a, **k):
        state["judge_calls"] += 1
        return state["judge"]

    monkeypatch.setattr(evaluation, "judge", fake_judge)
    return state


def grade(question="How long is the flight?", answer="The flight takes about two hours."):
    return evaluate_answer(question, answer, make_sources(), use_judge=True)


def test_clean_answer_passes_without_calling_the_judge(stub):
    result = grade()
    assert (result.verdict, result.tier, stub["judge_calls"]) == ("pass", 1, 0)


def test_low_similarity_goes_to_the_judge_which_can_clear_it(stub):
    stub["tier1"] = Tier1(0.45, ["odd sentence."], [], [], 3)
    stub["judge"] = ({"unsupported": [], "contradicted": [], "legal_advice": False, "reason": "fine"}, 10, 5)
    result = grade()
    assert (result.verdict, result.tier, stub["judge_calls"]) == ("pass", 2, 1)
    assert result.prompt_tokens == 10


def test_judge_can_flag_unsupported_and_contradicted_claims(stub):
    stub["tier1"] = Tier1(0.9, [], ["5,000"], [], 3)
    stub["judge"] = ({"unsupported": ["a"], "contradicted": ["b"], "legal_advice": False}, 1, 1)
    result = grade()
    assert result.verdict == "flagged" and result.problems == ["a", "b"]


def test_visa_topic_always_reaches_the_judge_and_can_get_a_disclaimer(stub):
    stub["judge"] = ({"unsupported": [], "contradicted": [], "legal_advice": True}, 1, 1)
    result = grade("Do I need a visa for Qatar?", "Yes, you need a visa.")
    assert (result.verdict, stub["judge_calls"]) == ("disclaimer", 1)


def test_unparseable_judge_reply_falls_back_on_tier1_suspicion(stub):
    stub["tier1"] = Tier1(0.9, [], ["72.0"], [], 3)
    stub["judge"] = ({}, 1, 1)
    assert grade().verdict == "flagged"
    stub["tier1"] = Tier1(0.9, [], [], ["Munich"], 3)
    stub["judge"] = ({}, 1, 1)
    assert grade().problems == ["Munich"]


def test_tier1_only_mode_never_calls_the_judge(stub):
    stub["tier1"] = Tier1(0.4, ["weak."], [], [], 3)
    result = evaluate_answer("q", "a", make_sources(), use_judge=False)
    assert (result.verdict, stub["judge_calls"]) == ("flagged", 0)


def test_an_advisory_level_always_reaches_the_judge(stub):
    """A context of several countries contains 'Level 2' somewhere, so the number check alone cannot
    tell a Level 4 country falsely reported as Level 2 from the truth. High similarity must not skip it."""
    stub["judge"] = ({"unsupported": [], "contradicted": ["Ukraine is Level 2"], "legal_advice": False}, 5, 5)
    result = grade("Is Ukraine safe?", "Ukraine is Level 2: Exercise Increased Caution.")
    assert (result.verdict, stub["judge_calls"]) == ("flagged", 1)


def test_a_definite_entry_rule_always_gets_the_disclaimer_even_if_the_judge_disagrees(stub):
    stub["judge"] = ({"unsupported": [], "contradicted": [], "legal_advice": False}, 1, 1)
    result = grade("Vietnam entry?", "US citizens need a visa to enter Vietnam. Please confirm officially.")
    assert result.verdict == "disclaimer" and result.legal


def test_entry_rule_wording_is_recognised():
    assert all(LEGAL_CLAIM.search(t) for t in [
        "US citizens need a visa.", "No visa is required for 30 days.", "You can enter visa-free.",
        "An e-visa is required.", "Indians need an entry permit."])
    assert not any(LEGAL_CLAIM.search(t) for t in [
        "The passport rule is six months.", "The flight costs JPY 14,110.", "Visa Inc. cards earn points."])


def test_stray_script_is_found_but_currency_and_symbols_are_not():
    assert stray_script("not safe \u0628\u0633\u0628\u0628 the war") != []
    assert stray_script("Fares in \u20b9, \u20ac and \u00a3 \u2013 25\u00b0C \u2713 caf\u00e9") == []


def test_a_draft_with_stray_script_is_rejected_without_calling_the_judge(stub):
    result = grade("Is it safe?", "Ukraine is not safe \u0628\u0633\u0628\u0628 the ongoing war today.")
    assert (result.verdict, result.tier, stub["judge_calls"]) == ("flagged", 1, 0)


def test_headings_are_not_reported_as_unsupported_claims():
    answer = "## Kanpur to Delhi\nNo source covers this.\n## Kanpur to Bangalore\nIndiGo flies it."
    assert drop_headings(answer, ["Kanpur to Delhi", "IndiGo flies it"]) == ["IndiGo flies it"]


def test_statements_about_source_silence_are_not_claims():
    assert split_sentences("The source does not give a release date for Canada figures.") == []
    assert split_sentences("No rail strike in France is listed for this week.") == []
    assert split_sentences("The Nozomi costs JPY 14,110 one way to Tokyo.") != []


def test_a_labelled_estimate_is_allowed_but_only_where_the_marker_reaches():
    ctx = "Bus pass JPY 500. Train JPY 14110."
    assert ungrounded_numbers("Estimated transport subtotal: roughly JPY 14,760 for the trip.", ctx) == []
    # An unrelated hedge earlier on the line must not shield a falsified figure later on it.
    assert ungrounded_numbers("Roughly speaking there was plenty of demand and prices rose to JPY 9,999 overall.", ctx) != []


def test_accents_and_apostrophes_do_not_create_invented_names():
    ctx = "Travel to Cancun and Potosi. The FTC study."
    assert ungrounded_names("Avoid Canc\u00fan and Potos\u00ed, and note the FTC\u2019s study.", ctx) == []
    assert ungrounded_names("The outbreak reached Munich.", ctx) == ["Munich"]


def test_bullet_fragments_without_full_stops_are_checked():
    """Regression: an answer written as bullets yielded no claims, scored a vacuous 1.00 and skipped
    every check, so a falsified 72.0 in 'Fan diameter: 72.0 inches' was never examined."""
    answer = ("- Fan diameter of the CFM LEAP-1B engine: 72.0 inches [2]\n"
              "- Fan diameter of the CFM LEAP-1A engine: 78 inches [2]\n"
              "## Summary\n"
              "Key points include:\n"
              "Which airline?")
    found = split_sentences(answer)
    assert len(found) == 2 and "72.0 inches" in found[0]  # bullets kept; heading, intro and question dropped


def test_offers_of_further_help_are_not_claims():
    for offer in ["If you want, I can also turn this into a short timeline.",
                  "I can also summarize the State Department advice for Thailand.",
                  "Let me know if you would like a comparison table."]:
        assert split_sentences(offer) == [], offer


def test_no_extractable_claims_is_not_reported_as_a_perfect_score(monkeypatch):
    monkeypatch.setattr(evaluation, "embed_texts", lambda *a, **k: [[1.0, 0.0]])
    result = evaluation.tier1("Ok.", make_sources())
    assert result.sentences == 0 and result.min_sim == 1.0  # nothing to distrust when nothing is asserted
    result = evaluation.tier1("Ok. 999", make_sources())
    assert result.min_sim == 0.0  # ...but an unverifiable figure with no sentence around it is not 'fine'


def test_evidence_embeds_the_plain_text_and_only_displays_the_date():
    from datetime import datetime
    source = Source(1, "Title", "https://x.test", "x.test", datetime(2026, 9, 18), 0.7,
                    "A long enough sentence about jet fuel costs.", "feed", 1)
    pairs = evaluation.context_evidence([source])
    assert all("[2026-09-18]" not in embedded for embedded, _ in pairs)
    assert all(shown.endswith("[2026-09-18]") for _, shown in pairs)


def test_the_judge_is_shown_the_table_row_that_holds_the_claimed_figure():
    """A true fee sat in a table row that embeds poorly, so it was not among the nearest lines and
    the judge marked it unsupported. Lines that literally contain the figure are now added."""
    shown = ["Some unrelated sentence about lounges and airports.",
             "| Chase Sapphire Reserve | 200,000 points | $4,100 | $795 |",
             "Another unrelated line about hotels."]
    found = evaluation.figure_evidence("Annual fee: $795 per year", shown, already={0})
    assert found and "$795" in found[0]
    assert evaluation.figure_evidence("It is number 3 of 7.", shown, already=set()) == []  # bare small ints skipped


def test_a_long_table_row_is_windowed_around_the_figure_not_cut_before_it():
    row = "| " + "filler " * 120 + "| fan diameter 69.4 inches | " + "trailing " * 60
    pattern = evaluation._figure_pattern("69.4")
    assert "69.4" in evaluation._window(row, pattern)


def test_claims_quoting_unverifiable_figures_are_judged_before_merely_weak_ones():
    t1 = Tier1(0.4, [], ["18.4%"], [], 3, claims=[
        ("It is a fine card overall for many people.", 0.30, ["a"]),
        ("Tourism fell by 18.4% in August.", 0.70, ["b"]),
        ("Ukraine is Level 2 on the advisory scale.", 0.65, ["c"]),
    ])
    order = [c[0] for c in evaluation.select_claims(t1)]
    assert order[0].startswith("Tourism fell") and order[1].startswith("Ukraine")


def test_at_most_six_claims_are_sent_to_the_judge():
    t1 = Tier1(0.2, [], [], [], 20, claims=[(f"Claim number {i} is here today.", 0.1 + i / 100, ["e"]) for i in range(20)])
    assert len(evaluation.select_claims(t1)) == 6
