"""Answer evaluation: cheap checks first, a small-model judge only when they are inconclusive.

Tier 1 (no LLM call, costs one small embedding request):
  - each claim sentence of the answer is embedded and compared with the context chunks;
  - every number in the answer must appear in the context text, which catches altered figures
    that similarity cannot see ("Level 3" changed to "Level 1" is still a very similar sentence).
Tier 2 (small model): checks only the claims Tier 1 found suspicious, each with its closest source
  lines, when Tier 1 is inconclusive or the answer touches visa rules or advisory levels.
Tier 3 (what to do with a flagged answer) lives in agent.py.
"""

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .config import get_settings
from .llm import chat, embed_texts, parse_json
from .rag import Source

DISCLAIMER = ("\n\nNote: entry and visa rules depend on your nationality and change often. "
              "This is not legal advice; please confirm with the official government source or the embassy.")
CAVEAT = "\n\nNote: I couldn't verify the following against my sources, so treat it with caution:\n{items}"

CITATION = re.compile(r"\[\d+\](?:\[\d+\])*")
LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
# Sentences that hedge or point elsewhere make no factual claim of their own.
NON_CLAIM = re.compile(r"(?i)(?:^note:|^\s*note\b.{0,12}:"
                       r"|\b(?:source|sources|article|articles|page|pages|text)\b[^.]{0,50}\b(?:is|are) dated\b"
                       r"|\bnot (?:visible|shown|included|stated|given) in the (?:provided|given|available) (?:text|sources?|material)"
                       r"|\bi (?:cannot|can't|could not|couldn't) (?:identify|confirm|verify|tell|find)"
                       r"|\b(?:article|source|sources|page) (?:does not|doesn't|do not|did not) (?:quantify|say|state|give|mention|list|cover|specify|include)"
                       r"|\bmay be out of date|\bcould change the overall picture)"
                       r"|\b(if you (?:want|like|would like|prefer)|i can (?:also )?(?:turn|give|make|summari[sz]e|provide|help|compare|list|put|share)"
                       r"|let me know|feel free to ask|happy to help"
                       r"|confirm|check with|consult|official|not legal advice|i couldn.t|not (?:specified|mentioned|provided)"
                       r"|sources? (?:do(?:es)?n?.?t|do(?:es)? not)|no information|unclear"
                       r"|does not (?:give|list|cover|include|state|mention|specify)|is not listed|are not listed"
                       r"|no .{0,30}(?:is|are) (?:listed|given|provided|mentioned)"
                       r"|ultimately|depends on|should be based on|you (?:may|might|could) (?:want|consider)|consider (?:your|whether)"
                       r"|it.s (?:important|advisable|recommended)|this can be ideal)\b")
# Advisory levels are the highest-stakes fact this assistant states. The number check cannot verify
# them alone: a context holding several countries' advisories contains "Level 2" somewhere, so a
# Level 4 country falsely reported as Level 2 sails through. Any answer quoting a level goes to
# the judge, which reads the sentence against its own source.
ADVISORY_LEVEL = re.compile(r"(?i)\blevel\s*[1-4]\b")
# A definite statement of an entry rule ("you need a visa", "no visa required"). Detected in code,
# not left to the judge: a model that has just been told to add "confirm with the official source"
# tends to conclude the answer is already safe, and the disclaimer stopped appearing.
LEGAL_CLAIM = re.compile(
    r"(?i)\b(?:need(?:s|ed)?\s+(?:a\s+|an\s+|to\s+(?:have|get|obtain)\s+)?(?:valid\s+)?(?:e-?)?visas?"
    r"|(?:e-?)?visas?\s+(?:is|are|will be)\s+(?:required|needed|mandatory|not required|not needed)"
    r"|(?:no|without\s+a)\s+(?:e-?)?visa|visa[- ](?:free|exempt|on arrival)"
    r"|must\s+(?:obtain|have|apply|hold)|entry permit|permit\s+(?:is\s+)?required)")
LEGAL_TOPIC = re.compile(r"(?i)\b(visas?|e-?visas?|immigration|passports?|entry requirements?|overstay\w*|residency"
                         r"|work permits?|deport\w*|customs)\b")
NUMBER = re.compile(r"(?P<num>\d+:\d+|[$£€]?\d[\d,]*(?:\.\d+)?%?)")  # ratios like 4:3 count as one number
# A total the assistant worked out from sourced figures is allowed, as long as it says so. Numbers
# in such a sentence are not checked against the sources, because they are arithmetic, not quotes.
ESTIMATE = re.compile(r"(?i)(\bestimat\w+|\broughly\b|\bballpark\b|\bsub-?total\b|\bin total\b|\badds? up to\b|~)")
ESTIMATE_REACH = 40  # characters after the marker that the exemption covers

JUDGE_PROMPT = """You verify individual claims taken from a travel assistant's answer. Each numbered
CLAIM is followed by EVIDENCE: the lines from the assistant's sources that are closest to it, with
the source date in brackets when known. Reply with JSON only:
{"unsupported": [<claim numbers>], "contradicted": [<claim numbers>], "reason": "<one sentence>"}

- supported (leave it out): the evidence states the claim, or it is a fair paraphrase of it.
- contradicted: the evidence says something different (another number, level, date or fact).
- unsupported: the claim asserts a checkable fact (a price, number, duration, date, name, rule or
  statistic) that the evidence does not contain.
- You only see the closest lines, not every source. Do not mark a claim unsupported merely because
  its wording differs from the evidence. But a claim that names a specific benefit, product, place,
  event, organisation or announcement which appears in NONE of its evidence lines is unsupported:
  "the card includes free Global Entry" is unsupported if no line mentions Global Entry.
- A comparison or conclusion that follows from figures the answer already cites ("the LEAP-1B is
  smaller than the LEAP-1A" after citing 69.4 and 78 inches; a total of listed prices; "cheaper
  than") is supported. Only a conclusion that needs a fact you have not been shown is unsupported.
- Planning and arrangement language asserts no checkable fact ("Day 1: arrive in Osaka", "spend the
  day in Kyoto"): supported. So is an estimate or rough total the claim labels as such, and any
  statement about what the sources do or do not contain.
- A claim about "now", "this week" or "currently" is unsupported when the evidence is dated to a
  different period from the one implied.
- Use empty lists when every claim is fine. Be strict about facts and lenient about shape."""

@dataclass
class Tier1:
    min_sim: float
    weak: list[str]  # sentences below the pass threshold
    bad_numbers: list[str]  # numbers in the answer that do not appear in the sources
    bad_names: list[str]  # capitalised words in the answer that do not appear in the sources
    sentences: int
    # (sentence, best similarity, the closest source lines) for every claim sentence. The judge is
    # shown only the suspicious ones with their evidence, not the whole context again.
    claims: list[tuple[str, float, list[str]]] = field(default_factory=list)


@dataclass
class EvalResult:
    verdict: str  # pass | disclaimer | flagged
    tier: int  # highest tier that ran
    min_sim: float
    problems: list[str] = field(default_factory=list)  # fragments to remove or correct
    legal: bool = False
    notes: list[str] = field(default_factory=list)
    facts_bad: bool = False  # Tier 1 found numbers or names the sources do not contain
    legal_topic: bool = False  # answer or question touches visa/immigration rules
    prompt_tokens: int = 0
    completion_tokens: int = 0


FOREIGN_SCRIPT = re.compile(r"[^\x00-\u024F\u2000-\u206F\u20A0-\u20CF\u2190-\u21FF\u2200-\u22FF\u25A0-\u25FF\u2600-\u27BF\ufe0f]")


def stray_script(answer: str) -> list[str]:
    """Characters from another alphabet in an English answer, e.g. an Arabic word mid-sentence."""
    return sorted(set(FOREIGN_SCRIPT.findall(answer)))


def clean(text: str) -> str:
    text = CITATION.sub("", text).replace("**", "").replace("__", "")
    return LIST_MARKER.sub("", text)


def split_sentences(answer: str) -> list[str]:
    """The checkable statements in an answer: sentences and list items, not headings, questions,
    list intros ("...:") or hedges.

    Bullet answers matter. "Fan diameter of the LEAP-1B: 72.0 inches" ends in no full stop, and an
    earlier version that demanded one extracted no claims at all from such answers, which then
    scored a vacuous perfect similarity and skipped every check.
    """
    statements = []
    for line in clean(answer).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            words = len(sentence.split())
            long_enough = words >= 5 or (words >= 3 and any(c.isdigit() for c in sentence))
            if long_enough and not sentence.endswith((":", "?")) and not NON_CLAIM.search(sentence):
                statements.append(sentence)
    return statements


def _estimate_ranges(text: str) -> list[tuple[int, int]]:
    """Spans just after an estimate marker, where a figure is arithmetic rather than a quote.

    Deliberately narrow. An earlier version exempted the whole line and treated "approximately" as
    a marker, which let "dropped by 18.4% ... approximately 3.1 million visitors" hide a falsified
    percentage behind an unrelated hedge.
    """
    return [(m.end(), min(m.end() + ESTIMATE_REACH, len(text))) for m in ESTIMATE.finditer(text)]


def ungrounded_numbers(answer: str, context: str) -> list[str]:
    """Numbers in the answer that the context does not contain.
    - ratios (4:3), 3+ digits, decimals, percentages and currency must appear as written;
    - a 1-2 digit integer above 12 must appear as a standalone number;
    - integers up to 12 are too common to check, except advisory levels ("Level 3"), which must
      appear together with the word "level";
    - a figure just after an estimate marker ("estimated", "roughly", "in total") is skipped: the
      assistant is allowed to add sourced numbers up, so that is arithmetic rather than a quote.
    Titles belong in the context: "Level 3" lives in a State Dept title, not its body."""
    haystack = context.lower().replace(",", "")
    text = clean(answer)
    estimates = _estimate_ranges(text)
    missing = []
    for m in NUMBER.finditer(text):
        if any(start <= m.start() < end for start, end in estimates):
            continue
        number = m.group("num").lower().rstrip(".,").replace(",", "")
        if ":" in number or len(re.sub(r"\D", "", number)) >= 3 or any(c in number for c in "%.$\u00a3\u20ac"):
            found = number in haystack
        elif int(number) > 12:
            found = re.search(rf"(?<![\d.]){number}(?![\d.])", haystack) is not None
        elif re.findall(r"[A-Za-z]+", text[: m.start()])[-1:] == ["Level"]:
            found = f"level {number}" in haystack
        else:
            continue
        if not found:
            missing.append(m.group(0).strip())
    return missing


COMMON_CAPITALS = {"uk", "us", "usa", "eu", "un", "state", "i", "im", "id", "ill", "ive",
                   "january", "february", "march", "april", "may", "june", "july", "august",
                   "september", "october", "november", "december", "sept", "sep", "jan", "feb",
                   "mar", "apr", "jun", "jul", "aug", "oct", "nov", "dec",
                   "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                   "usd", "eur", "gbp", "inr", "jpy"}
# Letters include accented ones: a token regex limited to [A-Za-z] cut "Cancun" out of "Cancún"
# and then reported the fragment as an invented name.
WORD = re.compile(r"[^\W\d_][\w'\u2019.-]*", re.UNICODE)
CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n|:\s+|\s[-\u2013\u2014]\s")


def _normalise(token: str) -> str:
    """Fold a word to its comparable core: lowercase, no possessive, no punctuation, no accents."""
    import unicodedata

    plain = unicodedata.normalize("NFKD", token.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", re.sub(r"['\u2019]s$", "", plain))


def ungrounded_names(answer: str, context: str) -> list[str]:
    """Capitalised words in the answer that never occur in the context, e.g. an invented city or
    product. The first word of every sentence, line or clause after a colon is skipped: it is
    capitalised whatever it is."""
    words = {_normalise(w) for w in WORD.findall(context)}
    words.discard("")
    missing = []
    for clause in CLAUSE_SPLIT.split(clean(answer)):
        for token in WORD.findall(clause)[1:]:
            if not token[:1].isupper():
                continue
            for part in token.split("-"):
                norm = _normalise(part)
                if len(norm) > 1 and norm not in COMMON_CAPITALS and norm not in words \
                        and norm.rstrip("s") not in words:
                    missing.append(token.rstrip(".,;:"))
                    break
    return missing


EVIDENCE_PER_CLAIM = 4
MAX_CLAIMS = 6
EVIDENCE_CHARS = 320


def context_evidence(sources: list[Source]) -> list[tuple[str, str]]:
    """(text to embed, text to show the judge) for every title and sentence of the context.

    The date stamp is display-only. Embedding it shifted every similarity score and moved answers
    across the calibrated pass threshold.
    """
    out = []
    for source in sources:
        stamp = f" [{source.published_at.date().isoformat()}]" if source.published_at else ""
        out.append((source.title, f"{source.title}{stamp}"))
        for line in source.content.splitlines():
            for part in re.split(r"(?<=[.!?])\s+", line.strip()):
                if len(part.split()) >= 4:
                    out.append((part.strip(), f"{part.strip()}{stamp}"))
    return out


def _figure_pattern(number: str) -> re.Pattern[str] | None:
    """How a figure must appear in a source line to count as the same figure. Bare small integers
    are skipped: they match almost any line."""
    digits = re.sub(r"\D", "", number)
    if len(digits) < 2 and not any(c in number for c in "%.$\u00a3\u20ac"):
        return None
    core = re.escape(number.lstrip("$\u00a3\u20ac").rstrip("%"))
    return re.compile(rf"(?<![\d.]){core}(?![\d.])")


def _window(text: str, pattern: re.Pattern[str]) -> str:
    """The stretch of a long line around the figure, so a table row is not cut off before it."""
    m = pattern.search(text.replace(",", ""))
    if len(text) <= EVIDENCE_CHARS or not m:
        return text[:EVIDENCE_CHARS]
    start = max(0, m.start() - EVIDENCE_CHARS // 2)
    return ("..." if start else "") + text[start : start + EVIDENCE_CHARS]


def figure_evidence(sentence: str, shown: list[str], already: set[int]) -> list[str]:
    """Source lines that literally contain a figure the claim quotes. Semantic similarity is poor
    at table rows ("| Card | 200,000 points | $4,100 | $795 |"), so a true fee or spec was being
    judged against four unrelated lines and flagged unsupported."""
    found: list[str] = []
    for m in NUMBER.finditer(sentence):
        number = m.group("num").replace(",", "")
        pattern = _figure_pattern(number)
        if pattern is None:
            continue
        for i, line in enumerate(shown):
            if i not in already and pattern.search(line.replace(",", "")):
                already.add(i)
                found.append(_window(line, pattern))
                break
        if len(found) >= 2:
            break
    return found


def tier1(answer: str, sources: list[Source], run_label: str = "adhoc", query_id: str | None = None) -> Tier1:
    context = "\n".join(f"{s.title}\n{s.content}" for s in sources)
    bad_numbers = ungrounded_numbers(answer, context)
    bad_names = ungrounded_names(answer, context)
    sentences = split_sentences(answer)
    pairs = context_evidence(sources)
    evidence = [shown for _, shown in pairs]
    if not sentences or not pairs:
        # Nothing checkable was extracted. Say so plainly rather than reporting a perfect score.
        return Tier1(0.0 if bad_numbers or bad_names else 1.0, [], bad_numbers, bad_names, len(sentences))
    # One embedding request for answer and context sentences together.
    vectors = np.array(embed_texts(sentences + [text for text, _ in pairs], "embed_eval", run_label, query_id),
                       dtype=np.float32)
    answer_vectors, evidence_vectors = vectors[: len(sentences)], vectors[len(sentences):]
    similarity = answer_vectors @ evidence_vectors.T  # unit vectors: dot product = cosine
    best = similarity.max(axis=1)
    threshold = get_settings().eval_sim_pass
    weak = [s for s, sim in zip(sentences, best) if sim < threshold]
    claims = []
    for i, sentence in enumerate(sentences):
        nearest = [int(j) for j in np.argsort(-similarity[i])[:EVIDENCE_PER_CLAIM]]
        lines = [evidence[j][:EVIDENCE_CHARS] for j in nearest]
        lines += figure_evidence(sentence, evidence, set(nearest))
        claims.append((sentence, float(best[i]), lines))
    return Tier1(float(best.min()), weak, bad_numbers, bad_names, len(sentences), claims)


def select_claims(t1: Tier1) -> list[tuple[str, float, list[str]]]:
    """The claims worth a model's attention: weakly matched, carrying an unverifiable number or
    name, or high-stakes (an advisory level, an entry rule). Lowest similarity first, capped."""
    threshold = get_settings().eval_sim_pass

    def suspicious(sentence: str, sim: float) -> bool:
        return (sim < threshold or any(n in sentence for n in t1.bad_numbers)
                or any(n in sentence for n in t1.bad_names)
                or bool(LEGAL_CLAIM.search(sentence) or ADVISORY_LEVEL.search(sentence)))

    def priority(claim: tuple[str, float, list[str]]) -> tuple[int, float]:
        sentence, sim, _ = claim
        if any(n in sentence for n in t1.bad_numbers) or any(n in sentence for n in t1.bad_names):
            rank = 0  # quotes a figure or name that appears nowhere in the sources
        elif LEGAL_CLAIM.search(sentence) or ADVISORY_LEVEL.search(sentence):
            rank = 1  # high-stakes: an entry rule or an advisory level
        else:
            rank = 2  # merely a weak match
        return rank, sim

    picked = [c for c in t1.claims if suspicious(c[0], c[1])]
    if not picked:  # routed for topic alone (a visa word, say): check the least-supported sentences
        picked = list(t1.claims)
    return sorted(picked, key=priority)[:MAX_CLAIMS]


def judge(question: str, claims: list[tuple[str, float, list[str]]], run_label: str,
          query_id: str | None) -> tuple[dict, int, int]:
    """Ask the small model to verify specific claims. Returns ({unsupported, contradicted, reason},
    prompt tokens, completion tokens), with claim numbers already mapped back to sentences."""
    blocks = []
    for n, (sentence, _, lines) in enumerate(claims, start=1):
        blocks.append(f"{n}. {sentence}\n   evidence:\n" + "\n".join(f"   - {line}" for line in lines))
    user = f"Today's date: {date.today().isoformat()}\nQuestion: {question}\n\nCLAIMS:\n" + "\n".join(blocks)
    text, prompt, completion = chat(
        [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
        get_settings().small_model, "eval_judge", run_label, query_id, json_mode=True, temperature=0,
    )
    verdict = parse_json(text)
    if not verdict:
        return {}, prompt, completion

    def resolve(items) -> list[str]:
        out = []
        for item in items or []:
            if isinstance(item, int) or (isinstance(item, str) and item.strip().isdigit()):
                index = int(item) - 1
                if 0 <= index < len(claims):
                    out.append(claims[index][0])
            elif isinstance(item, str) and item.strip():
                out.append(item)  # the model quoted text instead of numbering it
        return out

    return ({"unsupported": resolve(verdict.get("unsupported")), "contradicted": resolve(verdict.get("contradicted")),
             "reason": verdict.get("reason", "")}, prompt, completion)


def drop_headings(answer: str, problems: list[str]) -> list[str]:
    """Discard flagged fragments that are just a section heading of the answer. A heading asserts
    nothing on its own, and quoting one back as "unsupported" produces a meaningless caveat."""
    headings = {line.lstrip("#").strip().lower() for line in answer.splitlines() if line.lstrip().startswith("#")}
    return [p for p in problems if p.strip().lower() not in headings]


def evaluate_answer(question: str, answer: str, sources: list[Source], run_label: str = "adhoc",
                    query_id: str | None = None, use_judge: bool = True) -> EvalResult:
    """Grade a draft answer against `sources`, the exact context it was written from."""
    s = get_settings()
    t1 = tier1(answer, sources, run_label, query_id)
    facts_bad = bool(t1.bad_numbers or t1.bad_names)
    legal_topic = bool(LEGAL_TOPIC.search(answer) or LEGAL_TOPIC.search(question))
    legal_claim = bool(LEGAL_CLAIM.search(clean(answer)))
    high_stakes = legal_topic or bool(ADVISORY_LEVEL.search(answer))
    foreign = stray_script(answer)
    notes = [f"tier1: min sentence similarity {t1.min_sim:.2f} over {t1.sentences} sentences"
             + (f"; numbers not in sources {t1.bad_numbers}" if t1.bad_numbers else "")
             + (f"; names not in sources {t1.bad_names}" if t1.bad_names else "")]
    common = {"min_sim": t1.min_sim, "facts_bad": facts_bad, "legal_topic": legal_topic}

    if foreign:  # cheap and unambiguous: reject without asking a model
        return EvalResult("flagged", 1, problems=[f"text contains characters from another script: {''.join(foreign)}"],
                          notes=notes + ["tier1: stray non-Latin script"], **common)
    if t1.min_sim >= s.eval_sim_pass and not facts_bad and not high_stakes:
        if legal_claim:  # a definite entry rule always carries the disclaimer, judge or no judge
            return EvalResult("disclaimer", 1, legal=True, notes=notes + ["tier1: entry-rule statement"], **common)
        return EvalResult("pass", 1, notes=notes, **common)
    suspicious = t1.weak or t1.bad_numbers + t1.bad_names
    if t1.min_sim < s.eval_sim_fail:
        return EvalResult("flagged", 1, problems=t1.weak, notes=notes + ["tier1: clearly ungrounded"], **common)
    if not use_judge:
        return EvalResult("flagged" if suspicious else "pass", 1, problems=suspicious, legal=legal_topic,
                          notes=notes, **common)

    verdict, prompt_tokens, completion_tokens = judge(question, select_claims(t1), run_label, query_id)
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if not verdict:  # unparseable judge output: fall back on Tier 1's own suspicion
        return EvalResult("flagged" if suspicious else "pass", 2, problems=suspicious, legal=legal_topic,
                          notes=notes + ["tier2: judge reply unparseable"], **common, **usage)
    problems = drop_headings(answer, [str(x) for x in
                                      (verdict.get("unsupported") or []) + (verdict.get("contradicted") or [])])
    legal = legal_claim  # detected in code; the judge no longer sees the whole answer to decide it
    notes.append(f"tier2: {verdict.get('reason', '')}".strip())
    if problems:
        return EvalResult("flagged", 2, problems=problems, legal=legal, notes=notes, **common, **usage)
    return EvalResult("disclaimer" if legal else "pass", 2, legal=legal, notes=notes, **common, **usage)
