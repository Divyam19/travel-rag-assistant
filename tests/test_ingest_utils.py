import feedparser

from travelrag.ingest import content_hash, feed_entry_text, normalize_url


def test_normalize_url_drops_tracking_and_fragment():
    a = normalize_url("https://Example.com/Post/?utm_source=x&id=7#comments")
    assert a == "https://example.com/Post?id=7"


def test_normalize_url_treats_trailing_slash_as_same():
    assert normalize_url("https://example.com/a/") == normalize_url("https://example.com/a")


def test_content_hash_ignores_case_and_whitespace():
    assert content_hash("Hello   World\n") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello there")


def test_feed_entry_text_strips_html_and_keeps_paragraphs():
    entry = feedparser.FeedParserDict(summary="<p>First&nbsp;para.</p><p>Second <b>para</b>.</p>")
    assert feed_entry_text(entry) == "First para.\n\nSecond para."


def test_feed_entry_text_empty():
    assert feed_entry_text(feedparser.FeedParserDict(summary="  ")) == ""


def test_probe_and_ingester_agree_on_what_counts_as_usable():
    """The probe once judged 'usable' at 150 words while the ingester accepted 120, so its report
    did not predict what ingestion would store. It now imports the ingester's threshold."""
    from travelrag import ingest, probe
    assert probe.MIN_WORDS_FETCHED is ingest.MIN_WORDS_FETCHED
