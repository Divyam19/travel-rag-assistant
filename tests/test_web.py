from datetime import date

import pytest

from travelrag.db import connect
from travelrag.web import parse_results, spend_credit

TEST_DAY = date(2000, 1, 1)


def test_parse_results_filters_and_dedupes():
    raw = [
        {"title": "A", "url": "https://a.test/x", "content": "text"},
        {"title": "A again", "url": "https://a.test/x/?utm_source=z", "content": "text"},  # same page
        {"title": "empty", "url": "https://b.test", "content": "  "},
        {"title": "bad scheme", "url": "javascript:alert(1)", "content": "text"},
        {"title": "", "url": "https://c.test", "content": "more"},
    ]
    results = parse_results(raw)
    assert [r.url for r in results] == ["https://a.test/x", "https://c.test"]
    assert results[1].title == "https://c.test"  # falls back to the URL when there is no title


@pytest.fixture
def clean_day():
    with connect(with_vectors=False) as conn:
        conn.execute("delete from tavily_usage where day = %s", (TEST_DAY,))
    yield
    with connect(with_vectors=False) as conn:
        conn.execute("delete from tavily_usage where day = %s", (TEST_DAY,))


def test_daily_cap_is_enforced(clean_day):
    with connect(with_vectors=False) as conn:
        conn.autocommit = True
        assert [spend_credit(conn, 3, TEST_DAY) for _ in range(5)] == [True, True, True, False, False]
        assert conn.execute("select calls from tavily_usage where day = %s", (TEST_DAY,)).fetchone()[0] == 3


def test_zero_cap_blocks_everything(clean_day):
    with connect(with_vectors=False) as conn:
        conn.autocommit = True
        assert spend_credit(conn, 0, TEST_DAY) is False
