"""
Tests for src/safe_web.py: the app fetches only public http or https addresses, and checks every redirect step.
No network is used: host lookups and page requests are replaced.
"""

import socket

import pytest

from src import safe_web as sw


def resolver(table):
    def resolve(host, port, proto=None):
        if host not in table:
            raise socket.gaierror("unknown host")
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
                for a in table[host]]
    return resolve


HOSTS = resolver({"news.example.com": ["93.184.216.34"], "local.example.com": ["127.0.0.1"],
                  "lan.example.com": ["192.168.1.10"], "meta.example.com": ["169.254.169.254"],
                  "mixed.example.com": ["93.184.216.34", "10.0.0.5"], "v6.example.com": ["::1"],
                  "mapped.example.com": ["::ffff:192.168.1.10"]})


class Reply:
    def __init__(self, status, text="", location=None):
        self.status_code, self.text = status, text
        self.headers = {"Location": location} if location else {}


@pytest.mark.parametrize("url", ["https://news.example.com/a", "http://news.example.com:8080/b"])
def test_a_public_web_address_is_allowed(url):
    assert sw.address_problem(url, HOSTS) == ""


@pytest.mark.parametrize("url, problem", [
    ("ftp://news.example.com/a", "only http and https"),
    ("javascript:alert(1)", "only http and https"),
    ("https:///no-host", "no host"),
    ("https://unknown.example.com/a", "cannot be found"),
    ("http://local.example.com/a", "not a public"),
    ("http://lan.example.com/a", "not a public"),
    ("http://meta.example.com/latest", "not a public"),
    ("http://mixed.example.com/a", "not a public"),          # one private address is enough to refuse
    ("http://v6.example.com/a", "not a public"),
    ("http://mapped.example.com/a", "not a public"),
    ("http://news.example.com:99999/a", "cannot be read"),
])
def test_other_addresses_are_refused(url, problem):
    assert problem in sw.address_problem(url, HOSTS)


def test_a_page_without_redirects_is_returned_as_it_is():
    sent = []

    def get(url, **options):
        sent.append((url, options["allow_redirects"]))
        return Reply(200, "page")
    assert sw.safe_http_get("https://news.example.com/a", resolve=HOSTS, get=get) == (200, "page")
    assert sent == [("https://news.example.com/a", False)]       # redirects are never followed automatically


def test_a_redirect_to_a_private_address_is_refused_before_it_is_requested():
    sent = []

    def get(url, **options):
        sent.append(url)
        return Reply(302, location="http://lan.example.com/admin")
    with pytest.raises(sw.UnsafeAddress, match="not a public"):
        sw.safe_http_get("https://news.example.com/a", resolve=HOSTS, get=get)
    assert sent == ["https://news.example.com/a"]


def test_a_relative_public_redirect_is_followed_and_too_many_redirects_stop():
    sent = []

    def get(url, **options):
        sent.append(url)
        return Reply(200, "final") if url.endswith("/b") else Reply(301, location="/b")
    assert sw.safe_http_get("https://news.example.com/a", resolve=HOSTS, get=get) == (200, "final")
    assert sent == ["https://news.example.com/a", "https://news.example.com/b"]

    def loop(url, **options):
        return Reply(302, location="/again")
    with pytest.raises(sw.UnsafeAddress, match="redirects"):
        sw.safe_http_get("https://news.example.com/a", resolve=HOSTS, get=loop)


def test_a_refused_page_is_recorded_as_an_error_by_the_full_text_fetcher(tmp_path):
    from src.fulltext import FullTextFetcher, RobotsCheck

    class AllowAll(RobotsCheck):
        def allows(self, url):
            return True

    def get(url, **options):
        raise AssertionError("a private address must never be requested")
    fetcher = FullTextFetcher(http_get=lambda url, timeout=10: sw.safe_http_get(url, timeout, HOSTS, get),
                              cache_dir=tmp_path, delay_seconds=0, robots=AllowAll())
    result = fetcher.fetch("http://lan.example.com/story")
    assert result["status"] == "error" and "not a public" in result["error"]


def test_the_app_and_the_command_line_use_the_safe_request():
    from pathlib import Path
    root = Path(sw.__file__).resolve().parents[1]
    assert "FullTextFetcher(http_get=safe_http_get)" in (root / "src" / "app_flow.py").read_text(encoding="utf-8")
    assert "FullTextFetcher(http_get=safe_http_get)" in (root / "src" / "research.py").read_text(encoding="utf-8")
