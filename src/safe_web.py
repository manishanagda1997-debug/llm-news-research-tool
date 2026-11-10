"""
Safe page requests for the app: only public web addresses are fetched.

The app reads each article's page from the link NewsAPI gives. Before every request, and again at every redirect,
the address must be an http or https link whose host resolves only to public internet addresses: never this
computer (localhost), a private or local network, or another reserved range. Redirects are followed one step at a
time (at most MAX_REDIRECTS), so a public page cannot send the request on to a private address.

The check resolves the host just before each request. A host whose address changes between the check and the
request (DNS rebinding) is not covered; the app runs on the user's own computer only, and NewsAPI's free plan
does not allow a public deployment.

safe_http_get has the same form as src/fulltext.py's http_get, so the full-text fetcher and its robots.txt check
take it in place of the plain request. A refused address raises UnsafeAddress; the fetcher records it as an error,
and robots.txt that cannot be read means the page is not fetched.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

MAX_REDIRECTS = 5
WEB_SCHEMES = ("http", "https")


class UnsafeAddress(Exception):
    pass


def address_problem(url: str, resolve=socket.getaddrinfo) -> str:
    """Why this address may not be fetched; "" when it is a public http or https address."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return "the address cannot be read"
    if parts.scheme.lower() not in WEB_SCHEMES:
        return "only http and https addresses are fetched"
    host = parts.hostname
    if not host:
        return "the address has no host"
    try:
        found = resolve(host, port or (443 if parts.scheme.lower() == "https" else 80), proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError):
        return "the host cannot be found"
    addresses = {info[4][0] for info in found}
    if not addresses:
        return "the host cannot be found"
    for text in addresses:
        try:
            address = ipaddress.ip_address(text.split("%")[0])
        except ValueError:
            return "the host gives an address that cannot be read"
        if getattr(address, "ipv4_mapped", None):
            address = address.ipv4_mapped
        if not address.is_global:
            return "the host is not a public internet address"
    return ""


def safe_http_get(url: str, timeout: int = 10, resolve=socket.getaddrinfo, get=None) -> tuple[int, str]:
    """(status, text) of a public page; every redirect step is checked before it is followed."""
    import requests
    from src.fulltext import USER_AGENT
    get = get or requests.get
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        problem = address_problem(current, resolve)
        if problem:
            raise UnsafeAddress(f"not fetched: {problem}")
        response = get(current, headers={"User-Agent": USER_AGENT}, timeout=timeout, allow_redirects=False)
        location = response.headers.get("Location") if response.status_code in (301, 302, 303, 307, 308) else None
        if not location:
            return response.status_code, response.text
        current = urljoin(current, location)
    raise UnsafeAddress(f"not fetched: more than {MAX_REDIRECTS} redirects")
