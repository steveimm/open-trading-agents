from __future__ import annotations

import gzip
import io
import json
import socket
import time
from datetime import date
from typing import Any
from unittest.mock import Mock

import pytest

from open_trading_agents.dataflows import web
from open_trading_agents.dataflows.config import run_config
from open_trading_agents.dataflows.tools import FinancialTools, ToolContext, create_server


def _public_dns(host: str, port: int, **kwargs: Any) -> list[tuple[Any, ...]]:
    """Resolve test public hosts without performing DNS requests."""
    address = host if host in ("127.0.0.1", "169.254.169.254", "::1", "10.0.0.1") else "93.184.216.34"
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://localhost/a",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/",
        "http://[::1]/",
        "https://user:secret@example.org/",
        "https://server.local/",
        "https://example.org/\nheader",
    ],
)
def test_fetch_rejects_local_destinations_schemes_and_credentials(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """A model-provided page URL must not read local files, services, or cloud metadata."""
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    request = Mock(side_effect=AssertionError("No connection may be opened"))
    monkeypatch.setattr(web, "_request", request)
    result = json.loads(web.fetch_webpage(url, "2024-03-15"))
    assert "DATA_UNAVAILABLE" in result["error"]
    request.assert_not_called()


def test_redirects_revalidate_addresses_and_never_send_to_private_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public source cannot redirect the page reader into a private service."""
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    request = Mock(return_value=web.WebResponse("https://example.org/", 302, {"location": "http://127.0.0.1/private"}, b""))
    monkeypatch.setattr(web, "_request", request)
    result = json.loads(web.fetch_webpage("https://example.org/", "2024-03-15"))
    assert "DATA_UNAVAILABLE" in result["error"] and request.call_count == 1


def test_redirect_cycles_and_excessive_hops_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop cyclic or indefinitely growing redirects rather than retrying without a bound."""
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    request = Mock(return_value=web.WebResponse("https://example.org/", 302, {"location": "/"}, b""))
    monkeypatch.setattr(web, "_request", request)
    assert "loop" in json.loads(web.fetch_webpage("https://example.org/", "2024-03-15"))["error"]
    request.reset_mock(side_effect=True)
    request.side_effect = [web.WebResponse(f"https://example.org/{index}", 302, {"location": f"/{index + 1}"}, b"") for index in range(4)]
    assert "redirect limit" in json.loads(web.fetch_webpage("https://example.org/0", "2024-03-15"))["error"]
    assert request.call_count == 4


def test_mixed_dns_answer_is_rejected_before_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """One private answer makes an otherwise public-looking hostname ineligible."""
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: _public_dns("example.org", 443) + _public_dns("127.0.0.1", 443))
    with pytest.raises(web.WebAccessError):
        web._public_target("https://example.org/")


def test_connection_uses_pinned_ip_and_original_tls_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid a second DNS lookup while preserving certificate-hostname validation."""
    target = web.PublicTarget("https://example.org/", "example.org", 443, ("93.184.216.34",), "/", True)
    raw_socket = Mock()
    connect = Mock(return_value=raw_socket)
    context = Mock()
    monkeypatch.setattr(socket, "create_connection", connect)
    monkeypatch.setattr(web.ssl, "create_default_context", Mock(return_value=context))
    connection = web._PinnedConnection(target, target.addresses[0], 3)
    connection.connect()
    connect.assert_called_once_with(("93.184.216.34", 443), 3)
    context.wrap_socket.assert_called_once_with(raw_socket, server_hostname="example.org")


def _response(body: bytes, headers: dict[str, str] | None = None) -> Mock:
    """Make a bounded-reader response with the requested content headers."""
    source = io.BytesIO(body)
    response = Mock(status=200)
    response.getheaders.return_value = list((headers or {}).items())
    response.read1.side_effect = source.read1
    response.read.side_effect = source.read
    return response


@pytest.mark.parametrize("kind", ["declared", "streamed", "gzip"])
def test_decoded_response_size_is_enforced(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """Reject huge responses including chunked bodies and compressed expansion."""
    body = b"x" * 1000
    headers = {"content-length": "1000"} if kind == "declared" else {}
    if kind == "gzip":
        body = gzip.compress(body)
        headers = {"content-encoding": "gzip"}
    connection = Mock()
    connection.getresponse.return_value = _response(body, headers)
    monkeypatch.setattr(web, "_PinnedConnection", Mock(return_value=connection))
    target = web.PublicTarget("https://example.org/", "example.org", 443, ("93.184.216.34",), "/", True)
    with pytest.raises(web.WebAccessError, match="limit"):
        web._request(target, time.monotonic() + 5, max_bytes=200)
    connection.close.assert_called_once()


def test_text_extraction_discards_executable_hidden_content_and_keeps_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep visible source evidence without executing scripts or following embedded resource URLs."""
    html = b"""<html><head><title>Quarterly &amp; annual results</title>
    <meta property="article:published_time" content="2024-03-10T10:00:00Z">
    <meta property="article:modified_time" content="2025-04-01T00:00:00Z">
    <script>fetch('http://127.0.0.1/secret'); SECRET_SCRIPT</script><style>SECRET_STYLE</style></head>
    <body><nav>SECRET_NAV</nav><p hidden>SECRET_HIDDEN</p><div style="display:none">SECRET_CSS</div>
    <article><h1>Financial results</h1><p>Revenue was 100.</p><a href="/release">Full release</a></article>
    <iframe src="http://127.0.0.1/">SECRET_FRAME</iframe><img src="https://tracker.invalid/pixel"></body></html>"""
    fetch = Mock(return_value=web.WebResponse("https://example.org/results", 200, {"content-type": "text/html"}, html))
    monkeypatch.setattr(web, "_fetch", fetch)
    result = json.loads(web.fetch_webpage("https://example.org/results", "2024-03-15"))
    assert result["title"] == "Quarterly & annual results"
    assert result["reported_publication_date"] == "2024-03-10T10:00:00Z"
    assert result["reported_modification_date"] == "2025-04-01T00:00:00Z"
    assert "not an archived" in result["availability_notice"]
    assert "Revenue was 100." in result["text"] and "SECRET" not in result["text"]
    assert result["links"] == [{"title": "Full release", "url": "https://example.org/release"}]
    fetch.assert_called_once()


def test_future_publication_is_withheld_and_text_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not expose a known future publication as evidence for an earlier analysis."""
    monkeypatch.setattr(
        web,
        "_fetch",
        Mock(
            return_value=web.WebResponse(
                "https://example.org/release",
                200,
                {"content-type": "text/html"},
                b'<meta property="article:published_time" content="2024-03-16"><p>Future outcome</p>',
            )
        ),
    )
    result = json.loads(web.fetch_webpage("https://example.org/release", "2024-03-15"))
    assert result["text"] == "" and "after the analysis cutoff" in result["availability_notice"]
    monkeypatch.setattr(
        web,
        "_fetch",
        Mock(
            return_value=web.WebResponse(
                "https://example.org/release",
                200,
                {"content-type": "text/plain"},
                b"a" * 1000,
            )
        ),
    )
    result = json.loads(web.fetch_webpage("https://example.org/release", date.today().isoformat(), max_chars=100))
    assert len(result["text"]) == 100 and result["truncated"]


def test_search_preserves_citations_filters_known_future_dates_and_uses_explicit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search results must retain source URLs and avoid claiming point-in-time coverage."""
    client = Mock()
    client.text.return_value = [
        {"title": "Known release", "href": "https://example.org/past", "body": "A useful snippet", "date": "2024-03-01"},
        {"title": "Future", "href": "https://example.org/future", "body": "Future outcome", "date": "2024-04-01"},
        {"title": "Local", "href": "http://127.0.0.1/private", "body": "private"},
    ]
    monkeypatch.setattr(web, "DDGS", Mock(return_value=client))
    with run_config({"web_search_backend": "yahoo"}):
        result = json.loads(web.search_web("company results", "2024-03-15", 3))
    assert [source["url"] for source in result["sources"]] == ["https://example.org/past"]
    assert "not from an archived" in result["availability_notice"]
    client.text.assert_called_once_with("company results", max_results=3, backend="yahoo")
    with run_config({"web_search_backend": "auto"}), pytest.raises(ValueError, match="explicit"):
        web.search_web("company results", "2024-03-15")


@pytest.mark.asyncio
async def test_mcp_web_tools_keep_analysis_date_out_of_model_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind source browsing to the workflow's date and allow only selected web tools."""
    server = create_server(ToolContext("NVDA", "2024-03-15"), ("search_web", "fetch_webpage"))
    for tool in await server.list_tools():
        assert "trade_date" not in tool.input_schema.get("properties", {})
    source = Mock(return_value="source")
    monkeypatch.setattr("open_trading_agents.dataflows.tools.fetch_webpage", source)
    FinancialTools(ToolContext("NVDA", "2024-03-15")).fetch_webpage("https://example.org/release")
    source.assert_called_once_with("https://example.org/release", "2024-03-15", 20000)
