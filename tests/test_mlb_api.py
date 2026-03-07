"""Tests for MLB API client — rate limiting, caching, error handling.

These tests use mocking to avoid hitting the real API.
"""

import time
from unittest.mock import patch, MagicMock

import pytest
import requests

from src.data.mlb_api import (
    _get,
    _throttle,
    _cache_get,
    _cache_set,
    clear_cache,
    MLBApiError,
    MIN_REQUEST_GAP,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear the response cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestCaching:
    def test_cache_set_and_get(self):
        _cache_set("test-key", {"foo": "bar"}, ttl=60)
        result = _cache_get("test-key")
        assert result == {"foo": "bar"}

    def test_cache_miss_returns_none(self):
        assert _cache_get("nonexistent") is None

    def test_cache_expires(self):
        _cache_set("expire-me", {"x": 1}, ttl=0.01)
        time.sleep(0.02)
        assert _cache_get("expire-me") is None

    def test_clear_cache(self):
        _cache_set("key1", {"a": 1})
        _cache_set("key2", {"b": 2})
        clear_cache()
        assert _cache_get("key1") is None
        assert _cache_get("key2") is None


class TestGet:
    @patch("src.data.mlb_api.requests.get")
    def test_successful_request(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": "test"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _get("https://example.com/api/test", cache_ttl=0)
        assert result == {"data": "test"}
        mock_get.assert_called_once()

    @patch("src.data.mlb_api.requests.get")
    def test_caches_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"cached": True}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # First call hits API
        _get("https://example.com/api/cached", cache_ttl=300)
        # Second call should use cache
        result = _get("https://example.com/api/cached", cache_ttl=300)

        assert result == {"cached": True}
        assert mock_get.call_count == 1  # Only one real request

    @patch("src.data.mlb_api.requests.get")
    def test_non_429_error_raises_immediately(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=mock_resp
        )
        mock_get.return_value = mock_resp

        with pytest.raises(MLBApiError, match="MLB API request failed"):
            _get("https://example.com/api/broken", cache_ttl=0)

    @patch("src.data.mlb_api.requests.get")
    def test_relative_url_prepends_base(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _get("/standings", cache_ttl=0)
        called_url = mock_get.call_args[0][0]
        assert called_url.startswith("https://statsapi.mlb.com/api/v1")
        assert called_url.endswith("/standings")


class TestErrorMessages:
    def test_mlb_api_error_is_exception(self):
        err = MLBApiError("something broke")
        assert str(err) == "something broke"
        assert isinstance(err, Exception)
