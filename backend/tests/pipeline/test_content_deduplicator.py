"""Tests for ContentDeduplicator."""

import pytest

from modules.pipeline.adapters.storage.content_deduplicator import ContentDeduplicator
from modules.pipeline.domain.models import Resource
from tests.pipeline.fakes import FakeResourceRepository


@pytest.fixture
def fake_repo():
    """Create a fake resource repository."""
    return FakeResourceRepository()


@pytest.fixture
def deduplicator(fake_repo):
    """Create a ContentDeduplicator with fake repository."""
    return ContentDeduplicator(resources=fake_repo)


class TestURLNormalization:
    """Test URL normalization in compute_hash."""

    def test_lowercase_scheme_and_host(self, deduplicator):
        """Test that scheme and host are lowercased."""
        url1 = "HTTPS://WWW.EXAMPLE.COM/path"
        url2 = "https://www.example.com/path"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url1, content=content)
        hash2 = deduplicator.compute_hash(source_url=url2, content=content)

        assert hash1 == hash2

    def test_remove_fragment(self, deduplicator):
        """Test that fragments are removed."""
        url1 = "https://example.com/path#section"
        url2 = "https://example.com/path"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url1, content=content)
        hash2 = deduplicator.compute_hash(source_url=url2, content=content)

        assert hash1 == hash2

    def test_remove_tracking_params(self, deduplicator):
        """Test that tracking parameters are removed."""
        url1 = "https://example.com/path?utm_source=google&utm_medium=email&param=value"
        url2 = "https://example.com/path?param=value"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url1, content=content)
        hash2 = deduplicator.compute_hash(source_url=url2, content=content)

        assert hash1 == hash2

    def test_sort_query_params(self, deduplicator):
        """Test that query parameters are sorted."""
        url1 = "https://example.com/path?a=1&b=2"
        url2 = "https://example.com/path?b=2&a=1"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url1, content=content)
        hash2 = deduplicator.compute_hash(source_url=url2, content=content)

        assert hash1 == hash2

    def test_strip_trailing_slash(self, deduplicator):
        """Test that trailing slash is stripped."""
        url1 = "https://example.com/path/"
        url2 = "https://example.com/path"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url1, content=content)
        hash2 = deduplicator.compute_hash(source_url=url2, content=content)

        assert hash1 == hash2

    def test_drop_www_prefix(self, deduplicator):
        """Test that www. prefix is dropped."""
        url1 = "https://www.example.com/path"
        url2 = "https://example.com/path"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url1, content=content)
        hash2 = deduplicator.compute_hash(source_url=url2, content=content)

        assert hash1 == hash2

    def test_all_tracking_params(self, deduplicator):
        """Test all tracking parameters are removed."""
        tracking_params = [
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "ref", "session", "sessionid"
        ]

        for param in tracking_params:
            url1 = f"https://example.com/path?{param}=value&keep=this"
            url2 = "https://example.com/path?keep=this"
            content = b"test content"

            hash1 = deduplicator.compute_hash(source_url=url1, content=content)
            hash2 = deduplicator.compute_hash(source_url=url2, content=content)

            assert hash1 == hash2, f"Failed for parameter: {param}"


class TestContentNormalization:
    """Test content normalization in compute_hash."""

    def test_bytes_decode_with_errors(self, deduplicator):
        """Test that bytes are decoded with error handling."""
        content_bytes = b"test content"
        content_str = "test content"
        url = "https://example.com/path"

        hash1 = deduplicator.compute_hash(source_url=url, content=content_bytes)
        hash2 = deduplicator.compute_hash(source_url=url, content=content_str)

        assert hash1 == hash2

    def test_nfc_normalization(self, deduplicator):
        """Test NFC normalization."""
        # Using a character that can be represented in multiple ways
        content1 = "café"  # NFC form
        content2 = "cafe\u0301"  # NFD form (e + combining acute)
        url = "https://example.com/path"

        hash1 = deduplicator.compute_hash(source_url=url, content=content1)
        hash2 = deduplicator.compute_hash(source_url=url, content=content2)

        assert hash1 == hash2

    def test_lowercase(self, deduplicator):
        """Test lowercasing."""
        content1 = "TEST CONTENT"
        content2 = "test content"
        url = "https://example.com/path"

        hash1 = deduplicator.compute_hash(source_url=url, content=content1)
        hash2 = deduplicator.compute_hash(source_url=url, content=content2)

        assert hash1 == hash2

    def test_collapse_whitespace(self, deduplicator):
        """Test whitespace collapse."""
        content1 = "test  content"  # Multiple spaces
        content2 = "test content"  # Single space
        url = "https://example.com/path"

        hash1 = deduplicator.compute_hash(source_url=url, content=content1)
        hash2 = deduplicator.compute_hash(source_url=url, content=content2)

        assert hash1 == hash2

    def test_strip_whitespace(self, deduplicator):
        """Test whitespace stripping."""
        content1 = " test content "
        content2 = "test content"
        url = "https://example.com/path"

        hash1 = deduplicator.compute_hash(source_url=url, content=content1)
        hash2 = deduplicator.compute_hash(source_url=url, content=content2)

        assert hash1 == hash2

    def test_newlines_and_tabs(self, deduplicator):
        """Test newlines and tabs are collapsed."""
        content1 = "test\ncontent\tmore"
        content2 = "test content more"
        url = "https://example.com/path"

        hash1 = deduplicator.compute_hash(source_url=url, content=content1)
        hash2 = deduplicator.compute_hash(source_url=url, content=content2)

        assert hash1 == hash2


class TestDeterminism:
    """Test that hashing is deterministic."""

    def test_same_input_same_hash(self, deduplicator):
        """Test that same input produces same hash."""
        url = "https://example.com/path"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url, content=content)
        hash2 = deduplicator.compute_hash(source_url=url, content=content)

        assert hash1 == hash2

    def test_different_inputs_different_hashes(self, deduplicator):
        """Test that different inputs produce different hashes."""
        url = "https://example.com/path"

        hash1 = deduplicator.compute_hash(source_url=url, content=b"content1")
        hash2 = deduplicator.compute_hash(source_url=url, content=b"content2")

        assert hash1 != hash2

    def test_url_difference(self, deduplicator):
        """Test that different URLs produce different hashes."""
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url="https://example.com/path1", content=content)
        hash2 = deduplicator.compute_hash(source_url="https://example.com/path2", content=content)

        assert hash1 != hash2


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_content_url_only_hash(self, deduplicator):
        """Test that empty content results in URL-only hash."""
        url = "https://example.com/path"
        content = ""

        hash1 = deduplicator.compute_hash(source_url=url, content=content)
        hash2 = deduplicator.compute_hash(source_url=url, content=None)

        assert hash1 == hash2

    def test_empty_bytes_content(self, deduplicator):
        """Test that empty bytes content is handled."""
        url = "https://example.com/path"
        content = b""

        hash1 = deduplicator.compute_hash(source_url=url, content=content)
        hash2 = deduplicator.compute_hash(source_url=url, content=None)

        assert hash1 == hash2

    def test_very_long_url(self, deduplicator):
        """Test very long URL."""
        long_path = "/path/" + "a" * 1000
        url = f"https://example.com{long_path}"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url, content=content)
        hash2 = deduplicator.compute_hash(source_url=url, content=content)

        assert hash1 == hash2

    def test_special_characters_in_url(self, deduplicator):
        """Test special characters in URL."""
        url = "https://example.com/path?param=hello%20world"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url, content=content)
        hash2 = deduplicator.compute_hash(source_url=url, content=content)

        assert hash1 == hash2

    def test_unicode_content(self, deduplicator):
        """Test Unicode content."""
        url = "https://example.com/path"
        content = "Hello 世界 مرحبا 你好"

        hash1 = deduplicator.compute_hash(source_url=url, content=content)
        hash2 = deduplicator.compute_hash(source_url=url, content=content)

        assert hash1 == hash2

    def test_case_sensitive_path(self, deduplicator):
        """Test that paths are case-sensitive."""
        url1 = "https://example.com/Path"
        url2 = "https://example.com/path"
        content = b"test content"

        hash1 = deduplicator.compute_hash(source_url=url1, content=content)
        hash2 = deduplicator.compute_hash(source_url=url2, content=content)

        assert hash1 != hash2  # Paths should remain case-sensitive


class TestIsDuplicate:
    """Test is_duplicate method."""

    def test_new_hash_returns_false(self, deduplicator, fake_repo):
        """Test that new hash returns False."""
        new_hash = "a" * 64  # SHA256 hash length

        assert not deduplicator.is_duplicate(new_hash)

    def test_existing_hash_returns_true(self, deduplicator, fake_repo):
        """Test that existing hash returns True."""
        # Create a resource with a specific content hash
        from modules.pipeline.domain.enums import ResourceStatus, SourceType
        resource = Resource(
            resource_id="test-id",
            source_type=SourceType.WEB,
            source_url="https://example.com/test",
            status=ResourceStatus.SUBMITTED,
            raw_object_key="raw/test",
            content_hash="a" * 64,
        )
        fake_repo.save(resource)

        assert deduplicator.is_duplicate("a" * 64)

    def test_repository_interaction(self, deduplicator, fake_repo):
        """Test that is_duplicate calls repository correctly."""
        from modules.pipeline.domain.enums import ResourceStatus, SourceType
        # Create multiple resources
        for i in range(3):
            resource = Resource(
                resource_id=f"test-id-{i}",
                source_type=SourceType.WEB,
                source_url=f"https://example.com/test{i}",
                status=ResourceStatus.SUBMITTED,
                raw_object_key=f"raw/test{i}",
                content_hash=f"{'a' * 63}{i}",
            )
            fake_repo.save(resource)

        assert deduplicator.is_duplicate("a" * 63 + "0")
        assert deduplicator.is_duplicate("a" * 63 + "1")
        assert deduplicator.is_duplicate("a" * 63 + "2")
        assert not deduplicator.is_duplicate("b" * 64)


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_real_world_url_normalization(self, deduplicator):
        """Test with realistic URLs."""
        url1 = "https://www.example.com/path/?utm_source=google&utm_medium=email#section"
        url2 = "https://example.com/path?utm_source=facebook&utm_medium=social"
        content = b"Same content"

        hash1 = deduplicator.compute_hash(source_url=url1, content=content)
        hash2 = deduplicator.compute_hash(source_url=url2, content=content)

        assert hash1 == hash2

    def test_duplicate_detection_workflow(self, deduplicator, fake_repo):
        """Test complete duplicate detection workflow."""
        from modules.pipeline.domain.enums import ResourceStatus, SourceType
        url = "https://example.com/document"
        content = b"Important document content"

        # First resource
        resource1 = Resource(
            resource_id="resource-1",
            source_type=SourceType.WEB,
            source_url=url,
            status=ResourceStatus.SUBMITTED,
            raw_object_key="raw/doc1",
            content_hash=deduplicator.compute_hash(source_url=url, content=content),
        )
        fake_repo.save(resource1)

        # Check if duplicate
        content_hash = deduplicator.compute_hash(source_url=url, content=content)
        assert deduplicator.is_duplicate(content_hash)

        # Different content should not be duplicate
        different_content = b"Different document content"
        different_hash = deduplicator.compute_hash(source_url=url, content=different_content)
        assert not deduplicator.is_duplicate(different_hash)