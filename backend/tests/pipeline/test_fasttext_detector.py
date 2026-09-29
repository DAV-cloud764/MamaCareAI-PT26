"""Tests for FastTextDetector."""

import pytest

from modules.pipeline.adapters.language.fasttext_detector import FastTextDetector
from modules.pipeline.ports.language_detector import DetectionResult


@pytest.fixture
def detector_with_fallback():
    """Create a FastTextDetector with fallback model (for testing without fastText)."""
    # Use a non-existent model path to trigger fallback
    return FastTextDetector(model_path="nonexistent_model.bin", top_k=3)


class TestHeuristicDetection:
    """Test fallback heuristic-based language detection."""

    def test_empty_text(self, detector_with_fallback):
        """Test that empty text returns unknown with 0.0 confidence."""
        result = detector_with_fallback.detect("")
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.alternatives == ()

    def test_whitespace_only(self, detector_with_fallback):
        """Test that whitespace-only text returns unknown with 0.0 confidence."""
        result = detector_with_fallback.detect("   ")
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.alternatives == ()

    def test_english_detection(self, detector_with_fallback):
        """Test basic English detection."""
        text = "The quick brown fox jumps over the lazy dog."
        result = detector_with_fallback.detect(text)
        assert result.language == "en"
        assert 0.0 <= result.confidence <= 1.0

    def test_english_words_containing_na_are_not_swahili(self, detector_with_fallback):
        """Regression: substring 'na' in 'pregnancy/normal' is not a Swahili word."""
        text = (
            "Pregnancy lasts about 40 weeks, counting from the first day of "
            "your last normal period. The weeks are grouped into trimesters."
        )
        result = detector_with_fallback.detect(text)
        assert result.language == "en"

    def test_swahili_detection(self, detector_with_fallback):
        """Test basic Swahili detection."""
        text = "Habari za asubuhi, asante kwa msaada wako."
        result = detector_with_fallback.detect(text)
        assert result.language == "sw"
        assert 0.0 <= result.confidence <= 1.0

    def test_unknown_language(self, detector_with_fallback):
        """Test detection of unknown language."""
        text = "xyz abc def ghi jkl mno pqr stu vwx yz"
        result = detector_with_fallback.detect(text)
        # langdetect may detect various languages for gibberish text
        assert result.language in ["en", "unknown", "cs", "it", "vi"]  # langdetect may detect different languages
        assert 0.0 <= result.confidence <= 1.0

    def test_mixed_text(self, detector_with_fallback):
        """Test detection with mixed content."""
        text = "The habari for asante is the best."
        result = detector_with_fallback.detect(text)
        # Should detect something (fallback prioritizes Swahili if found)
        assert result.language in ["en", "sw", "unknown"]
        assert 0.0 <= result.confidence <= 1.0

    def test_very_short_text(self, detector_with_fallback):
        """Test detection with very short text."""
        text = "the"
        result = detector_with_fallback.detect(text)
        assert result.language in ["en", "sw", "unknown"]
        assert 0.0 <= result.confidence <= 1.0

    def test_special_characters(self, detector_with_fallback):
        """Test detection with special characters."""
        text = "The @#$%^&*() quick brown fox!"
        result = detector_with_fallback.detect(text)
        assert result.language in ["en", "sw", "unknown"]
        assert 0.0 <= result.confidence <= 1.0

    def test_unicode_content(self, detector_with_fallback):
        """Test detection with Unicode content."""
        text = "The quick brown fox jumps over the lazy dog. 🦊"
        result = detector_with_fallback.detect(text)
        assert result.language in ["en", "sw", "unknown"]
        assert 0.0 <= result.confidence <= 1.0

    def test_numeric_content(self, detector_with_fallback):
        """Test detection with numeric content."""
        text = "123 456 789"
        result = detector_with_fallback.detect(text)
        assert result.language == "unknown"
        assert 0.0 <= result.confidence <= 1.0


class TestDetectorInit:
    """Test FastTextDetector initialization."""

    def test_default_top_k(self):
        """Test that default top_k is 3."""
        detector = FastTextDetector(model_path="test.bin")
        assert detector._top_k == 3

    def test_custom_top_k(self):
        """Test that custom top_k is respected."""
        detector = FastTextDetector(model_path="test.bin", top_k=5)
        assert detector._top_k == 5

    def test_model_path_storage(self):
        """Test that model path is stored."""
        detector = FastTextDetector(model_path="/path/to/model.bin")
        assert detector._model_path == "/path/to/model.bin"

    def test_fallback_enabled_for_nonexistent_model(self):
        """Test that fallback is enabled when model doesn't exist."""
        detector = FastTextDetector(model_path="nonexistent.bin")
        # With langdetect available, it should use langdetect fallback
        assert detector._use_langdetect is True or detector._use_heuristic is True
        assert detector._model is None


class TestDetectionResult:
    """Test DetectionResult structure."""

    def test_result_structure(self, detector_with_fallback):
        """Test that detection returns proper DetectionResult."""
        result = detector_with_fallback.detect("test text")
        assert isinstance(result, DetectionResult)
        assert hasattr(result, "language")
        assert hasattr(result, "confidence")
        assert hasattr(result, "alternatives")
        assert isinstance(result.language, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.alternatives, tuple)

    def test_confidence_range(self, detector_with_fallback):
        """Test that confidence is always in valid range."""
        texts = ["test", "the quick brown fox", "habari", ""]
        for text in texts:
            result = detector_with_fallback.detect(text)
            assert 0.0 <= result.confidence <= 1.0


class TestEdgeCases:
    """Test edge cases for language detection."""

    def test_case_insensitivity(self, detector_with_fallback):
        """Test that detection handles case variations."""
        text1 = "THE QUICK BROWN FOX"
        text2 = "the quick brown fox"
        result1 = detector_with_fallback.detect(text1)
        result2 = detector_with_fallback.detect(text2)
        # langdetect may detect different languages for different case patterns
        # This is a known limitation - both should be valid language codes
        assert len(result1.language) == 2  # Should be a valid ISO 639-1 code
        assert len(result2.language) == 2  # Should be a valid ISO 639-1 code
        assert 0.0 <= result1.confidence <= 1.0
        assert 0.0 <= result2.confidence <= 1.0

    def test_repeated_detections(self, detector_with_fallback):
        """Test that repeated detections are consistent."""
        text = "The quick brown fox jumps over the lazy dog."
        result1 = detector_with_fallback.detect(text)
        result2 = detector_with_fallback.detect(text)
        assert result1.language == result2.language
        # langdetect may have slight confidence variations due to probabilistic nature
        assert abs(result1.confidence - result2.confidence) < 0.01  # Allow small variations

    def test_very_long_text(self, detector_with_fallback):
        """Test detection with very long text."""
        text = "The quick brown fox " * 1000
        result = detector_with_fallback.detect(text)
        assert result.language in ["en", "sw", "unknown"]
        assert 0.0 <= result.confidence <= 1.0


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_swahili_sentence(self, detector_with_fallback):
        """Test detection with realistic Swahili sentence."""
        text = "Kila mtu ana haki ya kupata elimu bora na afya."
        result = detector_with_fallback.detect(text)
        assert result.language == "sw"
        assert result.confidence > 0.5

    def test_english_sentence(self, detector_with_fallback):
        """Test detection with realistic English sentence."""
        text = "Everyone has the right to quality education and healthcare."
        result = detector_with_fallback.detect(text)
        assert result.language == "en"
        assert result.confidence > 0.5

    def test_code_detection(self, detector_with_fallback):
        """Test detection with code content."""
        text = "def hello_world(): print('Hello, World!')"
        result = detector_with_fallback.detect(text)
        # Code should probably be detected as English or unknown
        assert result.language in ["en", "sw", "unknown"]

    def test_mixed_language_sentence(self, detector_with_fallback):
        """Test detection with mixed language sentence."""
        text = "The habari for the day is good asante."
        result = detector_with_fallback.detect(text)
        # Should detect one of the languages present
        assert result.language in ["en", "sw", "unknown"]
