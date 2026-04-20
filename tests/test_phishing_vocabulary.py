from src.phishing_vocabulary import PHISHING_VOCABULARY


def test_vocabulary_is_frozenset():
    assert isinstance(PHISHING_VOCABULARY, frozenset)


def test_canonical_phrases_present():
    required = [
        "bank account", "social security", "wire transfer", "gift card",
        "verify", "urgent", "arrested", "warrant", "otp", "bitcoin",
        "password", "refund", "lottery", "ransom",
    ]
    for phrase in required:
        assert phrase in PHISHING_VOCABULARY, f"Missing: {phrase!r}"


def test_all_lowercase_no_whitespace():
    for phrase in PHISHING_VOCABULARY:
        assert phrase == phrase.lower(), f"Not lowercase: {phrase!r}"
        assert phrase == phrase.strip(), f"Has surrounding whitespace: {phrase!r}"
        assert "  " not in phrase, f"Double space: {phrase!r}"
