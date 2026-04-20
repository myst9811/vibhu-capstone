PHISHING_VOCABULARY: frozenset = frozenset({
    # Identity & credentials
    "bank account", "account number", "routing number", "social security", "ssn",
    "credit card", "debit card", "card number", "cvv", "pin", "password",
    "one time password", "otp", "authentication code", "verification code",
    # Urgency & threat
    "urgent", "immediately", "suspended", "locked", "blocked", "frozen",
    "arrested", "arrest", "warrant", "lawsuit", "legal action", "police",
    "irs", "federal", "deportation",
    # Money movement
    "wire transfer", "transfer funds", "send money", "gift card",
    "bitcoin", "cryptocurrency", "crypto", "cash app", "zelle",
    "western union", "money order", "refund", "prize", "won", "lottery",
    "inheritance", "unclaimed funds",
    # Scam scenarios
    "investment opportunity", "guaranteed return", "double your money",
    "tech support", "microsoft", "apple support", "remote access",
    "kidnapped", "hostage", "ransom",
    # High-signal single words
    "verify", "confirm", "validate", "click", "link", "suspicious",
    "compromise", "breach", "leaked",
})
