"""Unit tests for the sender heuristics and serving constants.

These import the inference module without loading any model, since the model is
only loaded when a PhishGuardModel is instantiated.
"""
from inference import POC_NOTE, RECOMMENDED_ACTION, sender_notes


def test_sender_notes_empty_is_silent():
    assert sender_notes("") == []
    assert sender_notes("   ") == []


def test_sender_notes_flags_free_email_provider():
    notes = sender_notes("john@gmail.com")
    assert any("free email" in n.lower() for n in notes)


def test_sender_notes_flags_lookalike_domain():
    notes = sender_notes("security@paypa1-verify.com")
    assert any("unusual" in n.lower() for n in notes)


def test_sender_notes_flags_malformed_address():
    notes = sender_notes("not-an-email")
    assert any("valid email" in n.lower() for n in notes)


def test_recommended_actions_cover_all_classes():
    assert set(RECOMMENDED_ACTION) == {"safe", "scam", "malware"}
    for message in RECOMMENDED_ACTION.values():
        assert isinstance(message, str) and message


def test_poc_note_present():
    assert POC_NOTE
