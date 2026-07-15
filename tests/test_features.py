"""Unit tests for the feature construction (no model artifacts needed)."""
from features import STRUCT_ORDER, build_combined, clean_text, structural_features
from sklearn.feature_extraction.text import TfidfVectorizer


def test_clean_text_tokenizes_urls_and_emails():
    out = clean_text("Visit https://example.com or email me@test.com NOW!!!")
    assert "urltoken" in out
    assert "emailtoken" in out
    assert out == out.lower()
    assert "!" not in out


def test_structural_features_keys_and_values():
    feats = structural_features("Re: Payment", "Please pay $50 at http://x.com now!")
    assert len(STRUCT_ORDER) == 18
    assert set(feats.keys()) == set(STRUCT_ORDER)
    assert feats["has_url"] == 1
    assert feats["num_urls"] == 1
    assert feats["has_money_symbol"] == 1
    assert feats["num_exclaim"] == 1
    assert feats["subject_is_reply"] == 1


def test_structural_features_clean_email():
    feats = structural_features("Lunch", "See you at noon, thanks.")
    assert feats["has_url"] == 0
    assert feats["has_money_symbol"] == 0
    assert feats["subject_is_reply"] == 0


def test_build_combined_shape_matches_features():
    vec = TfidfVectorizer().fit(["hello world money", "pay now urltoken document"])
    X, clean, feats = build_combined("Re: hi", "pay money at http://x.com", vec)
    assert X.shape[0] == 1
    assert X.shape[1] == len(STRUCT_ORDER) + len(vec.get_feature_names_out())
    assert isinstance(feats, dict)
