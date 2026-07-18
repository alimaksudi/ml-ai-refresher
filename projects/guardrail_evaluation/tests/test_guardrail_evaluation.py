from guardrail_evaluation.lab import (
    OUTPUT_CASES,
    evaluate_output,
    policy_guard,
    redact_pii,
    run_guardrail_lab,
)


def test_authorized_pii_is_redacted_not_blindly_blocked():
    text, kinds = redact_pii("Email alice@example.com about SSN 123-45-6789.")
    assert "alice@example.com" not in text
    assert "123-45-6789" not in text
    assert set(kinds) == {"EMAIL_REDACTED", "SSN_REDACTED"}
    decision = policy_guard("Email the receipt to alice@example.com.")
    assert decision["action"] == "REDACT"


def test_benign_security_language_is_not_blocked_by_one_keyword():
    decision = policy_guard(
        "I am testing how an attack string is parsed by my own code."
    )
    assert decision["action"] == "ALLOW"


def test_support_correctness_and_citation_are_independent():
    measured = {
        case["case_id"]: evaluate_output(case["output"]) for case in OUTPUT_CASES
    }
    assert measured["O03"] == {
        "schema": True,
        "citation": True,
        "support": True,
        "correct": False,
    }
    assert measured["O04"] == {
        "schema": True,
        "citation": False,
        "support": False,
        "correct": True,
    }


def test_invalid_schema_fails_every_downstream_contract():
    assert evaluate_output({"answer": "text", "claims": "P1"}) == {
        "schema": False,
        "citation": False,
        "support": False,
        "correct": False,
    }


def test_full_report_exposes_escape_and_rejects_release():
    report = run_guardrail_lab()
    candidate = report["input_guard"]["policy_candidate"]
    baseline = report["input_guard"]["keyword_baseline"]
    assert candidate["exact_action_accuracy"] > baseline["exact_action_accuracy"]
    assert report["known_escape"]["predicted_action"] == "ALLOW"
    assert candidate["critical_escape_rate"] > 0
    assert report["output_contracts"]["contract_test_accuracy"] == 1
    assert report["release_decision"] == "review_or_reject"
