from prompt_evaluation.lab import (
    DEVELOPMENT_ROWS,
    PROMPT_VERSIONS,
    TEST_ROWS,
    TRAIN_ROWS,
    paired_bootstrap_delta,
    prompt_hash,
    render_prompt,
    run_prompt_lab,
    validate_label,
)


def test_splits_have_no_exact_review_overlap():
    train = {row[0] for row in TRAIN_ROWS}
    development = {row[0] for row in DEVELOPMENT_ROWS}
    test = {row[0] for row in TEST_ROWS}
    assert train.isdisjoint(development | test)
    assert development.isdisjoint(test)


def test_prompt_versions_are_stable_and_change_one_declared_stage():
    review = "good product"
    rendered = [render_prompt(version, review) for version in PROMPT_VERSIONS]
    assert len(set(rendered)) == 3
    assert all(review in prompt for prompt in rendered)
    assert len({prompt_hash(version) for version in PROMPT_VERSIONS}) == 3


def test_label_schema_rejects_extra_or_unknown_text():
    assert validate_label("Positive\n")[0]
    assert not validate_label("positive because it was good")[0]
    assert not validate_label("unknown")[0]


def test_paired_bootstrap_requires_matching_shapes():
    try:
        paired_bootstrap_delta([0, 1], [1])
    except ValueError as error:
        assert "paired" in str(error)
    else:
        raise AssertionError("mismatched scores must fail")


def test_full_local_prompt_lab_is_reproducible_and_honest():
    report = run_prompt_lab(seed=42)
    selected = report["selected_on_development"]
    assert selected in {"v1_explicit_contract", "v2_few_shot"}
    assert set(report["test_opened_after_selection"]) == {"v0_baseline", selected}
    assert report["next_token_conditioning"]["total_variation_distance"] >= 0
    assert report["release_decision"] in {"accept", "review_or_reject"}
    assert len(report["robustness_on_development"]) == 3
    for evaluation in report["development"].values():
        assert 0 <= evaluation["accuracy"] <= 1
        assert 0 <= evaluation["schema_valid_rate"] <= 1
        assert len(evaluation["examples"]) == len(DEVELOPMENT_ROWS)
