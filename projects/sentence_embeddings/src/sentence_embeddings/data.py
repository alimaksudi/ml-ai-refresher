"""Small, human-readable support-search dataset with explicit split boundaries."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntentExamples:
    intent: str
    train_queries: tuple[str, ...]
    train_documents: tuple[str, ...]
    evaluation_query: str
    evaluation_document: str
    hard_negative_intent: str


EXAMPLES = (
    IntentExamples(
        "reset_password",
        ("I forgot my password", "Help me create a new login password", "My password no longer works"),
        ("Reset a forgotten password from account security", "Use password recovery to choose a new password"),
        "Help me reset a forgotten login password.",
        "Choose a new password with account recovery.",
        "unlock_account",
    ),
    IntentExamples(
        "unlock_account",
        ("My account is locked", "Too many login attempts blocked me", "I cannot enter my locked profile"),
        ("Unlock an account after failed sign in attempts", "Wait or verify your identity to unlock the profile"),
        "Failed login attempts locked my account.",
        "Verify your identity to unlock a blocked profile.",
        "reset_password",
    ),
    IntentExamples(
        "refund_purchase",
        ("I want a refund for my order", "Return my purchase and send the money back", "Can I reverse this purchase?"),
        ("Request a refund for an eligible purchase", "Return an order before its refund deadline"),
        "Can I return an order for a refund?",
        "Request money back for an eligible purchase.",
        "duplicate_charge",
    ),
    IntentExamples(
        "duplicate_charge",
        ("I was charged twice", "The same payment appears two times", "There is a duplicate card charge"),
        ("Report a duplicated payment on your statement", "Dispute a transaction that was charged more than once"),
        "Why was the same card payment charged twice?",
        "Report a card charge that appears two times.",
        "refund_purchase",
    ),
    IntentExamples(
        "track_delivery",
        ("Where is my package?", "Track the delivery of my order", "When will my parcel arrive?"),
        ("Use the tracking number to follow a shipment", "Check current package location and delivery date"),
        "How can I track where my parcel is?",
        "Follow a package with its tracking number and delivery date.",
        "change_address",
    ),
    IntentExamples(
        "change_address",
        ("Change where my order will be delivered", "I entered the wrong shipping address", "Update my delivery location"),
        ("Edit the shipping address before dispatch", "Change an order destination while it is still processing"),
        "Update the wrong delivery address before dispatch.",
        "Edit an order destination while it is still processing.",
        "track_delivery",
    ),
    IntentExamples(
        "cancel_subscription",
        ("Cancel my subscription", "Stop my recurring membership", "I do not want the plan to renew"),
        ("Turn off renewal to cancel a subscription", "End a recurring membership from plan settings"),
        "Stop my recurring plan from renewing.",
        "Turn off renewal in settings to end a membership.",
        "download_invoice",
    ),
    IntentExamples(
        "download_invoice",
        ("I need an invoice", "Download my billing receipt", "Where is the receipt for my payment?"),
        ("Download an invoice from billing history", "Find payment receipts in the billing page"),
        "Where can I download the receipt for my payment?",
        "Find an invoice for a payment in billing history.",
        "cancel_subscription",
    ),
)


def all_training_texts() -> list[str]:
    return [text for row in EXAMPLES for text in (*row.train_queries, *row.train_documents)]


def evaluation_queries() -> list[str]:
    return [row.evaluation_query for row in EXAMPLES]


def evaluation_documents() -> list[str]:
    return [row.evaluation_document for row in EXAMPLES]


def assert_no_text_leakage() -> None:
    train = {text.casefold().strip() for text in all_training_texts()}
    evaluation = {
        text.casefold().strip()
        for row in EXAMPLES
        for text in (row.evaluation_query, row.evaluation_document)
    }
    overlap = train & evaluation
    if overlap:
        raise ValueError(f"train/evaluation text overlap: {sorted(overlap)}")
