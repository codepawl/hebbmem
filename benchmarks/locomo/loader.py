"""Parse LoCoMo JSON into structured data for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.locomo.download import load_locomo


@dataclass
class Turn:
    speaker: str
    text: str
    turn_id: str
    session_num: int


@dataclass
class QAPair:
    question: str
    answer: str
    category: str
    evidence_turn_ids: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    conversation_id: str
    turns: list[Turn]
    qa_pairs: list[QAPair]
    num_sessions: int


CATEGORY_MAP = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    "1": "single_hop",
    "2": "multi_hop",
    "3": "temporal",
    "4": "open_domain",
}


def parse_conversations(
    limit: int | None = None,
) -> list[Conversation]:
    """Parse LoCoMo JSON into Conversation objects."""
    raw = load_locomo()
    conversations = []

    for entry in raw:
        conv_id = str(entry.get("sample_id", ""))
        conv_data = entry.get("conversation", {})

        # Find session keys: session_1, session_2, etc.
        session_keys = sorted(
            k
            for k in conv_data
            if k.startswith("session_") and not k.endswith("_date_time")
        )

        turns: list[Turn] = []
        for sess_idx, sess_key in enumerate(session_keys):
            session = conv_data[sess_key]
            if not isinstance(session, list):
                continue
            for turn_data in session:
                turn_id = str(turn_data.get("dia_id", ""))
                speaker = str(turn_data.get("speaker", ""))
                text = str(turn_data.get("text", ""))
                if text:
                    turns.append(
                        Turn(
                            speaker=speaker,
                            text=text,
                            turn_id=turn_id,
                            session_num=sess_idx,
                        )
                    )

        # Parse QA pairs
        qa_pairs: list[QAPair] = []
        qa_raw = entry.get("qa", [])
        if isinstance(qa_raw, list):
            for qa in qa_raw:
                cat = qa.get("category", "")
                # Skip adversarial/unanswerable (category 5)
                if cat == 5 or cat == "5":
                    continue
                cat_name = CATEGORY_MAP.get(cat, str(cat))

                question = str(qa.get("question", ""))
                answer = str(qa.get("answer", ""))
                evidence = qa.get("evidence", [])
                ev_ids = (
                    [str(e) for e in evidence] if isinstance(evidence, list) else []
                )

                if question:
                    qa_pairs.append(
                        QAPair(
                            question=question,
                            answer=answer,
                            category=cat_name,
                            evidence_turn_ids=ev_ids,
                        )
                    )

        if turns and qa_pairs:
            conversations.append(
                Conversation(
                    conversation_id=conv_id,
                    turns=turns,
                    qa_pairs=qa_pairs,
                    num_sessions=len(session_keys),
                )
            )

        if limit and len(conversations) >= limit:
            break

    return conversations
