from typing import Iterable, List

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
SUITS = ["c", "d", "h", "s"]


def index_to_card_str(idx: int) -> str:
    if idx < 0 or idx >= 52:
        raise ValueError(f"Card index out of range: {idx}")
    rank = RANKS[idx // 4]
    suit = SUITS[idx % 4]
    return f"{rank}{suit}"


def cards_to_str(cards: Iterable[int]) -> List[str]:
    return [index_to_card_str(card) for card in cards]


def hand_to_str(cards: Iterable[int]) -> str:
    return "".join(cards_to_str(cards))
