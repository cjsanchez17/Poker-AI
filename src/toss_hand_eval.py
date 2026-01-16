import itertools
from typing import Iterable, List

from treys import Card, Evaluator


def to_treys_cards(cards: Iterable[str]) -> List[int]:
    return [Card.new(card) for card in cards]


def best_hand_rank(three_hole: Iterable[str], six_board: Iterable[str]) -> int:
    evaluator = Evaluator()
    best_score = 10**9
    hole_cards = list(three_hole)
    board_cards = list(six_board)
    for hole2 in itertools.combinations(hole_cards, 2):
        treys_hole = to_treys_cards(hole2)
        for board5 in itertools.combinations(board_cards, 5):
            score = evaluator.evaluate(to_treys_cards(board5), treys_hole)
            if score < best_score:
                best_score = score
    return best_score
