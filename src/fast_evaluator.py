"""
This is a fast evaluator used for training. It works with string representation of
cards. However, it cannot tell you if you won with a pair, three of a kind, etc.
"""

import random
from itertools import combinations
from phevaluator import evaluate_cards
import numpy as np
import treys

def phEvaluatorSetup(n):
    """
    Sets up n scenarios using the phevaluator library.
    """
    deck = []

    def shuffle_deck():
        deck.clear()
        for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
            for suit in ["h", "d", "s", "c"]:
                deck.append(rank + suit)
        random.shuffle(deck)

    boards = []
    player_hands = []
    opponent_hands = []
    for _ in range(n):
        shuffle_deck()
        boards.append(deck[:6])
        player_hands.append(deck[6:8])
        opponent_hands.append(deck[8:10])

    return boards, player_hands, opponent_hands


def Deck(excluded_cards=[]):
    # Returns a shuffled deck
    deck = []
    for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
        for suit in ["h", "d", "s", "c"]:
            if rank + suit not in excluded_cards:
                deck.append(rank + suit)

    random.shuffle(deck)
    return deck


def evaluate_best_cards(cards):
    """Return the best (lowest) phevaluator score for 7+ cards."""
    if len(cards) <= 7:
        return evaluate_cards(*cards)

    best_score = None
    for combo in combinations(cards, 7):
        score = evaluate_cards(*combo)
        if best_score is None or score < best_score:
            best_score = score
    return best_score


def get_player_score(player_cards, board=[]):
    """Wrapper for the evaluate_cards function by phevaluator."""
    assert len(player_cards) == 2
    assert len(board) <= 6
    # Returns a score using the phevaluator library
    return evaluate_best_cards(player_cards + board)
