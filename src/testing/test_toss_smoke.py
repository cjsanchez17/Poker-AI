import os
import sys
import unittest

devpath = os.path.relpath(os.path.join(".."), start=os.path.dirname(__file__))
sys.path = [devpath] + sys.path

from toss_abstraction import generate_postdiscard_dataset, generate_prediscard_dataset
from toss_cards import index_to_card_str
from toss_hand_eval import best_hand_rank


class TossSmokeTests(unittest.TestCase):
    def test_card_index_mapping(self) -> None:
        self.assertEqual(index_to_card_str(0), "Ac")
        self.assertEqual(index_to_card_str(51), "Ks")

    def test_dataset_unique_cards(self) -> None:
        player_hands, opponent_hands = generate_prediscard_dataset(50, seed=1)
        boards, _ = generate_postdiscard_dataset(
            player_hands, opponent_hands, seed=2, compute_winners=False
        )
        for player, opponent, board in zip(player_hands, opponent_hands, boards):
            combined = list(player) + list(opponent) + list(board)
            self.assertEqual(len(combined), len(set(combined)))

    def test_best_hand_rank_ordering(self) -> None:
        # In Toss or Hold'em, players start with 3 hole cards and finish with 6 board cards.
        # best_hand_rank picks the best 5-card hand using any 2 of the 3 hole cards plus 5 board cards.
        hero_hole = ["Ah", "Kh", "Qh"]
        villain_hole = ["As", "Ad", "2c"]
        board = ["Jh", "Th", "9h", "2d", "3c", "4s"]
        hero_rank = best_hand_rank(hero_hole, board)
        villain_rank = best_hand_rank(villain_hole, board)
        self.assertLess(hero_rank, villain_rank)


if __name__ == "__main__":
    unittest.main()
