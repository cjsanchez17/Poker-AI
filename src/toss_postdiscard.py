import base
from base import Action, Player
from typing import List

import toss_abstraction
from toss_cards import cards_to_str
from toss_hand_eval import best_hand_rank

DISCRETE_ACTIONS = ["k", "bMIN", "bMAX", "c", "f"]

player_hands = None
opponent_hands = None
boards = None
winners = None


class PostDiscardTossHistory(base.History):
    def __init__(self, history: List[Action] = [], sample_id: int = 0):
        super().__init__(history)
        self.sample_id = sample_id
        self.stage_i = history.count("/")

    def is_terminal(self):
        if len(self.history) == 0:
            return False
        folded = self.history[-1] == "f"
        is_showdown = self.stage_i == 4 and self._game_stage_ended()
        return folded or is_showdown

    def actions(self) -> List[Action]:
        if self.is_chance():
            return []
        if self.is_terminal():
            raise Exception("Cannot call actions on a terminal history")

        last_action = self.history[-1]
        if last_action == "k":
            return ["k", "bMIN", "bMAX"]
        if self.history[-2:] == ["k", "bMIN"]:
            return ["f", "c"]
        if last_action == "bMIN":
            return ["bMAX", "f", "c"]
        if last_action == "bMAX":
            return ["f", "c"]
        return ["k", "bMIN", "bMAX"]

    def player(self) -> Player:
        if len(self.history) < 5:
            return -1
        if self._game_stage_ended():
            return -1
        if self.history[-1] == "/":
            if self.stage_i == 2:
                return 0
            return -1
        last_game_stage = self.get_last_game_stage()
        return (len(last_game_stage) + 1) % 2

    def _game_stage_ended(self) -> bool:
        return self.history[-1] == "c" or self.history[-1] == "f" or self.history[-2:] == ["k", "k"]

    def get_last_game_stage(self) -> List[Action]:
        last_game_stage_start_idx = max(idx for idx, val in enumerate(self.history) if val == "/")
        return self.history[last_game_stage_start_idx + 1 :]

    def is_chance(self):
        return super().is_chance()

    def _hand_to_string(self, hand) -> str:
        if isinstance(hand[0], int):
            return "".join(cards_to_str(hand))
        return "".join(hand)

    def sample_chance_outcome(self) -> Action:
        assert self.is_chance()
        if len(self.history) == 0:
            return self._hand_to_string(player_hands[self.sample_id])
        if len(self.history) == 1:
            return self._hand_to_string(opponent_hands[self.sample_id])
        if len(self.history) == 2:
            return "/"
        if len(self.history) == 3:
            return "".join(cards_to_str(boards[self.sample_id][:4]))
        if len(self.history) == 4:
            return "/"
        if self.history[-1] != "/":
            return "/"
        if self.stage_i == 3:
            return cards_to_str([boards[self.sample_id][4]])[0]
        if self.stage_i == 4:
            return cards_to_str([boards[self.sample_id][5]])[0]
        raise ValueError("Invalid chance state")

    def terminal_utility(self, i: Player) -> int:
        assert self.is_terminal()
        assert i in [0, 1]

        pot_size, _ = self._get_total_pot_size(self.history)

        if self.history[-1] == "f":
            pot_size, latest_bet = self._get_total_pot_size(self.history[:-2])
            if self.history[-3] == "bMIN":
                pot_size += latest_bet

            last_game_stage = self.get_last_game_stage()
            if len(last_game_stage) % 2 == i:
                return -pot_size / 2
            return pot_size / 2

        winner = None
        if winners is not None:
            winner = winners[self.sample_id]
        else:
            player_hand = self._split_cards(self.history[0])
            opponent_hand = self._split_cards(self.history[1])
            board = self._extract_board()
            player_rank = best_hand_rank(player_hand, board)
            opponent_rank = best_hand_rank(opponent_hand, board)
            if player_rank < opponent_rank:
                winner = 1
            elif player_rank > opponent_rank:
                winner = -1
            else:
                winner = 0

        if winner == 0:
            return 0
        if (winner == 1 and i == 0) or (winner == -1 and i == 1):
            return pot_size / 2
        return -pot_size / 2

    def _split_cards(self, cards: str) -> List[str]:
        return [cards[i : i + 2] for i in range(0, len(cards), 2)]

    def _extract_board(self) -> List[str]:
        board_cards: List[str] = []
        for idx, action in enumerate(self.history):
            if idx < 2:
                continue
            if action in DISCRETE_ACTIONS or action == "/":
                continue
            board_cards.extend(self._split_cards(action))
        return board_cards

    def _get_total_pot_size(self, history: List[Action]):
        total = 0
        stage_total = 4
        latest_bet = 0

        for action in history:
            if action == "/":
                total += stage_total
                stage_total = 0
                latest_bet = 0
            elif action == "bMIN":
                latest_bet = max(2, int(total / 3))
                stage_total += latest_bet
            elif action == "bMAX":
                latest_bet = total
                stage_total += latest_bet
            elif action == "c":
                stage_total = 2 * latest_bet

        total += stage_total
        return total, latest_bet

    def __add__(self, action: Action):
        return PostDiscardTossHistory(self.history + [action], self.sample_id)

    def get_infoSet_key(self) -> List[Action]:
        assert not self.is_chance()
        assert not self.is_terminal()

        infoset = []
        player = self.player()
        hand = self.history[player]
        hand_id = toss_abstraction.prediscard_clusters.get(hand, hand)
        infoset.append(str(hand_id))
        if len(self.history) > 3:
            flop4 = self.history[3]
            flop_id = toss_abstraction.flop4_clusters.get(flop4, flop4)
            infoset.append(str(flop_id))
        for action in self.history:
            if action in DISCRETE_ACTIONS or action == "/":
                infoset.append(action)
        return infoset


class PostDiscardTossInfoSet(base.InfoSet):
    def __init__(self, infoSet_key: List[Action], actions: List[Action], player: Player):
        assert len(infoSet_key) >= 1
        super().__init__(infoSet_key, actions, player)


def create_infoSet(infoSet_key: List[Action], actions: List[Action], player: Player):
    return PostDiscardTossInfoSet(infoSet_key, actions, player)


def create_history(sample_id: int):
    return PostDiscardTossHistory(sample_id=sample_id)

