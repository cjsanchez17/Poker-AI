import base
from base import Action, Player
from typing import List

import toss_abstraction
from toss_cards import cards_to_str

DISCRETE_ACTIONS = ["k", "bMIN", "bMAX", "c", "f"]

player_hands = None
opponent_hands = None


class PreDiscardTossHistory(base.History):
    def __init__(self, history: List[Action] = [], sample_id: int = 0):
        super().__init__(history)
        self.sample_id = sample_id

    def is_terminal(self):
        if len(self.history) == 0:
            return False
        if "f" in self.history:
            return True
        return self._game_stage_ended()

    def actions(self) -> List[Action]:
        if self.is_chance():
            return []
        if self.is_terminal():
            raise Exception("Cannot call actions on a terminal history")

        last_action = self.history[-1]
        if last_action == "k":
            return ["k", "bMIN", "bMAX"]
        if last_action == "bMIN":
            return ["bMAX", "f", "c"]
        if last_action == "bMAX":
            return ["f", "c"]
        if last_action == "/":
            return ["k", "bMIN", "bMAX"]
        return ["k", "bMIN", "bMAX"]

    def player(self) -> Player:
        if len(self.history) < 3:
            return -1
        if self._game_stage_ended():
            return -1
        last_action = self.history[-1]
        if last_action == "/":
            return 0
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
        return "/"

    def terminal_utility(self, i: Player) -> int:
        assert self.is_terminal()
        assert i in [0, 1]

        if "f" not in self.history:
            return 0

        pot_size, _ = self._get_total_pot_size(self.history)
        fold_idx = self.history.index("f")
        pot_size, latest_bet = self._get_total_pot_size(self.history[: fold_idx - 1])
        if fold_idx >= 3 and self.history[fold_idx - 1] == "bMIN":
            pot_size += latest_bet

        if len(self.history) % 2 == i:
            return -pot_size / 2
        return pot_size / 2

    def _get_total_pot_size(self, history: List[Action]):
        stage_total = 3
        latest_bet = 2

        for action in history:
            if action == "bMIN":
                old_stage_total = stage_total
                stage_total = latest_bet + stage_total
                latest_bet = old_stage_total
            elif action == "bMAX":
                stage_total = latest_bet + 400
                latest_bet = 400
            elif action == "c":
                stage_total = 2 * latest_bet

        return stage_total, latest_bet

    def __add__(self, action: Action):
        return PreDiscardTossHistory(self.history + [action], self.sample_id)

    def get_infoSet_key(self) -> List[Action]:
        assert not self.is_chance()
        assert not self.is_terminal()

        player = self.player()
        infoset = []
        hand = self.history[player]
        hand_id = toss_abstraction.prediscard_clusters.get(hand, hand)
        infoset.append(str(hand_id))
        for action in self.history:
            if action in DISCRETE_ACTIONS:
                infoset.append(action)
        return infoset


class PreDiscardTossInfoSet(base.InfoSet):
    def __init__(self, infoSet_key: List[Action], actions: List[Action], player: Player):
        assert len(infoSet_key) >= 1
        super().__init__(infoSet_key, actions, player)


def create_infoSet(infoSet_key: List[Action], actions: List[Action], player: Player):
    return PreDiscardTossInfoSet(infoSet_key, actions, player)


def create_history(sample_id: int):
    if player_hands:
        sample_id = sample_id % len(player_hands)
    return PreDiscardTossHistory(sample_id=sample_id)
