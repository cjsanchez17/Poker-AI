import random
from typing import Dict, Iterable, List, Optional

import joblib

import toss_abstraction
from toss_preflop import PreDiscardTossHistory
from toss_postdiscard import PostDiscardTossHistory


class TossStrategy:
    def __init__(
        self,
        prediscard_strategy_path: str,
        postdiscard_strategy_path: str,
        prediscard_clusters_path: Optional[str] = None,
        flop4_clusters_path: Optional[str] = None,
    ) -> None:
        self.prediscard_strategy = joblib.load(prediscard_strategy_path)
        self.postdiscard_strategy = joblib.load(postdiscard_strategy_path)

        if prediscard_clusters_path:
            toss_abstraction.load_prediscard_clusters(prediscard_clusters_path)
        if flop4_clusters_path:
            toss_abstraction.load_flop4_clusters(flop4_clusters_path)

    def _sample_action(self, strategy: Dict[str, float]) -> str:
        actions = list(strategy.keys())
        weights = list(strategy.values())
        return random.choices(actions, weights=weights, k=1)[0]

    def act_prediscard(self, history: List[str], sample_id: int = 0) -> str:
        node = PreDiscardTossHistory(history=history, sample_id=sample_id)
        info_key = "".join(node.get_infoSet_key())
        infoset = self.prediscard_strategy.get(info_key)
        if infoset is None:
            return random.choice(node.actions())
        strategy = infoset.get_average_strategy()
        return self._sample_action(strategy)

    def act_postdiscard(self, history: List[str], sample_id: int = 0) -> str:
        node = PostDiscardTossHistory(history=history, sample_id=sample_id)
        info_key = "".join(node.get_infoSet_key())
        infoset = self.postdiscard_strategy.get(info_key)
        if infoset is None:
            return random.choice(node.actions())
        strategy = infoset.get_average_strategy()
        return self._sample_action(strategy)


def build_prediscard_history(player_hand: str, opponent_hand: str, actions: Iterable[str]) -> List[str]:
    history = [player_hand, opponent_hand, "/"]
    history.extend(actions)
    return history


def build_postdiscard_history(
    player_hand: str,
    opponent_hand: str,
    flop4: str,
    actions: Iterable[str],
) -> List[str]:
    history = [player_hand, opponent_hand, "/", flop4, "/"]
    history.extend(actions)
    return history
