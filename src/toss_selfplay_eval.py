import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib

import toss_abstraction
import toss_preflop
import toss_postdiscard
from base import Action
from toss_preflop import PreDiscardTossHistory
from toss_postdiscard import PostDiscardTossHistory


@dataclass
class EvalStats:
    total_utility: float = 0.0
    hands_played: int = 0
    fallback_actions: int = 0
    total_actions: int = 0

    def record_action(self, used_fallback: bool) -> None:
        self.total_actions += 1
        if used_fallback:
            self.fallback_actions += 1


def sample_action(
    node, strategy: Dict[str, object], rng: random.Random
) -> Tuple[Action, bool]:
    info_key = "".join(node.get_infoSet_key())
    infoset = strategy.get(info_key)
    if infoset is None:
        return rng.choice(node.actions()), True
    avg_strategy = infoset.get_average_strategy()
    actions = list(avg_strategy.keys())
    weights = list(avg_strategy.values())
    return rng.choices(actions, weights=weights, k=1)[0], False


def play_hand(
    history_cls,
    strategy: Dict[str, object],
    sample_id: int,
    rng: random.Random,
    stats: EvalStats,
    max_actions: int = 200,
) -> int:
    node = history_cls(sample_id=sample_id)
    steps = 0
    while not node.is_terminal():
        if node.is_chance():
            node = node + node.sample_chance_outcome()
            steps += 1
            continue
        action, used_fallback = sample_action(node, strategy, rng)
        stats.record_action(used_fallback)
        node = node + action
        steps += 1
        if steps > max_actions:
            raise RuntimeError("Exceeded max actions; possible loop in history logic.")
    return node.terminal_utility(0)


def evaluate_stage(
    history_cls,
    strategy: Dict[str, object],
    num_hands: int,
    rng: random.Random,
) -> EvalStats:
    stats = EvalStats()
    for sample_id in range(num_hands):
        stats.total_utility += play_hand(history_cls, strategy, sample_id, rng, stats)
        stats.hands_played += 1
    return stats


def print_stats(label: str, stats: EvalStats) -> None:
    avg_utility = stats.total_utility / max(stats.hands_played, 1)
    fallback_rate = stats.fallback_actions / max(stats.total_actions, 1)
    print(f"{label} average utility (player 0): {avg_utility:.2f}")
    print(f"{label} fallback rate: {fallback_rate:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play evaluation for Toss-or-Hold'em strategies.")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--prediscard-strategy", required=True)
    parser.add_argument("--postdiscard-strategy", required=True)
    parser.add_argument("--prediscard-clusters", default=None)
    parser.add_argument("--flop4-clusters", default=None)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.prediscard_clusters:
        toss_abstraction.load_prediscard_clusters(args.prediscard_clusters)
    if args.flop4_clusters:
        toss_abstraction.load_flop4_clusters(args.flop4_clusters)

    toss_abstraction.load_prediscard_dataset(args.batch)
    toss_abstraction.load_postdiscard_dataset(args.batch)

    toss_preflop.player_hands = toss_abstraction.prediscard_player_hands
    toss_preflop.opponent_hands = toss_abstraction.prediscard_opponent_hands

    toss_postdiscard.player_hands = toss_abstraction.prediscard_player_hands
    toss_postdiscard.opponent_hands = toss_abstraction.prediscard_opponent_hands
    toss_postdiscard.boards = toss_abstraction.postdiscard_boards
    toss_postdiscard.winners = toss_abstraction.postdiscard_winners

    prediscard_strategy = joblib.load(args.prediscard_strategy)
    postdiscard_strategy = joblib.load(args.postdiscard_strategy)

    max_hands = min(args.hands, len(toss_abstraction.prediscard_player_hands))

    pre_stats = evaluate_stage(PreDiscardTossHistory, prediscard_strategy, max_hands, rng)
    post_stats = evaluate_stage(PostDiscardTossHistory, postdiscard_strategy, max_hands, rng)

    print_stats("Prediscard", pre_stats)
    print_stats("Postdiscard", post_stats)


if __name__ == "__main__":
    main()
