import argparse
from typing import Dict, Iterable, List, Tuple

import joblib

import toss_abstraction
from toss_cards import cards_to_str
from toss_preflop import PreDiscardTossHistory
from toss_postdiscard import PostDiscardTossHistory


def build_prediscard_history(player_hand: Iterable[int], opponent_hand: Iterable[int]) -> List[str]:
    return ["".join(cards_to_str(player_hand)), "".join(cards_to_str(opponent_hand)), "/"]


def build_postdiscard_history(
    player_hand: Iterable[int], opponent_hand: Iterable[int], flop4: Iterable[int]
) -> List[str]:
    return [
        "".join(cards_to_str(player_hand)),
        "".join(cards_to_str(opponent_hand)),
        "/",
        "".join(cards_to_str(flop4)),
        "/",
    ]


def evaluate_coverage(
    strategy: Dict[str, object], histories: Iterable[List[str]], history_cls
) -> Tuple[int, int]:
    found = 0
    total = 0
    for sample_id, history in enumerate(histories):
        node = history_cls(history=history, sample_id=sample_id)
        info_key = "".join(node.get_infoSet_key())
        if info_key in strategy:
            found += 1
        total += 1
    return found, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Toss-or-Hold'em strategy coverage.")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--prediscard-strategy", required=True)
    parser.add_argument("--postdiscard-strategy", required=True)
    parser.add_argument("--prediscard-clusters", default=None)
    parser.add_argument("--flop4-clusters", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.prediscard_clusters:
        toss_abstraction.load_prediscard_clusters(args.prediscard_clusters)
    if args.flop4_clusters:
        toss_abstraction.load_flop4_clusters(args.flop4_clusters)

    toss_abstraction.load_prediscard_dataset(args.batch)
    toss_abstraction.load_postdiscard_dataset(args.batch)

    prediscard_strategy = joblib.load(args.prediscard_strategy)
    postdiscard_strategy = joblib.load(args.postdiscard_strategy)

    max_samples = args.max_samples or len(toss_abstraction.prediscard_player_hands)
    player_hands = toss_abstraction.prediscard_player_hands[:max_samples]
    opponent_hands = toss_abstraction.prediscard_opponent_hands[:max_samples]
    boards = toss_abstraction.postdiscard_boards[:max_samples]

    prediscard_histories = [
        build_prediscard_history(player, opponent)
        for player, opponent in zip(player_hands, opponent_hands)
    ]
    postdiscard_histories = [
        build_postdiscard_history(player, opponent, board[:4])
        for player, opponent, board in zip(player_hands, opponent_hands, boards)
    ]

    prediscard_found, prediscard_total = evaluate_coverage(
        prediscard_strategy, prediscard_histories, PreDiscardTossHistory
    )
    postdiscard_found, postdiscard_total = evaluate_coverage(
        postdiscard_strategy, postdiscard_histories, PostDiscardTossHistory
    )

    print(
        "Prediscard coverage: "
        f"{prediscard_found}/{prediscard_total} "
        f"({prediscard_found / max(prediscard_total, 1):.1%})"
    )
    print(
        "Postdiscard coverage: "
        f"{postdiscard_found}/{postdiscard_total} "
        f"({postdiscard_found / max(postdiscard_total, 1):.1%})"
    )


if __name__ == "__main__":
    main()
