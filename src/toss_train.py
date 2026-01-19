import argparse

import toss_abstraction
import toss_preflop
import toss_postdiscard
from toss_cfr import PreDiscardTossCFR, PostDiscardTossCFR


def load_clusters(prediscard_path: str, flop4_path: str) -> None:
    try:
        toss_abstraction.load_prediscard_clusters(prediscard_path)
    except FileNotFoundError:
        print(f"Prediscard clusters not found at {prediscard_path}. Using raw hand strings.")
    try:
        toss_abstraction.load_flop4_clusters(flop4_path)
    except FileNotFoundError:
        print(f"Flop4 clusters not found at {flop4_path}. Using raw board strings.")


def train_prediscard(batch: int, iterations: int, output_prefix: str) -> None:
    toss_abstraction.load_prediscard_dataset(batch)
    toss_preflop.player_hands = toss_abstraction.prediscard_player_hands
    toss_preflop.opponent_hands = toss_abstraction.prediscard_opponent_hands

    cfr = PreDiscardTossCFR(iterations=iterations)
    cfr.solve(debug=False, method="vanilla")
    cfr.export_infoSets(f"{output_prefix}_prediscard_{batch}.joblib")


def train_postdiscard(batch: int, iterations: int, output_prefix: str) -> None:
    toss_abstraction.load_prediscard_dataset(batch)
    toss_abstraction.load_postdiscard_dataset(batch)
    toss_postdiscard.player_hands = toss_abstraction.prediscard_player_hands
    toss_postdiscard.opponent_hands = toss_abstraction.prediscard_opponent_hands
    toss_postdiscard.boards = toss_abstraction.postdiscard_boards
    toss_postdiscard.winners = toss_abstraction.postdiscard_winners

    cfr = PostDiscardTossCFR(iterations=iterations)
    cfr.solve(debug=False, method="vanilla")
    cfr.export_infoSets(f"{output_prefix}_postdiscard_{batch}.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFR strategies for Toss-or-Hold'em")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default="toss_strategy")
    parser.add_argument("--prediscard-clusters", default="dataset/toss_prediscard_clusters.pkl")
    parser.add_argument("--flop4-clusters", default="dataset/toss_flop4_clusters.pkl")
    parser.add_argument("--generate-datasets", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--train-prediscard", action="store_true")
    parser.add_argument("--train-postdiscard", action="store_true")
    args = parser.parse_args()

    if args.generate_datasets:
        player_hands, opponent_hands = toss_abstraction.generate_prediscard_dataset(
            args.samples, seed=args.seed
        )
        toss_abstraction.save_prediscard_dataset(args.batch, player_hands, opponent_hands)

        boards, winners = toss_abstraction.generate_postdiscard_dataset(
            player_hands, opponent_hands, seed=args.seed, compute_winners=True
        )
        toss_abstraction.save_postdiscard_dataset(args.batch, boards, winners)

    if args.cluster:
        toss_abstraction.load_prediscard_dataset(args.batch)
        toss_abstraction.load_postdiscard_dataset(args.batch)
        prediscard_map = toss_abstraction.cluster_prediscard_hands(
            toss_abstraction.prediscard_player_hands
        )
        flop4_map = toss_abstraction.cluster_flop4_boards(
            [board[:4] for board in toss_abstraction.postdiscard_boards]
        )
        toss_abstraction.save_clusters(args.prediscard_clusters, prediscard_map)
        toss_abstraction.save_clusters(args.flop4_clusters, flop4_map)

    if args.train_prediscard or args.train_postdiscard:
        load_clusters(args.prediscard_clusters, args.flop4_clusters)

    if args.train_prediscard:
        train_prediscard(args.batch, args.iterations, args.output_prefix)

    if args.train_postdiscard:
        train_postdiscard(args.batch, args.iterations, args.output_prefix)


if __name__ == "__main__":
    main()
