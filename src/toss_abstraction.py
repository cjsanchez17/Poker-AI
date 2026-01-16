import itertools
import random
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.cluster import KMeans

from toss_cards import cards_to_str
from toss_hand_eval import best_hand_rank

PREDISCARD_CLUSTER_COUNT = 100
FLOP4_CLUSTER_COUNT = 100

prediscard_player_hands: List[List[int]] = []
prediscard_opponent_hands: List[List[int]] = []
postdiscard_boards: List[List[int]] = []
postdiscard_winners: List[int] = []

prediscard_clusters: Dict[str, int] = {}
flop4_clusters: Dict[str, int] = {}


def generate_prediscard_dataset(num_samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    player_hands = np.zeros((num_samples, 3), dtype=np.int16)
    opponent_hands = np.zeros((num_samples, 3), dtype=np.int16)

    for idx in range(num_samples):
        deck = list(range(52))
        rng.shuffle(deck)
        player_hands[idx] = deck[:3]
        opponent_hands[idx] = deck[3:6]

    return player_hands, opponent_hands


def generate_postdiscard_dataset(
    player_hands: np.ndarray,
    opponent_hands: np.ndarray,
    seed: int | None = None,
    compute_winners: bool = True,
) -> Tuple[np.ndarray, np.ndarray | None]:
    rng = random.Random(seed)
    num_samples = len(player_hands)
    boards = np.zeros((num_samples, 6), dtype=np.int16)
    winners = np.zeros(num_samples, dtype=np.int8) if compute_winners else None

    for idx in range(num_samples):
        used = set(player_hands[idx].tolist() + opponent_hands[idx].tolist())
        deck = [card for card in range(52) if card not in used]
        board = rng.sample(deck, 6)
        boards[idx] = board

        if compute_winners:
            player_rank = best_hand_rank(cards_to_str(player_hands[idx]), cards_to_str(board))
            opponent_rank = best_hand_rank(cards_to_str(opponent_hands[idx]), cards_to_str(board))
            if player_rank < opponent_rank:
                winners[idx] = 1
            elif player_rank > opponent_rank:
                winners[idx] = -1
            else:
                winners[idx] = 0

    return boards, winners


def save_prediscard_dataset(batch: int, player_hands: np.ndarray, opponent_hands: np.ndarray) -> None:
    np.save(f"dataset/toss_player_hands_{batch}.npy", player_hands)
    np.save(f"dataset/toss_opponent_hands_{batch}.npy", opponent_hands)


def save_postdiscard_dataset(batch: int, boards: np.ndarray, winners: np.ndarray | None) -> None:
    np.save(f"dataset/toss_boards_{batch}.npy", boards)
    if winners is not None:
        np.save(f"dataset/toss_winners_{batch}.npy", winners)


def load_prediscard_dataset(batch: int = 0) -> None:
    global prediscard_player_hands, prediscard_opponent_hands
    prediscard_player_hands = np.load(f"dataset/toss_player_hands_{batch}.npy").tolist()
    prediscard_opponent_hands = np.load(f"dataset/toss_opponent_hands_{batch}.npy").tolist()


def load_postdiscard_dataset(batch: int = 0) -> None:
    global postdiscard_boards, postdiscard_winners
    postdiscard_boards = np.load(f"dataset/toss_boards_{batch}.npy").tolist()
    winners_path = f"dataset/toss_winners_{batch}.npy"
    postdiscard_winners = np.load(winners_path).tolist()


def estimate_hand_strength(three_hole: Iterable[int], samples: int = 200) -> float:
    deck = list(range(52))
    hole = list(three_hole)
    for card in hole:
        deck.remove(card)
    wins = 0
    ties = 0

    for _ in range(samples):
        sample = random.sample(deck, 9)
        opponent = sample[:3]
        board = sample[3:9]
        hero_rank = best_hand_rank(cards_to_str(hole), cards_to_str(board))
        opp_rank = best_hand_rank(cards_to_str(opponent), cards_to_str(board))
        if hero_rank < opp_rank:
            wins += 1
        elif hero_rank == opp_rank:
            ties += 1

    return (wins + ties * 0.5) / samples


def cluster_prediscard_hands(three_card_hands: Iterable[Iterable[int]]) -> Dict[str, int]:
    features = np.array([estimate_hand_strength(hand) for hand in three_card_hands]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=PREDISCARD_CLUSTER_COUNT, random_state=42)
    labels = kmeans.fit_predict(features)
    mapping = {"".join(cards_to_str(hand)): int(label) for hand, label in zip(three_card_hands, labels)}
    return mapping


def estimate_flop4_strength(flop4: Iterable[int], samples: int = 200) -> float:
    deck = list(range(52))
    board = list(flop4)
    for card in board:
        deck.remove(card)
    wins = 0
    ties = 0

    for _ in range(samples):
        sample = random.sample(deck, 8)
        hero = sample[:3]
        opp = sample[3:6]
        runout = sample[6:8]
        final_board = board + runout
        hero_rank = best_hand_rank(cards_to_str(hero), cards_to_str(final_board))
        opp_rank = best_hand_rank(cards_to_str(opp), cards_to_str(final_board))
        if hero_rank < opp_rank:
            wins += 1
        elif hero_rank == opp_rank:
            ties += 1

    return (wins + ties * 0.5) / samples


def cluster_flop4_boards(flop4_boards: Iterable[Iterable[int]]) -> Dict[str, int]:
    features = np.array([estimate_flop4_strength(board) for board in flop4_boards]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=FLOP4_CLUSTER_COUNT, random_state=42)
    labels = kmeans.fit_predict(features)
    mapping = {"".join(cards_to_str(board)): int(label) for board, label in zip(flop4_boards, labels)}
    return mapping


def load_prediscard_clusters(path: str) -> None:
    global prediscard_clusters
    prediscard_clusters = joblib.load(path)


def load_flop4_clusters(path: str) -> None:
    global flop4_clusters
    flop4_clusters = joblib.load(path)


def save_clusters(path: str, clusters: Dict[str, int]) -> None:
    joblib.dump(clusters, path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate datasets and clusters for Toss-or-Hold'em")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--prediscard-clusters", default="dataset/toss_prediscard_clusters.pkl")
    parser.add_argument("--flop4-clusters", default="dataset/toss_flop4_clusters.pkl")
    args = parser.parse_args()

    if args.generate:
        player_hands, opponent_hands = generate_prediscard_dataset(args.samples, seed=args.seed)
        save_prediscard_dataset(args.batch, player_hands, opponent_hands)

        boards, winners = generate_postdiscard_dataset(
            player_hands, opponent_hands, seed=args.seed, compute_winners=True
        )
        save_postdiscard_dataset(args.batch, boards, winners)

    if args.cluster:
        load_prediscard_dataset(args.batch)
        load_postdiscard_dataset(args.batch)

        prediscard_map = cluster_prediscard_hands(prediscard_player_hands)
        flop4_map = cluster_flop4_boards([board[:4] for board in postdiscard_boards])

        save_clusters(args.prediscard_clusters, prediscard_map)
        save_clusters(args.flop4_clusters, flop4_map)
