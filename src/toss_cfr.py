import base
from toss_preflop import create_history as create_pred_history, create_infoSet as create_pred_infoset
from toss_postdiscard import create_history as create_post_history, create_infoSet as create_post_infoset


class PreDiscardTossCFR(base.CFR):
    def __init__(self, iterations: int = 1000000):
        super().__init__(create_pred_infoset, create_pred_history, iterations=iterations)


class PostDiscardTossCFR(base.CFR):
    def __init__(self, iterations: int = 1000000):
        super().__init__(create_post_infoset, create_post_history, iterations=iterations)


if __name__ == "__main__":
    import toss_abstraction
    import toss_preflop
    import toss_postdiscard

    ITERATIONS = 50000

    pre_cfr = PreDiscardTossCFR(iterations=ITERATIONS)
    post_cfr = PostDiscardTossCFR(iterations=ITERATIONS)

    for batch in range(1):
        toss_abstraction.load_prediscard_dataset(batch)
        toss_preflop.player_hands = toss_abstraction.prediscard_player_hands
        toss_preflop.opponent_hands = toss_abstraction.prediscard_opponent_hands
        pre_cfr.solve(debug=False, method="vanilla")
        pre_cfr.export_infoSets(f"prediscard_infoSets_batch_{batch}.joblib")

        toss_abstraction.load_postdiscard_dataset(batch)
        toss_postdiscard.player_hands = toss_abstraction.prediscard_player_hands
        toss_postdiscard.opponent_hands = toss_abstraction.prediscard_opponent_hands
        toss_postdiscard.boards = toss_abstraction.postdiscard_boards
        toss_postdiscard.winners = toss_abstraction.postdiscard_winners
        post_cfr.solve(debug=False, method="vanilla")
        post_cfr.export_infoSets(f"postdiscard_infoSets_batch_{batch}.joblib")
