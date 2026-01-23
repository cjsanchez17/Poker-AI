'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import joblib
import os
import random
import sys

DISCRETE_ACTIONS = ["k", "bMIN", "bMAX", "c", "f"]

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print(f"[CFR] BASE_DIR={BASE_DIR}")
print(f"[CFR] SRC_DIR={SRC_DIR}")

try:
    import postflop_holdem  # type: ignore
except ImportError:
    postflop_holdem = None

if postflop_holdem is None:
    try:
        import importlib.util

        postflop_path = os.path.join(SRC_DIR, "postflop_holdem.py")
        if os.path.exists(postflop_path):
            spec = importlib.util.spec_from_file_location("postflop_holdem", postflop_path)
            if spec and spec.loader:
                postflop_holdem = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(postflop_holdem)
    except Exception:
        postflop_holdem = None

try:
    from abstraction import predict_cluster  # type: ignore
except ImportError:
    predict_cluster = None

# from discard_helper import choose_card_to_toss

class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.cfr_history = []
        self.last_board_len = 0
        self.player_index = None
        self.postflop_infosets = None

        infoset_path = os.path.join(SRC_DIR, "postflop_infoSets_batch_19.joblib")
        print(f"[CFR] Infoset path: {infoset_path}")
        if os.path.exists(infoset_path):
            if postflop_holdem is not None:
                sys.modules["__main__"].PostflopHoldemInfoSet = (
                    postflop_holdem.PostflopHoldemInfoSet
                )
            try:
                self.postflop_infosets = joblib.load(infoset_path)
                print(f"[CFR] Loaded infosets: {len(self.postflop_infosets)}")
            except Exception as error:
                print(f"Failed to load CFR infosets: {error}")
        else:
            print("[CFR] Infoset file not found.")

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # the total number of seconds your bot has left to play this game
        game_clock = game_state.game_clock
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        self.player_index = active
        self.cfr_history = []
        self.last_board_len = 0

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0,2,3,4,5,6 representing when this round ended
        my_cards = previous_state.hands[active]  # your cards
        # opponent's cards or [] if not revealed
        opp_cards = previous_state.hands[1-active]
        pass

    def handle_engine_update(self, clause, round_state, active):
        clause_type = clause[0]
        if clause_type == "H":
            hand_str = "".join(clause[1:].split(","))
            self.player_index = active
            if active == 0:
                self.cfr_history = [hand_str, "XXXX"]
            else:
                self.cfr_history = ["XXXX", hand_str]
            self.last_board_len = 0
        elif clause_type == "O":
            opp_hand = "".join(clause[1:].split(","))
            if self.player_index == 0:
                self.cfr_history[1] = opp_hand
            else:
                self.cfr_history[0] = opp_hand
        elif clause_type == "B":
            board = round_state.board
            board_len = len(board)
            if board_len >= 4 and self.last_board_len < 4:
                self.cfr_history.append("/")
                self.cfr_history.append("".join(board[:4]))
            if board_len >= 5 and self.last_board_len < 5:
                self.cfr_history.append("/")
                self.cfr_history.append(board[4])
            if board_len >= 6 and self.last_board_len < 6:
                self.cfr_history.append("/")
                self.cfr_history.append(board[5])
            self.last_board_len = board_len
        elif clause_type in {"F", "C", "K", "R"}:
            if clause_type == "F":
                self.cfr_history.append("f")
            elif clause_type == "C":
                self.cfr_history.append("c")
            elif clause_type == "K":
                self.cfr_history.append("k")
            elif clause_type == "R":
                self.cfr_history.append(f"b{clause[1:]}")

    def _get_stage(self, history):
        if "/" in history:
            return history[: history.index("/")]
        return history

    def _perform_postflop_abstraction(self, history):
        history = list(history)
        pot_total = BIG_BLIND * 2

        if "/" in history:
            flop_start = history.index("/")
            for action in history[:flop_start]:
                if action[0] == "b":
                    bet_size = int(action[1:])
                    pot_total = 2 * bet_size
        else:
            return history

        abstracted_history = history[:2]
        stage_start = flop_start
        stage = self._get_stage(history[stage_start + 1 :])
        latest_bet = 0
        while True:
            abstracted_history.append("/")
            if len(stage) >= 4 and stage[3] != "c":
                abstracted_history.append(stage[0])
                if stage[-1] == "c":
                    if len(stage) % 2 == 1:
                        abstracted_history += ["bMAX", "c"]
                    else:
                        if stage[0] == "k":
                            abstracted_history += ["k", "bMAX", "c"]
                        else:
                            abstracted_history += ["bMIN", "bMAX", "c"]
                else:
                    if len(stage) % 2 == 0:
                        abstracted_history.append("bMAX")
                    else:
                        abstracted_history += ["bMIN", "bMAX"]
            else:
                for action in stage:
                    if action[0] == "b":
                        bet_size = int(action[1:])
                        latest_bet = bet_size
                        if abstracted_history[-1] == "bMIN":
                            abstracted_history.append("bMAX")
                        elif abstracted_history[-1] == "bMAX":
                            abstracted_history[-1] = "bMIN"
                            abstracted_history.append("bMAX")
                        else:
                            if bet_size >= pot_total:
                                abstracted_history.append("bMAX")
                            else:
                                abstracted_history.append("bMIN")
                        pot_total += bet_size
                    elif action == "c":
                        pot_total += latest_bet
                        abstracted_history.append("c")
                    else:
                        abstracted_history.append(action)
            if "/" not in history[stage_start + 1 :]:
                break
            stage_start = history[stage_start + 1 :].index("/") + (stage_start + 1)
            stage = self._get_stage(history[stage_start + 1 :])

        return abstracted_history

    def _pick_strategy_action(self, strategy):
        actions = list(strategy.keys())
        weights = list(strategy.values())
        return random.choices(actions, weights=weights, k=1)[0]

    def _build_infoset_key(self, history):
        if predict_cluster is None or self.player_index is None:
            return None

        infoset = []
        stage_i = 0
        if self.player_index == 0:
            hand = [history[0][:2], history[0][2:4]]
        else:
            hand = [history[1][:2], history[1][2:4]]
        community_cards = []

        for action in history:
            if action not in DISCRETE_ACTIONS:
                if action == "/":
                    stage_i += 1
                    continue
                if stage_i != 0:
                    community_cards += [
                        action[i : i + 2] for i in range(0, len(action), 2)
                    ]
                if stage_i in {1, 2, 3}:
                    infoset.append(str(predict_cluster(hand + community_cards)))
            else:
                infoset.append(action)

        return "".join(infoset)

    def _map_cfr_action(self, abstracted_action, round_state):
        legal_actions = round_state.legal_actions()
        my_pip = round_state.pips[self.player_index]
        my_stack = round_state.stacks[self.player_index]
        opp_stack = round_state.stacks[1 - self.player_index]
        total_pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)
        min_raise, max_raise = round_state.raise_bounds()

        if abstracted_action == "bMIN" and RaiseAction in legal_actions:
            desired_bet = max(BIG_BLIND, int(total_pot / 3))
            desired_total = my_pip + desired_bet
            raise_amount = max(min_raise, min(desired_total, max_raise))
            return RaiseAction(raise_amount)
        if abstracted_action == "bMAX" and RaiseAction in legal_actions:
            desired_bet = total_pot
            desired_total = my_pip + desired_bet
            raise_amount = max(min_raise, min(desired_total, max_raise))
            return RaiseAction(raise_amount)
        if abstracted_action == "f" and FoldAction in legal_actions:
            return FoldAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        street = round_state.street
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.board  # the board cards
        # the number of chips you have contributed to the pot this round of betting
        my_pip = round_state.pips[active]
        # the number of chips your opponent has contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]
        # the number of chips you have remaining
        my_stack = round_state.stacks[active]
        # the number of chips your opponent has remaining
        opp_stack = round_state.stacks[1-active]
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        # the number of chips you have contributed to the pot
        my_contribution = STARTING_STACK - my_stack
        # the number of chips your opponent has contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack

        # Only use DiscardAction if it's in legal_actions (which already checks street)
        # legal_actions() returns DiscardAction only when street is 2 or 3
        if DiscardAction in legal_actions:
            # # Always discards the lowest rank card in your hand

            ranks = "23456789TJQKA" # order of ranks
            rank_list = [-1, -1, -1] # uninitialized

            for card in range(3): # loop through cards in hand
                rank_list[card] = ranks.index(my_cards[card][0]) # get rank of each card

            # find the card with the minimum rank
            if rank_list[0] <= rank_list[1] and rank_list[0] <= rank_list[2]:
                return DiscardAction(0)
            elif rank_list[1] <= rank_list[2]:
                return DiscardAction(1)
            else:
                return DiscardAction(2)
            # DiscardAction(0)
        
        if len(round_state.board) < 4:
            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction()

        abstracted_history = self._perform_postflop_abstraction(self.cfr_history)
        infoset_key = self._build_infoset_key(abstracted_history)
        if infoset_key and self.postflop_infosets:
            if infoset_key in self.postflop_infosets:
                strategy = self.postflop_infosets[infoset_key].get_average_strategy()
                abstracted_action = self._pick_strategy_action(strategy)
                print(f"[CFR] Using infoset {infoset_key} -> {abstracted_action}")
                return self._map_cfr_action(abstracted_action, round_state)

        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
