"""Deterministic poker helpers for scenario gameplay."""

from __future__ import annotations

from itertools import combinations
import random
from typing import Iterable


RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "CDHS"
RANK_TO_VALUE = {char: idx + 2 for idx, char in enumerate(RANK_CHARS)}
SUIT_TO_ORDER = {char: idx for idx, char in enumerate(SUIT_CHARS)}


def _parse_card(card: str) -> tuple[int, str]:
    text = str(card or "").strip().upper()
    if len(text) != 2:
        raise ValueError(f"Invalid card: {card}")
    rank_char = text[0]
    suit_char = text[1]
    if rank_char not in RANK_TO_VALUE or suit_char not in SUIT_TO_ORDER:
        raise ValueError(f"Invalid card: {card}")
    return (RANK_TO_VALUE[rank_char], suit_char)


def _card_sort_key(card: str) -> tuple[int, int]:
    rank, suit = _parse_card(card)
    return (-rank, SUIT_TO_ORDER[suit])


def make_shuffled_deck(*, seed: str) -> list[str]:
    deck = [f"{rank}{suit}" for suit in SUIT_CHARS for rank in RANK_CHARS]
    rng = random.Random(str(seed))
    rng.shuffle(deck)
    return deck


def _straight_high(ranks: Iterable[int]) -> int | None:
    unique = sorted(set(int(value) for value in ranks), reverse=True)
    if len(unique) < 5:
        return None
    if unique == [14, 5, 4, 3, 2]:
        return 5
    for idx in range(len(unique) - 4):
        window = unique[idx : idx + 5]
        if all(window[i] - 1 == window[i + 1] for i in range(4)):
            return int(window[0])
    return None


def evaluate_five(cards: list[str]) -> tuple[int, tuple[int, ...]]:
    """Evaluate a 5-card poker hand.

    Category values (higher is better):
    9 = straight flush
    8 = four of a kind
    7 = full house
    6 = flush
    5 = straight
    4 = three of a kind
    3 = two pair
    2 = one pair
    1 = high card
    """

    if len(cards) != 5:
        raise ValueError("evaluate_five expects exactly 5 cards")

    parsed = [_parse_card(card) for card in cards]
    ranks = [rank for rank, _ in parsed]
    suits = [suit for _, suit in parsed]

    rank_counts: dict[int, int] = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    count_groups = sorted(((count, rank) for rank, count in rank_counts.items()), reverse=True)

    flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if flush and straight_high is not None:
        return (9, (straight_high,))

    if count_groups[0][0] == 4:
        quad_rank = count_groups[0][1]
        kicker = max(rank for rank in ranks if rank != quad_rank)
        return (8, (quad_rank, kicker))

    if count_groups[0][0] == 3 and count_groups[1][0] == 2:
        return (7, (count_groups[0][1], count_groups[1][1]))

    if flush:
        return (6, tuple(sorted(ranks, reverse=True)))

    if straight_high is not None:
        return (5, (straight_high,))

    if count_groups[0][0] == 3:
        trip_rank = count_groups[0][1]
        kickers = sorted((rank for rank in ranks if rank != trip_rank), reverse=True)
        return (4, (trip_rank, *kickers))

    if count_groups[0][0] == 2 and count_groups[1][0] == 2:
        high_pair = max(count_groups[0][1], count_groups[1][1])
        low_pair = min(count_groups[0][1], count_groups[1][1])
        kicker = max(rank for rank in ranks if rank not in {high_pair, low_pair})
        return (3, (high_pair, low_pair, kicker))

    if count_groups[0][0] == 2:
        pair_rank = count_groups[0][1]
        kickers = sorted((rank for rank in ranks if rank != pair_rank), reverse=True)
        return (2, (pair_rank, *kickers))

    return (1, tuple(sorted(ranks, reverse=True)))


def compare_hands(hand_a: list[str], hand_b: list[str]) -> int:
    score_a = evaluate_five(hand_a)
    score_b = evaluate_five(hand_b)
    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


def evaluate_best_five(cards: list[str]) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
    if len(cards) < 5:
        raise ValueError("evaluate_best_five expects at least 5 cards")

    best_score: tuple[int, tuple[int, ...]] | None = None
    best_cards: tuple[str, ...] | None = None
    for combo in combinations(cards, 5):
        normalized = tuple(sorted((str(card).upper() for card in combo), key=_card_sort_key))
        score = evaluate_five(list(normalized))
        if best_score is None or score > best_score:
            best_score = score
            best_cards = normalized
        elif score == best_score and best_cards is not None and normalized < best_cards:
            best_cards = normalized

    if best_score is None or best_cards is None:
        raise ValueError("Unable to evaluate hand")
    return (best_score[0], best_score[1], best_cards)


def choose_pass_cards(hand: list[str], *, count: int) -> list[str]:
    if count <= 0:
        return []
    cards = [str(card).upper() for card in hand]
    if not cards:
        return []
    count = min(len(cards), int(count))

    rank_counts: dict[int, int] = {}
    suit_counts: dict[str, int] = {}
    for card in cards:
        rank, suit = _parse_card(card)
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        suit_counts[suit] = suit_counts.get(suit, 0) + 1

    scored: list[tuple[float, int, str]] = []
    for card in cards:
        rank, suit = _parse_card(card)
        keep_value = 0.0
        keep_value += float(rank) / 20.0
        if rank_counts.get(rank, 0) >= 2:
            keep_value += 3.5
        if suit_counts.get(suit, 0) >= 4:
            keep_value += 1.25
        scored.append((keep_value, rank, card))

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[2] for item in scored[:count]]


def arrange_reveal_order(best_five: list[str]) -> list[str]:
    cards = [str(card).upper() for card in best_five]
    cards.sort(key=lambda card: (_parse_card(card)[0], SUIT_TO_ORDER[_parse_card(card)[1]]))
    return cards


def hand_category_label(category: int) -> str:
    mapping = {
        9: "Straight Flush",
        8: "Four of a Kind",
        7: "Full House",
        6: "Flush",
        5: "Straight",
        4: "Three of a Kind",
        3: "Two Pair",
        2: "One Pair",
        1: "High Card",
    }
    return mapping.get(int(category), "Unknown")
