import random

import torch

from src.data.rules import (
    PREDEFINED,
    bits_to_sb,
    bits_to_string,
    parse_rule,
    random_rule_bits,
    sb_to_bits,
    sb_to_string,
)


def test_round_trip_predefined():
    for name, (S, B) in PREDEFINED.items():
        # Bits round-trip
        bits = sb_to_bits(S, B)
        S2, B2 = bits_to_sb(bits)
        assert S2 == S and B2 == B
        # String parse should match predefined sets
        Sx, Bx = parse_rule(name)
        assert Sx == S and Bx == B
        # sb_to_string is canonical S.../B...; parsing it should match
        s = sb_to_string(S, B)
        S3, B3 = parse_rule(s)
        assert S3 == S and B3 == B
        # bits_to_string should match sb_to_string
        bits_s = bits_to_string(bits)
        assert bits_s == s


def test_parse_and_roundtrip_strings():
    for rule_str in ["S23/B3", "S23/B36", "B2/S", "b3/s23", "s/B3"]:
        S, B = parse_rule(rule_str)
        bits = sb_to_bits(S, B)
        S2, B2 = bits_to_sb(bits)
        assert S2 == S and B2 == B


def test_fuzz_random_rules():
    torch.manual_seed(0)
    random.seed(0)
    for _ in range(200):
        bits = random_rule_bits(p=0.5)
        S, B = bits_to_sb(bits)
        bits2 = sb_to_bits(S, B)
        assert torch.equal(bits.to(torch.bool), bits2)
