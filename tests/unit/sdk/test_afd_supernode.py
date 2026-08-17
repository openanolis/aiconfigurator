# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Super-node (NVL72-class) fabric tiering for the AFD cross-pool transfer.

A rack-scale system has three fabric tiers, not one: NVLink inside a node,
NVSwitch inside the rack, and a scale-out fabric between racks.
``SystemSpec.get_p2p_bandwidth`` already selected among them, but nothing on
the AFD path reached it -- ``P2P._query_p2p_table`` hardcoded
``inter_node_bw``, so a topology that left the scale-up domain was priced as
if it had not. ``inter_rack_latency`` was declared in gb200/gb300 and read by
nobody.

Covered here:

1. Both tier selectors, at the boundaries, including the no-rack-tier
   fallback.
2. ``inter_rack_latency`` actually reaching the latency term -- a guard
   against it silently becoming a dead field again.
3. ``num_gpus=None`` preserving the legacy flat pricing bit-for-bit, which is
   what every pipeline-parallel caller relies on.
4. The AFD session computing the A+F span and only handing it to the
   cross-pool legs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import AFDTransfer
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator_core.sdk.operations.communication import P2P
from aiconfigurator_core.sdk.system_spec import SystemSpec

pytestmark = pytest.mark.unit


# Mirrors gb200.yaml: 4 GPUs/node, 72/rack, NVLink inside the rack and a
# 9x slower scale-out fabric between racks.
_RACK_NODE_SPEC = {
    "num_gpus_per_node": 4,
    "num_gpus_per_rack": 72,
    "intra_node_bw": 900e9,
    "inter_node_bw": 900e9,
    "inter_rack_bw": 100e9,
    "p2p_latency": 10e-6,
    "inter_rack_latency": 5e-6,
}

# Mirrors h200_sxm.yaml: 8 GPUs/node and no rack tier at all.
_FLAT_NODE_SPEC = {
    "num_gpus_per_node": 8,
    "intra_node_bw": 450e9,
    "inter_node_bw": 50e9,
    "p2p_latency": 10e-6,
}


def _spec(node_spec: dict) -> SystemSpec:
    return SystemSpec({"node": dict(node_spec)})


class _SpecDatabase:
    """Minimal perf DB exposing the two tier selectors used by ``P2P``."""

    def __init__(self, node_spec: dict, mode: common.DatabaseMode = common.DatabaseMode.EMPIRICAL) -> None:
        self.system_spec = _spec(node_spec)
        self._default_database_mode = mode

    def _get_p2p_bandwidth(self, num_gpus: int) -> float:
        return self.system_spec.get_p2p_bandwidth(num_gpus)

    def _get_p2p_latency(self, num_gpus: int) -> float:
        return self.system_spec.get_p2p_latency(num_gpus)

    def query_p2p(self, message_bytes, database_mode=None, num_gpus=None):
        return P2P._query_p2p_table(self, message_bytes, database_mode, num_gpus)


class TestBandwidthTierSelection:
    @pytest.mark.parametrize(
        ("num_gpus", "expected"),
        [
            (1, 900e9),  # inside a node
            (4, 900e9),  # exactly one node
            (5, 900e9),  # crosses nodes, still inside the rack
            (72, 900e9),  # exactly one rack
            (73, 100e9),  # first GPU past the rack
            (144, 100e9),  # two racks
        ],
    )
    def test_bandwidth_tiers_at_boundaries(self, num_gpus, expected):
        assert _spec(_RACK_NODE_SPEC).get_p2p_bandwidth(num_gpus) == expected

    @pytest.mark.parametrize(
        ("num_gpus", "expected"),
        [
            (1, 450e9),
            (8, 450e9),
            (9, 50e9),
            (1000, 50e9),  # no rack tier -> never escalates past inter_node
        ],
    )
    def test_bandwidth_without_rack_tier_never_escalates(self, num_gpus, expected):
        assert _spec(_FLAT_NODE_SPEC).get_p2p_bandwidth(num_gpus) == expected


class TestLatencyTierSelection:
    @pytest.mark.parametrize("num_gpus", [1, 4, 5, 72])
    def test_within_rack_uses_p2p_latency(self, num_gpus):
        assert _spec(_RACK_NODE_SPEC).get_p2p_latency(num_gpus) == 10e-6

    @pytest.mark.parametrize("num_gpus", [73, 144])
    def test_across_racks_uses_inter_rack_latency(self, num_gpus):
        assert _spec(_RACK_NODE_SPEC).get_p2p_latency(num_gpus) == 5e-6

    @pytest.mark.parametrize("num_gpus", [1, 8, 9, 1000])
    def test_without_rack_tier_falls_back_to_p2p_latency(self, num_gpus):
        assert _spec(_FLAT_NODE_SPEC).get_p2p_latency(num_gpus) == 10e-6

    def test_missing_inter_rack_latency_falls_back(self):
        node_spec = {k: v for k, v in _RACK_NODE_SPEC.items() if k != "inter_rack_latency"}
        # Rack tier declared but no latency for it: keep p2p_latency rather
        # than inventing a number.
        assert _spec(node_spec).get_p2p_latency(1000) == 10e-6

    def test_inter_rack_latency_is_actually_consumed(self):
        """Guard against ``inter_rack_latency`` regressing to a dead field.

        Changing only that key must move the cross-rack latency and leave the
        within-rack latency alone.
        """
        slow = dict(_RACK_NODE_SPEC, inter_rack_latency=500e-6)
        baseline = _SpecDatabase(_RACK_NODE_SPEC)
        slowed = _SpecDatabase(slow)

        assert float(slowed.query_p2p(1024, None, 144)) > float(baseline.query_p2p(1024, None, 144))
        assert float(slowed.query_p2p(1024, None, 72)) == float(baseline.query_p2p(1024, None, 72))


class TestFlatPricingPreserved:
    """``num_gpus=None`` must reproduce the pre-tiering formula exactly."""

    @pytest.mark.parametrize("message_bytes", [1, 1024, 1 << 20, 1 << 24])
    def test_none_span_matches_inter_node_formula(self, message_bytes):
        db = _SpecDatabase(_RACK_NODE_SPEC)
        expected = (message_bytes / _RACK_NODE_SPEC["inter_node_bw"] + _RACK_NODE_SPEC["p2p_latency"]) * 1000
        assert float(db.query_p2p(message_bytes)) == pytest.approx(expected, rel=1e-12)

    def test_none_span_ignores_rack_tier(self):
        """A 144-GPU payload priced without a span stays on ``inter_node_bw``.

        This is why the span is opt-in: pipeline-parallel P2P moves between
        two adjacent ranks, so the deployment size is not its span.
        """
        db = _SpecDatabase(_RACK_NODE_SPEC)
        assert float(db.query_p2p(1 << 20)) == float(db.query_p2p(1 << 20, None, 72))
        assert float(db.query_p2p(1 << 20)) != float(db.query_p2p(1 << 20, None, 144))

    def test_sol_mode_also_honors_the_span(self):
        sol = _SpecDatabase(_RACK_NODE_SPEC, mode=common.DatabaseMode.SOL)
        within = float(sol.query_p2p(1 << 24, None, 72))
        across = float(sol.query_p2p(1 << 24, None, 144))
        # SOL drops the latency constant, so the 9x bandwidth gap is exact.
        assert across == pytest.approx(within * 9.0, rel=1e-9)


class _SpanRecordingDatabase:
    """Records the span each ``query_p2p`` call was made with."""

    system_spec: ClassVar[dict] = {"gpu": {"mem_capacity": 80 * (1 << 30)}}

    def __init__(self) -> None:
        self.spans: list[int | None] = []
        self.arg_counts: list[int] = []

    def query_p2p(self, message_bytes, database_mode=None, num_gpus=None):
        self.spans.append(num_gpus)
        self.arg_counts.append(3 if num_gpus is not None else 1)
        return PerformanceResult(latency=float(message_bytes), energy=0.0)


class TestAFDTransferSpan:
    _KW: ClassVar[dict] = {
        "hidden_size": 4096,
        "n_a_workers": 12,
        "n_f_workers": 4,
        "gpus_per_node": 4,
        "f_gpus_per_node": 4,
        "num_experts": 128,
        "topk": 8,
    }

    def test_span_is_forwarded_to_query_p2p(self):
        db = _SpanRecordingDatabase()
        op = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", span_gpus=72, **self._KW)
        op.query(db, x=64)
        assert db.spans == [72]

    def test_no_span_keeps_single_argument_call_shape(self):
        """A stub with a one-argument ``query_p2p`` must still work.

        Passing the span unconditionally would break every existing test
        double, so ``AFDTransfer`` omits it when unset -- the same rule the
        hetero work adopted for ``peer_database``.
        """

        class _OneArgDatabase:
            system_spec: ClassVar[dict] = {"gpu": {"mem_capacity": 80 * (1 << 30)}}

            def query_p2p(self, message_bytes):
                return PerformanceResult(latency=float(message_bytes), energy=0.0)

        op = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", **self._KW)
        assert op.span_gpus is None
        assert float(op.query(_OneArgDatabase(), x=64)) > 0.0

    def test_span_reaches_both_sides_under_hetero(self):
        """Bottleneck pricing and tiering compose: both sides get the span."""
        a_db = _SpanRecordingDatabase()
        f_db = _SpanRecordingDatabase()
        op = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", span_gpus=144, **self._KW)
        op.query(a_db, x=64, peer_database=f_db)
        assert a_db.spans == [144]
        assert f_db.spans == [144]

    @pytest.mark.parametrize("span", [0, None])
    def test_falsy_span_means_flat_pricing(self, span):
        op = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", span_gpus=span, **self._KW)
        assert op.span_gpus is None


class TestHeteroTimesSupernode:
    """The two features are orthogonal and must compose.

    Rack width is a per-system hardware fact, so under hetero A/F the two
    sides can resolve the *same* span to *different* tiers. Bottleneck
    pricing then has to pick the slower of the two tier-resolved latencies,
    not the A side's.
    """

    _KW: ClassVar[dict] = {
        "hidden_size": 4096,
        "n_a_workers": 68,
        "n_f_workers": 4,
        "gpus_per_node": 4,
        "f_gpus_per_node": 8,
        "num_experts": 128,
        "topk": 8,
    }

    def _op(self, span):
        return AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", span_gpus=span, **self._KW)

    def test_each_side_resolves_its_own_tier(self):
        """A=gb200 (rack=72) and F=h200_sxm (no rack tier) at span 144.

        The A side escalates to its 100GB/s inter-rack link; the F side has no
        rack tier declared and stays on its 50GB/s inter-node link. The F side
        is the slower one, so it must set the price.
        """
        a_db = _SpecDatabase(_RACK_NODE_SPEC)
        f_db = _SpecDatabase(_FLAT_NODE_SPEC)
        assert a_db._get_p2p_bandwidth(144) == 100e9
        assert f_db._get_p2p_bandwidth(144) == 50e9  # no rack tier -> inter_node

        op = self._op(144)
        priced = float(op.query(a_db, x=1024, peer_database=f_db))
        a_only = float(op.query(a_db, x=1024))
        f_only = float(op.query(f_db, x=1024))

        assert priced == pytest.approx(max(a_only, f_only))
        assert priced == pytest.approx(f_only)  # F is the bottleneck here
        assert priced > a_only

    def test_bottleneck_can_be_either_side(self):
        """Symmetry: swapping which DB is primary must not change the price."""
        a_db = _SpecDatabase(_RACK_NODE_SPEC)
        f_db = _SpecDatabase(_FLAT_NODE_SPEC)
        op = self._op(144)
        forward = float(op.query(a_db, x=1024, peer_database=f_db))
        reverse = float(op.query(f_db, x=1024, peer_database=a_db))
        assert forward == pytest.approx(reverse)

    def test_tier_still_applies_when_only_one_side_crosses_a_rack(self):
        """The A side escalates, but the F side is the bottleneck anyway.

        The F side (50 GB/s, no rack tier) is always slower than the A side's
        100 GB/s (cross-rack), so ``max(a, f)`` picks F regardless. But if
        we query the A side alone, the span *does* change its price -- proving
        the tier is live even though bottleneck pricing hides it.
        """
        a_db = _SpecDatabase(_RACK_NODE_SPEC)

        a_within = float(self._op(72).query(a_db, x=4096))
        a_across = float(self._op(144).query(a_db, x=4096))
        # The A side's own price must change at the rack boundary (the
        # gb200.yaml crossover issue causes the sign to depend on payload,
        # so just assert they're different -- the direction was validated by
        # TestBandwidthTierSelection already).
        assert a_within != a_across, "A-only pricing must react to the rack boundary"

    def test_homogeneous_span_pricing_is_unaffected_by_a_same_object_peer(self):
        """Passing the same DB as peer is a no-op, span or not."""
        db = _SpecDatabase(_RACK_NODE_SPEC)
        op = self._op(144)
        assert float(op.query(db, x=1024, peer_database=db)) == float(op.query(db, x=1024))


class TestSessionSpanWiring:
    """The session owns span derivation; the op only carries it."""

    def _comm_ops(self, *, n_a_nodes, n_f_nodes, a_gpus_per_node=4, f_gpus_per_node=4):
        from aiconfigurator.sdk.config import AFDConfig
        from aiconfigurator.sdk.inference_session import AFDInferenceSession

        cfg = AFDConfig(
            n_a_nodes=n_a_nodes,
            n_f_nodes=n_f_nodes,
            gpus_per_node=a_gpus_per_node,
            a_gpus_per_node=a_gpus_per_node,
            f_gpus_per_node=f_gpus_per_node,
            tp_a=1,
            f_moe_ep_size=1,
        )
        session = AFDInferenceSession.__new__(AFDInferenceSession)
        session._afd_config = cfg
        session._a_model_config = SimpleNamespace(comm_quant_mode=common.CommQuantMode.half)
        model = SimpleNamespace(_hidden_size=4096, _num_experts=128, _topk=8)
        return session._build_afd_comm_ops(model, model)

    @pytest.mark.parametrize(
        ("n_a_nodes", "n_f_nodes", "expected_span"),
        [
            (3, 1, 16),  # 4 nodes x 4 GPUs -- comfortably inside one rack
            (17, 1, 72),  # exactly the NVL72 domain, FastAFD's largest ratio
            (18, 1, 76),  # one node past it -- now a cross-rack deployment
        ],
    )
    def test_span_is_total_a_plus_f_gpus(self, n_a_nodes, n_f_nodes, expected_span):
        ops = self._comm_ops(n_a_nodes=n_a_nodes, n_f_nodes=n_f_nodes)
        assert ops.a2f.span_gpus == expected_span
        assert ops.f2a.span_gpus == expected_span

    def test_span_uses_per_pool_node_widths(self):
        """Under hetero A/F the two pools can have different node widths."""
        ops = self._comm_ops(n_a_nodes=2, n_f_nodes=1, a_gpus_per_node=4, f_gpus_per_node=8)
        assert ops.a2f.span_gpus == 2 * 4 + 1 * 8

    def test_only_cross_pool_legs_carry_a_span(self):
        """F-side AG/RS is intra-node and a_combine is a local HBM reduce.

        Neither can cross a rack, so neither takes a span -- giving them one
        would price a node-local collective on the scale-out fabric.
        """
        ops = self._comm_ops(n_a_nodes=17, n_f_nodes=1)
        for op in (ops.f_ag, ops.f_rs, ops.a_combine):
            assert not hasattr(op, "span_gpus")
