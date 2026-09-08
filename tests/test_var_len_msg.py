"""变长消息序列化测试。"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distributed.var_len_msg import (
    pack_var_len_messages,
    unpack_var_len_messages,
    mock_all_to_allv,
    test_round_trip_var_len,
)


class TestVarLenMsg:
    """变长消息测试。"""

    def test_round_trip_1d(self):
        """1D 消息打包/解包。"""
        msgs = [torch.tensor([1.0, 2.0]), torch.tensor([3.0]), torch.tensor([4.0, 5.0, 6.0])]
        sizes = [2, 1, 3]
        packed, offsets = pack_var_len_messages(msgs, sizes)
        assert packed.shape == (6,)
        assert offsets == [0, 2, 3, 6]
        unpacked = unpack_var_len_messages(packed, sizes)
        for orig, back in zip(msgs, unpacked):
            assert torch.allclose(orig, back)

    def test_round_trip_2d(self):
        """2D 消息（[N, head_dim]）打包/解包。"""
        d = 64
        msgs = [
            torch.randn(2, d),
            torch.randn(3, d),
            torch.randn(5, d),
        ]
        sizes = [2, 3, 5]
        packed, offsets = pack_var_len_messages(msgs, sizes)
        assert packed.shape == (10, d)
        assert offsets == [0, 2, 5, 10]
        unpacked = unpack_var_len_messages(packed, sizes)
        for orig, back in zip(msgs, unpacked):
            assert torch.allclose(orig, back)

    def test_size_mismatch_raises(self):
        """messages 长度 != sizes 时报错。"""
        with pytest.raises(AssertionError):
            pack_var_len_messages(
                [torch.tensor([1.0, 2.0]), torch.tensor([3.0])],
                sizes=[2, 3],  # 错：第二条只有 1 个
            )

    def test_mock_all_to_allv_2proc(self):
        """2 进程 mock All-to-Allv 端到端。"""
        msg_0_to_0 = torch.tensor([1.0, 2.0])
        msg_0_to_1 = torch.tensor([10.0, 11.0, 12.0])
        msg_1_to_0 = torch.tensor([100.0, 101.0, 102.0])
        msg_1_to_1 = torch.tensor([200.0, 201.0])

        rank_0_send = [msg_0_to_0, msg_0_to_1]
        rank_0_sizes = [2, 3]
        rank_0_packed, _ = pack_var_len_messages(rank_0_send, rank_0_sizes)

        rank_1_send = [msg_1_to_0, msg_1_to_1]
        rank_1_sizes = [3, 2]
        rank_1_packed, _ = pack_var_len_messages(rank_1_send, rank_1_sizes)

        # rank 0 收到
        peer_data_0 = {1: (rank_1_packed, rank_1_sizes)}
        rank_0_recv = mock_all_to_allv(
            rank=0, world_size=2,
            send_messages=rank_0_send,
            send_sizes=rank_0_sizes,
            peer_send_data=peer_data_0,
        )
        assert torch.allclose(rank_0_recv[0], msg_0_to_0)
        assert torch.allclose(rank_0_recv[1], msg_1_to_0)

        # rank 1 收到
        peer_data_1 = {0: (rank_0_packed, rank_0_sizes)}
        rank_1_recv = mock_all_to_allv(
            rank=1, world_size=2,
            send_messages=rank_1_send,
            send_sizes=rank_1_sizes,
            peer_send_data=peer_data_1,
        )
        assert torch.allclose(rank_1_recv[0], msg_0_to_1)
        assert torch.allclose(rank_1_recv[1], msg_1_to_1)

    def test_smoke_round_trip(self):
        """test_round_trip_var_len 自带的烟雾测试。"""
        test_round_trip_var_len()
