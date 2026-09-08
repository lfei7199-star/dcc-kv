"""M2 / Phase B.3: 变长消息序列化的 CPU 实现。

变长消息（DCC-KV 每条边 budget 不同）的打包/解包/发送/接收：
- 打包：把 [B0, d] + [B1, d] + ... 拼接成 [sum(B), d]
- 解包：按 recv_sizes 切分
- 模拟 All-to-Allv：在 2 进程上手动实现，验证对端能正确收到

不需要 GPU，CPU 即可。
"""
from __future__ import annotations

from typing import List, Tuple
import torch


def pack_var_len_messages(
    messages: List[torch.Tensor],
    sizes: List[int],
) -> Tuple[torch.Tensor, List[int]]:
    """变长消息打包。

    Args:
        messages: 每条消息是 [n_i, d]（n_i 可以不同）
        sizes: 每条消息的 n_i（必须等于 messages[i].shape[0]）

    Returns:
        packed: [sum(sizes), d] 拼接后的张量
        offsets: 累积偏移 [o_0, o_1, ...]，o_{i+1} = o_i + sizes[i]
    """
    assert len(messages) == len(sizes)
    for m, s in zip(messages, sizes):
        assert m.shape[0] == s, f"message size {m.shape[0]} != expected {s}"
    packed = torch.cat(messages, dim=0)
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    return packed, offsets


def unpack_var_len_messages(
    packed: torch.Tensor,
    sizes: List[int],
) -> List[torch.Tensor]:
    """变长消息解包。

    Args:
        packed: [sum(sizes), d]
        sizes: 每条消息的 n_i

    Returns:
        messages: List of [n_i, d]
    """
    assert packed.shape[0] == sum(sizes), f"packed size {packed.shape[0]} != sum {sum(sizes)}"
    messages = []
    offset = 0
    for s in sizes:
        messages.append(packed[offset:offset + s])
        offset += s
    return messages


def mock_all_to_allv(
    rank: int,
    world_size: int,
    send_messages: List[torch.Tensor],
    send_sizes: List[int],
    peer_send_data: dict,
) -> List[torch.Tensor]:
    """Mock All-to-Allv：单进程模拟（不调 distributed）。

    用途：单元测试 receive 端逻辑，无需 mp.spawn。

    Args:
        rank: 当前 rank
        world_size: 总 rank 数
        send_messages: 发给各 rank 的消息
        send_sizes: 发给各 rank 的消息大小
        peer_send_data: {src_rank: (packed_tensor, sizes)} 模拟其他 rank 发来的数据

    Returns:
        recv_messages: 从各 rank 收到的消息（按 rank 索引）
    """
    assert len(send_messages) == world_size
    assert len(send_sizes) == world_size
    # 接收部分
    recv_messages = []
    for src in range(world_size):
        if src == rank:
            # 自己是发送方
            recv_messages.append(send_messages[rank])
        else:
            # 从 peer_send_data 拿
            if src not in peer_send_data:
                raise ValueError(f"Missing mock data for src={src}")
            packed, sizes = peer_send_data[src]
            messages = unpack_var_len_messages(packed, sizes)
            recv_messages.append(messages[rank])
    return recv_messages


def test_round_trip_var_len():
    """单元测试：打包 → 模拟发送 → 解包，结果一致。"""
    # 2 rank, 各发 2 条变长消息
    msg_0_to_0 = torch.tensor([1.0, 2.0])  # rank 0 → rank 0
    msg_0_to_1 = torch.tensor([10.0, 11.0, 12.0])  # rank 0 → rank 1
    msg_1_to_0 = torch.tensor([100.0, 101.0, 102.0])  # rank 1 → rank 0
    msg_1_to_1 = torch.tensor([200.0, 201.0])  # rank 1 → rank 1

    # rank 0 的 send
    rank_0_send = [msg_0_to_0, msg_0_to_1]
    rank_0_sizes = [2, 3]
    rank_0_packed, _ = pack_var_len_messages(rank_0_send, rank_0_sizes)

    # rank 1 的 send
    rank_1_send = [msg_1_to_0, msg_1_to_1]
    rank_1_sizes = [3, 2]
    rank_1_packed, _ = pack_var_len_messages(rank_1_send, rank_1_sizes)

    # rank 0 收到的：来自 rank 0 (msg_0_to_0) + 来自 rank 1 (msg_1_to_0)
    peer_data_0 = {1: (rank_1_packed, rank_1_sizes)}
    rank_0_recv = mock_all_to_allv(
        rank=0, world_size=2,
        send_messages=rank_0_send,
        send_sizes=rank_0_sizes,
        peer_send_data=peer_data_0,
    )
    assert torch.allclose(rank_0_recv[0], msg_0_to_0)
    assert torch.allclose(rank_0_recv[1], msg_1_to_0)

    # rank 1 收到的
    peer_data_1 = {0: (rank_0_packed, rank_0_sizes)}
    rank_1_recv = mock_all_to_allv(
        rank=1, world_size=2,
        send_messages=rank_1_send,
        send_sizes=rank_1_sizes,
        peer_send_data=peer_data_1,
    )
    assert torch.allclose(rank_1_recv[0], msg_0_to_1)
    assert torch.allclose(rank_1_recv[1], msg_1_to_1)
