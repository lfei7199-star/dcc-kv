"""统一启动器：单进程直接调用，多进程 mp.spawn。

设计目标：
- CPU 上用 gloo backend 跑（M2 阶段验证分布式接口形状）
- GPU 上自动切 nccl backend（M3+ 阶段）
- 接口与 torchrun 等价
- 异常能正确传播到主进程

用法 1（推荐，函数式）：
    from src.distributed import launch_dist, setup_distributed, cleanup_distributed

    def my_fn(rank, args):
        setup_distributed(rank, nproc, backend="gloo")
        # ... do work ...
        cleanup_distributed()

    launch_dist(my_fn, nproc_per_node=2, args=("hello",), backend="gloo")

用法 2（CLI）：
    python -m src.distributed.launch_dist --nproc 2 --backend gloo my_script.py
"""
from __future__ import annotations

import os
import sys
import socket
import logging
import argparse
import traceback
from typing import Callable, Any, Optional

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 全局：跟踪是否在 worker 进程内（供 comm.py 等模块判断）
# -----------------------------------------------------------------------------
_WORKER_RANK: Optional[int] = None
_WORKER_WORLD_SIZE: Optional[int] = None


def get_worker_rank() -> Optional[int]:
    """返回当前 worker 进程的 rank（主进程中为 None）。"""
    return _WORKER_RANK


def get_worker_world_size() -> Optional[int]:
    """返回当前 worker 进程的 world_size（主进程中为 None）。"""
    return _WORKER_WORLD_SIZE


# -----------------------------------------------------------------------------
# 分布式初始化 / 清理
# -----------------------------------------------------------------------------
def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "gloo",
    master_addr: Optional[str] = None,
    master_port: Optional[str] = None,
    timeout_minutes: int = 30,
) -> None:
    """初始化 torch.distributed。

    Args:
        rank: 当前进程 rank
        world_size: 总进程数
        backend: "gloo" (CPU) / "nccl" (GPU)
        master_addr: master 节点地址（默认读环境变量）
        master_port: master 端口（默认读环境变量）
        timeout_minutes: 集合通信超时（分钟）

    Raises:
        RuntimeError: 如果已经初始化或环境变量缺失
    """
    global _WORKER_RANK, _WORKER_WORLD_SIZE

    if dist.is_initialized():
        logger.warning("torch.distributed already initialized; skipping init")
        _WORKER_RANK = rank
        _WORKER_WORLD_SIZE = world_size
        return

    # 设置 master 地址
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        os.environ["MASTER_PORT"] = master_port

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        # 默认端口；如有冲突会失败，再让用户指定
        os.environ["MASTER_PORT"] = "29500"

    timeout = datetime.timedelta(minutes=timeout_minutes)
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )

    _WORKER_RANK = rank
    _WORKER_WORLD_SIZE = world_size
    logger.info(
        f"[rank {rank}/{world_size}] initialized on {socket.gethostname()} "
        f"(backend={backend}, master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']})"
    )


def cleanup_distributed() -> None:
    """销毁 torch.distributed。每个 worker 进程结束时调用。"""
    if dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# 启动器主函数
# -----------------------------------------------------------------------------
def launch_dist(
    fn: Callable,
    nproc_per_node: int = 1,
    args: Any = None,
    backend: str = "gloo",
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
    timeout_minutes: int = 30,
) -> Any:
    """统一启动器。

    单进程：直接调用 fn(0, args)，返回 fn 的结果
    多进程：mp.spawn 启动 nproc_per_node 个进程，分别调用 fn(rank, args)

    Args:
        fn: Callable，签名 fn(rank, args) -> Any
            rank 是当前进程 rank，args 是用户传入的参数（每个进程共享）
        nproc_per_node: 每节点进程数
        args: 传给 fn 的第二个参数
        backend: 通信 backend（gloo/nccl）
        master_addr: master 节点地址
        master_port: master 端口
        timeout_minutes: 集合通信超时

    Returns:
        fn 的返回值（仅单进程模式；多进程模式返回 None）

    Raises:
        RuntimeError: 任何子进程的异常都会被传播到主进程
    """
    if nproc_per_node < 1:
        raise ValueError(f"nproc_per_node must be >= 1, got {nproc_per_node}")

    # 单进程模式：直接调用
    if nproc_per_node == 1:
        logger.info("Launching in single-process mode (no distributed init)")
        return fn(0, args)

    # 多进程模式：mp.spawn
    # 设置 master 地址
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # 用 Manager 收集子进程异常
    error_queue: mp.Queue = mp.Queue()

    def wrapped_fn(rank, args_tuple):
        """mp.spawn 包装：捕获异常并放入 error_queue。"""
        (user_args, world_size, be, timeout_min) = args_tuple
        try:
            setup_distributed(
                rank=rank,
                world_size=world_size,
                backend=be,
                timeout_minutes=timeout_min,
            )
            result = fn(rank, user_args)
            cleanup_distributed()
            return result
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[rank {rank}] exception:\n{tb}")
            error_queue.put((rank, str(e), tb))
            cleanup_distributed()
            # 不 raise，让 mp.spawn 正常 join；主进程从 error_queue 拿到异常

    logger.info(
        f"Launching {nproc_per_node} processes via mp.spawn "
        f"(backend={backend}, master={master_addr}:{master_port})"
    )

    try:
        mp.spawn(
            wrapped_fn,
            args=(args, nproc_per_node, backend, timeout_minutes),
            nprocs=nproc_per_node,
            join=True,
            daemon=False,
        )
    except Exception as e:
        raise RuntimeError(f"mp.spawn failed: {e}") from e

    # 检查子进程异常
    if not error_queue.empty():
        rank, msg, tb = error_queue.get()
        raise RuntimeError(
            f"Worker rank {rank} raised exception:\n{msg}\n\nTraceback:\n{tb}"
        )


# -----------------------------------------------------------------------------
# CLI 入口
# -----------------------------------------------------------------------------
def main() -> None:
    """CLI 入口：python -m src.distributed.launch_dist --nproc 2 my_script.py"""
    parser = argparse.ArgumentParser(
        description="DCC-KV distributed launcher (CPU-friendly gloo / GPU nccl)"
    )
    parser.add_argument("--nproc", type=int, default=1, help="processes per node")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--script", type=str, required=True, help="Python script to run")
    parser.add_argument(
        "--script-args", type=str, nargs="*", default=[],
        help="Extra args passed to the script"
    )
    args, _unknown = parser.parse_known_args()

    if not os.path.isfile(args.script):
        raise FileNotFoundError(f"Script not found: {args.script}")

    # 简化的脚本执行：exec script 的 main
    # 实际项目里这里会 dispatch 到不同的 entry point
    script_globals = {
        "__name__": "__main__",
        "__file__": args.script,
        "launch_dist": launch_dist,
        "setup_distributed": setup_distributed,
        "cleanup_distributed": cleanup_distributed,
    }
    script_globals["sys"] = sys
    script_globals["os"] = os
    script_globals["argparse"] = __import__("argparse")
    script_globals["argparse"].sys.argv = [args.script] + args.script_args

    with open(args.script) as f:
        exec(compile(f.read(), args.script, "exec"), script_globals)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()


# 需要在文件顶部 import（避免 setup_distributed 中 NameError）
import datetime
