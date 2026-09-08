"""DCC-KV 参考实现（M0-M1）。

这是基于 work/dcc_kv_reference/dcc_kv_reference.py 的等价 Python 实现，
用于 M2+ 阶段的 CPU-side testing、unit test 和 reference computation。

不依赖 GPU；纯 PyTorch CPU 即可。
"""
from .online_softmax import (
    OnlineSoftmaxState,
    online_softmax_from_attention,
    merge_softmax_states,
    verify_order_invariance,
)
from .representative_query import (
    farthest_point_sampling,
    rademacher_projection,
    select_representative_queries,
)
from .key_selection import (
    rms_per_token_score,
    select_topk_keys,
)
from .calibration import (
    nonneg_least_squares,
    fit_logit_bias,
)
from .value_regression import (
    ridge_regression_value,
    fit_compact_value,
)
from .compact_kv import CompactKV, build_compact_kv

__all__ = [
    # online softmax
    "OnlineSoftmaxState",
    "online_softmax_from_attention",
    "merge_softmax_states",
    "verify_order_invariance",
    # representative query
    "farthest_point_sampling",
    "rademacher_projection",
    "select_representative_queries",
    # key selection
    "rms_per_token_score",
    "select_topk_keys",
    # calibration
    "nonneg_least_squares",
    "fit_logit_bias",
    # value regression
    "ridge_regression_value",
    "fit_compact_value",
    # compact KV
    "CompactKV",
    "build_compact_kv",
]
