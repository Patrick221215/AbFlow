'''
Author: Patrick221215 1427584833@qq.com
Date: 2026-08-17 16:53:28
LastEditors: Patrick221215 1427584833@qq.com
LastEditTime: 2026-09-02 16:51:11
FilePath: /cjm/project/AbFlow/models/modules/am_enc_pair_time.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Compatibility guard for the modern explicit-pair AbFlow backbone.

The modern AMEncoder in am_enc.py owns an explicit z_ij pair state.  Historical
Pair-Time wrappers were designed for sparse AM_E_GCL edge messages and are not
mathematically the same object.  Mixing them silently would create two competing
pair authorities.

The formal modern parent must therefore use:
    ABFLOW_PAIR_TIME_SCOPE=off

AbFlow_model.py imports this class unconditionally but constructs it only when
Pair-Time is enabled, so ordinary off-mode imports remain fully compatible.
"""

from .am_enc import AMEncoder


class AMEncoderPairTime(AMEncoder):
    def __init__(self, *args, pair_time_scope="off", **kwargs):
        raise RuntimeError(
            "AMEncoderPairTime is disabled for the modern explicit single/pair "
            "backbone. Set ABFLOW_PAIR_TIME_SCOPE=off. The modern Pairformer z_ij "
            "is now the pair-representation authority."
        )
