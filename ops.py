import jax
from typing import Optional
import functools
from jax import core
import numpy as np
import jax.numpy as jnp
from jax._src.pallas.triton.lowering import register_lowering, LoweringRuleContext
from jax._src.lib.triton import dialect as tt_dialect
from jax.experimental import pallas as pl
from jax._src.lib.mlir.dialects import hlo
from jax._src.util import subvals
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
import logging
from jax._src.pallas.triton.lowering import _full, _associative_scan_lowering


triton_scan_p = core.Primitive("triton_scan")
triton_scan_p.multiple_results = True

def triton_scan_abstract_eval(*flat_args, flat_fn, axis, reverse):

    return [core.ShapedArray(x.shape, x.dtype) for x in flat_args]

triton_scan_p.def_abstract_eval(triton_scan_abstract_eval)

import jax.core as jax_core
from jax import tree_util
from jax._src import api_util
from jax._src.interpreters import partial_eval as pe
from jax._src import linear_util as lu
from jax._src.pallas.triton.lowering import _element_type, lower_jaxpr_to_triton_ir

def _associative_scan_lowering(flat_fun, ctx: LoweringRuleContext, flat_args, axes, reverse):
  # flat_args = tree_util.tree_leaves(args)
  # flat_args, tree_def = jax.tree.flatten(args)
  (axis,) = axes
  # dtype = ctx.avals_in[0].dtype
  # in_avals = [
  #     jax_core.ShapedArray((), dtype=dtype),
  #     jax_core.ShapedArray((), dtype=dtype),
  # ]
  # assert axis == 0
  # in_avals = [
      # jax_core.ShapedArray(x.shape[1:], dtype=x.dtype) for x in ctx.avals_in
  # ] * 2
  in_avals = [
      jax_core.ShapedArray((), dtype=x.dtype) for x in ctx.avals_in
  ] * 2
  # in_tree = tree_util.tree_structure((args, args))
  # flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      # lu.wrap_init(body), in_tree
  # )
  combine_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      flat_fun, in_avals
  )
  # out_tree = out_tree_thunk()
  # del out_tree  # Not needed
  if consts:
    raise NotImplementedError("Associative scan with constants not supported.")
  element_types = [_element_type(arg.type) for arg in flat_args]
  # element_types = [ir.RankedTensorType.get(shape=arg.type.shape[1:], element_type=arg.type.element_type) for arg in flat_args]
  scan_op = tt_dialect.ScanOp(flat_args, axis, reverse=reverse)
  param_types = element_types * 2
  entry = scan_op.regions[0].blocks.append(*param_types)
  with ir.InsertionPoint.at_block_begin(entry):
    results = lower_jaxpr_to_triton_ir(
        ctx.context, combine_jaxpr, None, *entry.arguments
    )
    tt_dialect.scan_return(results)
  scan_op.verify()
  result = list(scan_op.result)
  # result = jax.tree.unflatten(tree_def, result)
  # return list(scan_op.result)
  return result

@register_lowering(triton_scan_p)
def triton_scan_lowering(ctx: LoweringRuleContext, *flat_args, flat_fn, axis, reverse):
    result = _associative_scan_lowering(flat_fn, ctx, flat_args, (axis,), reverse)
    return result

def triton_scan(fn, args, axis, init=None, reverse=False):
    flat_args, tree_def = jax.tree.flatten(args)
    in_tree = tree_util.tree_structure((args, args))
    flat_fn, out_tree_thunk = api_util.flatten_fun_nokwargs(
        lu.wrap_init(fn), in_tree
    )

    flat_out = triton_scan_p.bind(*flat_args, flat_fn=flat_fn, axis=axis, reverse=reverse)
    out = jax.tree.unflatten(tree_def, flat_out)
    if init is not None:
        def expand_shape(src, ref):
            assert src.ndim == ref.ndim - 1, (src.ndim, ref.ndim)
            target_shape = list(src.shape)
            pos_axis = axis if axis >= 0 else axis + ref.ndim
            target_shape.insert(pos_axis, ref.shape[axis])
            return jnp.broadcast_to(jnp.expand_dims(src, pos_axis), target_shape)
        init = jax.tree_map(lambda src, ref: expand_shape(src, ref), init, args)
        # if reverse:
            # It could be non commutative
            # out = fn(out, init)
        # else:
        out = fn(init, out)
    return out

# Join last
join_last_p = core.Primitive("join_last")

def join_last(x, y):
    return join_last_p.bind(x, y)

def join_last_abstract_eval(x, y):
    assert x.shape == y.shape
    return core.ShapedArray(x.shape + (2,), x.dtype)

join_last_p.def_abstract_eval(join_last_abstract_eval)


@register_lowering(join_last_p)
def join_last_lowering(ctx: LoweringRuleContext, x, y):
    return tt_dialect.join(x, y)

def join_last_lowering_mlir(ctx, x, y):
    x, y = [mlir.reshape(ctx, entry, core.ShapedArray(x.type.shape + [1], ctx.avals_out[0].dtype)) for entry in (x, y)]
    result = [hlo.concatenate([x, y], mlir.i64_attr(x.type.rank - 1))]
    return result

mlir.register_lowering(join_last_p, join_last_lowering_mlir)



def join(x: jax.Array, y: jax.Array, axis: int, interleave: bool = False):

    if axis < 0:
        axis += x.ndim
    out = join_last(x, y)
    # assert x.shape[axis] % 2 == 0
    permutation = list(range(x.ndim))
    if interleave:
        permutation.insert(axis + 1, x.ndim)
    else:
        permutation.insert(axis, x.ndim)
    out = jnp.transpose(out, permutation)

    new_shape = list(x.shape)
    new_shape[axis] = 2 * new_shape[axis]

    out = jnp.reshape(out, new_shape)
      
    return out

def cumsum_kernel(
    init_ref,
    src_ref,
    dst_ref,
    axis: int,
    reverse: bool
):
    init = init_ref[:].astype(jnp.float32)
    src = src_ref[:].astype(jnp.float32)
    result = triton_scan(jax.lax.add, src, init=init, axis=axis, reverse=reverse)
    dst_ref[:] = result.astype(dst_ref.dtype)



def cumsum_pallas(
    src,
    axis,
    init: Optional[jax.Array] = None,
    reverse: bool = False,
    out_dtype: Optional[jax.typing.DTypeLike] = None,
    max_items: int = 16384,
    num_warps: int = 4
):
    ori_shape = src.shape
    pre_shape, T, post_shape = src.shape[:axis], src.shape[axis], src.shape[axis + 1:]

    N = int(np.rint(np.prod(pre_shape)))
    D = int(np.rint(np.prod(post_shape)))

    src = src.reshape(N, T, D)

    if init is not None:
        assert init.shape == pre_shape + post_shape
        init = init.reshape(N, D)
    else:
        init = jnp.zeros((N, D), dtype=src.dtype)


    if out_dtype is None:
        out_dtype = src.dtype



    assert max_items >= T, 'We only support cumsum of maximum length max_items'
    assert max_items % T == 0
    block_size_d = min(max_items // T, D)

    if block_size_d < D:
        assert block_size_d >= 8, 'This is dangerous man. Try using a smaller T'

    assert T * block_size_d <= max_items
    assert D % block_size_d == 0
    num_blocks_d = D // block_size_d

    block_size_n = min(max_items // (T * block_size_d), N)
    assert N % block_size_n == 0
    num_blocks_n = N // block_size_n

    # print(block_size_n, block_size_d)
    # print(num_blocks_n, num_blocks_d)

    func = pl.pallas_call(
        functools.partial(cumsum_kernel, axis=1, reverse=reverse),
        out_shape=jax.ShapeDtypeStruct(src.shape, out_dtype),
        in_specs=[
            pl.BlockSpec(lambda n, d: (n, d), (block_size_n, block_size_d)),
            pl.BlockSpec(lambda n, d: (n, 0, d), (block_size_n, T, block_size_d))
        ],
        out_specs=pl.BlockSpec(lambda n, d: (n, 0, d), (block_size_n, T, block_size_d)),
        grid=(num_blocks_n, num_blocks_d),
        interpret=False,
        compiler_params=dict(
            num_warps=num_warps
        )
    )

    result = func(init, src)
    result = result.reshape(ori_shape)
    return result


# @functools.partial(jax.jit, static_argnames=('block_size_t', 'axis', 'reverse'))
def cumsum_recursive(
    src,
    axis: int,
    init: Optional[jax.Array] = None,
    reverse: bool = False,
    block_size_t: int = 2048,
):
    """
    block size t: 1024 or 2048 is appropriate. 128 too small (too many recursion
    levels) and 16384 too big (too much non-contiguous loads)
    best value depends looks like nvidia gpus l1 cache line size is 128 bytes
    """
    ori_shape = src.shape
    pre_shape, T, post_shape = src.shape[:axis], src.shape[axis], src.shape[axis + 1:]

    N = int(np.rint(np.prod(pre_shape)))
    D = int(np.rint(np.prod(post_shape)))

    src = src.reshape(N, T, D)
    if init is not None:
        assert init.shape == pre_shape + post_shape
        init = init.reshape(N, D)

    N, T, D = src.shape
    def func(src, init=None):
        N, L, D = src.shape
        if L <= block_size_t:
            return cumsum_pallas(src, axis=1, init=init, reverse=reverse)
        else:
            assert L % block_size_t == 0
            num_blocks = L // block_size_t
            src = src.reshape(N, num_blocks, block_size_t, D)
            # (N, num_blocks, D)
            block_item = jnp.zeros((N, num_blocks, D), dtype=src.dtype)
            init_index = -1 if reverse else 0
            rest_slice = slice(0, -1) if reverse else slice(1, None)
            init_slice = slice(1, None) if reverse else slice(0, -1)

            if init is not None:
                block_item = block_item.at[:, init_index, :].set(init)
            block_item = block_item.at[:, rest_slice, :].set(src.sum(axis=-2)[:, init_slice, :])
            # (N, num_blocks, D)
            block_init = func(block_item)
            # src: (N, num_blocks, block_size_t, D)
            dst = cumsum_pallas(src, axis=-2, init=block_init, reverse=reverse)
            dst = dst.reshape(N, L, D)
            return dst

    dst = func(src, init=init)
    dst = dst.reshape(ori_shape)
    return dst

def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear rnn. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    # Note we use log space
    return A_j + A_i, jnp.exp(A_j) * b_i + b_j

def rnn_kernel(
    init_log_fgate_ref,
    init_kv_ref,
    log_fgate_ref,
    kv_ref,

    log_lambda_ref,
    out_ref,
    reverse: bool,
    last_only: bool
):

    T = kv_ref.shape[0]
    kv = kv_ref[:].astype(jnp.float32)
    log_fgate = log_fgate_ref[:].astype(jnp.float32)
    init_kv = init_kv_ref[:].astype(jnp.float32)
    init_log_fgate = init_log_fgate_ref[:].astype(jnp.float32)
    init_log_fgate = jnp.broadcast_to(init_log_fgate[:, None], init_kv.shape)
    log_fgate = jnp.broadcast_to(log_fgate[:, :, None], kv.shape)
    log_lambda, state = triton_scan(
                           binary_operator, (log_fgate, kv),
                           axis=0,
                           init=(init_log_fgate, init_kv),
                           reverse=reverse)

    # Same as indexing at 0 for obvious reasons
    log_lambda = jnp.max(log_lambda, axis=-1).astype(log_lambda_ref.dtype)
    state = state.astype(out_ref.dtype)
    if not last_only:
        log_lambda_ref[:] = log_lambda
        out_ref[:] = state
    else:
        mask = jnp.arange(T)
        mask = (mask <= 0) if reverse else (mask >= T - 1)
        pl.store(log_lambda_ref, (slice(None),), log_lambda, mask=mask[:, None])
        pl.store(out_ref, (slice(None),), state, mask=mask[:, None, None])



def rnn_pallas(
    log_fgate,
    kv,
    init_log_fgate: Optional[jax.Array] = None,
    init_kv: Optional[jax.Array] = None,
    reverse: bool = False,
    last_only: bool = False
):

    T, K, V = kv.shape
    assert log_fgate.shape == (T, K)


    assert (init_log_fgate is None) == (init_kv is None)
    if init_log_fgate is None:
        init_log_fgate = jnp.zeros_like(log_fgate[0])
        init_kv = jnp.zeros_like(kv[0])

    max_items = 8192
    assert T <= max_items
    assert max_items % T == 0
    block_size_v = min(max_items // T, V)
    assert V % block_size_v == 0
    num_blocks_v = V // block_size_v

    block_size_k = min(max_items // (T * block_size_v), K)
    assert K % block_size_k == 0
    num_blocks_k = K // block_size_k

    # print(block_size_n, block_size_d)
    # print(num_blocks_n, num_blocks_d)

    func = pl.pallas_call(
        functools.partial(rnn_kernel, reverse=reverse, last_only=last_only),
        out_shape=[
            jax.ShapeDtypeStruct(log_fgate.shape, log_fgate.dtype),
            jax.ShapeDtypeStruct(kv.shape, kv.dtype),
        ],
        in_specs=[
            pl.BlockSpec(lambda k, v: (k), (block_size_k,)),
            pl.BlockSpec(lambda k, v: (k, v), (block_size_k, block_size_v)),
            pl.BlockSpec(lambda k, v: (0, k), (T, block_size_k)),
            pl.BlockSpec(lambda k, v: (0, k, v), (T, block_size_k, block_size_v)),
        ],
        out_specs=[
            pl.BlockSpec(lambda k, v: (0, k), (T, block_size_k)),
            pl.BlockSpec(lambda k, v: (0, k, v), (T, block_size_k, block_size_v)),
        ],
        grid=(num_blocks_k, num_blocks_v),
        interpret=False,
        compiler_params=dict(
            num_warps=4
        ),
    )

    log_lambda, state = func(init_log_fgate, init_kv, log_fgate, kv)
    return log_lambda, state

def rnn_recursive(
    log_fgate,
    kv,
    init_log_fgate: Optional[jax.Array] = None,
    init_kv: Optional[jax.Array] = None,
    block_size_t: int = 256,
    reverse: bool = False
):
    T, K, V = kv.shape
    assert log_fgate.shape == (T, K)
    def func(log_fgate, kv, init_log_fgate=None, init_kv=None):
        assert (init_log_fgate is None) == (init_kv is None)
        L, K, V = kv.shape
        if L <= block_size_t:
            return rnn_pallas(
                log_fgate,
                kv,
                init_log_fgate=init_log_fgate,
                init_kv=init_kv,
                reverse=reverse
            )
        else:
            assert L % block_size_t == 0
            num_blocks = L // block_size_t
            log_fgate = log_fgate.reshape(num_blocks, block_size_t, K)
            kv = kv.reshape(num_blocks, block_size_t, K, V)
            # (N, num_blocks, D)
            # Note we use float32 here
            block_log_fgate = jnp.zeros((num_blocks, K), dtype=jnp.float32)
            block_kv = jnp.zeros((num_blocks, K, V), dtype=jnp.float32)
            init_index = -1 if reverse else 0
            last_index = 0 if reverse else -1
            rest_slice = slice(0, -1) if reverse else slice(1, None)
            init_slice = slice(1, None) if reverse else slice(0, -1)

            # We actually want to do a reduction, not cumsum
            log_lambda, state = jax.vmap(functools.partial(rnn_pallas, reverse=reverse, last_only=True))(
                log_fgate,
                kv,
            )
            reduced_log_fgate = log_lambda[:, last_index]
            reduced_kv = state[:, last_index]

            if init_kv is not None:
                block_log_fgate = block_log_fgate.at[init_index].set(init_log_fgate)
                block_kv = block_kv.at[init_index].set(init_kv)
            block_log_fgate = block_log_fgate.at[rest_slice].set(reduced_log_fgate[init_slice])
            block_kv = block_kv.at[rest_slice].set(reduced_kv[init_slice])
            # (N, num_blocks, D)
            block_init_log_fgate, block_init_kv = func(block_log_fgate, block_kv)
            # src: (N, num_blocks, block_size_t, D)
            log_lambda, state = jax.vmap(functools.partial(rnn_pallas, reverse=reverse))(
                log_fgate,
                kv,
                init_log_fgate=block_init_log_fgate,
                init_kv=block_init_kv
            )
            log_lambda = log_lambda.reshape(L, K)
            state = state.reshape(L, K, V)
            return log_lambda, state
    return func(log_fgate, kv, init_log_fgate, init_kv)


if __name__ == '__main__':
    pass
