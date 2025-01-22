import jax.numpy as jnp
from jaxtyping import Array, DTypeLike


def combine_masks(*masks, dtype: DTypeLike = jnp.float32):
    """Combine attention masks.

    Args:
      *masks: set of attention mask arguments to combine, some can be None.
      dtype: final mask dtype

    Returns:
      Combined mask, reduced by logical and, returns None if no masks given.
    """
    _masks: list[Array] = [m for m in masks if m is not None]

    if not _masks:
        return None
    assert all((x.ndim == _masks[0].ndim for x in _masks)), (
        f"_masks must have same rank: {tuple((x.ndim for x in _masks))}"
    )
    mask, *other_masks = _masks

    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)

    return mask.astype(dtype)


def combine_biases(*masks: Array):
    """Combine attention biases.

    Args:
      *masks: set of attention bias arguments to combine, some can be None.

    Returns:
      Combined mask, reduced by summation, returns None if no masks given.
    """
    masks: list[Array] = [m for m in masks if m is not None]

    if not masks:
        return None
    assert all((x.ndim == masks[0].ndim for x in masks)), (
        f"masks must have same rank: {tuple((x.ndim for x in masks))}"
    )

    mask, *other_masks = masks

    for other_mask in other_masks:
        mask = mask + other_mask

    return mask
