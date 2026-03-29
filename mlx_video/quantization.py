from functools import lru_cache
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path


class QQLinearWithBias(nn.QuantizedLinear):
    """QuantizedLinear variant that also quantizes activations via ``mx.qqmm``."""

    def __call__(self, x):
        x = mx.qqmm(
            x,
            self["weight"],
            scales=self["scales"],
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            x = x + self["bias"]
        return x


@lru_cache(maxsize=None)
def activation_quantized_matmul_supported(mode: str) -> bool:
    if mode == "mxfp8":
        group_size, bits = 32, 8
    elif mode == "nvfp4":
        group_size, bits = 16, 4
    else:
        return False

    try:
        weight = mx.ones((2, group_size), dtype=mx.float32)
        quantized_weight, scales, *_ = mx.quantize(
            weight,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        x = mx.ones((2, group_size), dtype=mx.float32)
        y = mx.qqmm(
            x,
            quantized_weight,
            scales=scales,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        mx.eval(y)
    except RuntimeError as exc:
        if "QQMatmul" in str(exc) and "NYI" in str(exc):
            return False
        raise

    return True


def is_quantized_linear_module(module: nn.Module) -> bool:
    return isinstance(module, nn.QuantizedLinear)


def dequantize_linear_weight(module: nn.QuantizedLinear) -> mx.array:
    return mx.dequantize(
        module.weight,
        module.scales,
        module.biases,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    )


def requantize_linear_module_like(
    module: nn.QuantizedLinear,
    weight: mx.array,
) -> nn.QuantizedLinear:
    layer_cls = QQLinearWithBias if isinstance(module, QQLinearWithBias) else nn.QuantizedLinear
    requantized = layer_cls(
        weight.shape[1],
        weight.shape[0],
        bias="bias" in module,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    )
    quantized_values = mx.quantize(
        weight,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    )
    requantized.weight = quantized_values[0]
    requantized.scales = quantized_values[1]
    requantized.biases = quantized_values[2] if len(quantized_values) == 3 else None
    if "bias" in module:
        requantized.bias = module.bias
    return requantized


def quantize_modules(
    model: nn.Module,
    group_size: int = None,
    bits: int = None,
    *,
    mode: str = "affine",
    quantize_input: bool = False,
    class_predicate: Optional[Callable[[str, nn.Module], Union[bool, dict]]] = None,
):
    """Quantize modules, using ``QQLinearWithBias`` when activation quantization is requested."""

    class_predicate = class_predicate or (lambda _, m: hasattr(m, "to_quantized"))
    activation_quantized_count = 0
    activation_fallback_count = 0

    def _maybe_quantize(path, module):
        nonlocal activation_quantized_count, activation_fallback_count
        bool_or_params = class_predicate(path, module)
        if not bool_or_params:
            return module
        if not hasattr(module, "to_quantized"):
            raise ValueError(f"Unable to quantize model of type {type(module)}")

        if isinstance(bool_or_params, bool):
            kwargs = {
                "group_size": group_size,
                "bits": bits,
                "mode": mode,
            }
            if quantize_input:
                kwargs["quantize_input"] = True
        elif isinstance(bool_or_params, dict):
            kwargs = {
                "group_size": group_size,
                "bits": bits,
                "mode": mode,
                **bool_or_params,
            }
        else:
            raise ValueError(
                "``class_predicate`` must return a bool or a dict of quantization parameters"
            )

        quantize_input_for_module = kwargs.pop("quantize_input", False)
        quant_mode = kwargs.get("mode", mode)
        if quantize_input_for_module:
            if quant_mode not in ("mxfp8", "nvfp4"):
                raise ValueError(
                    "Activation quantization is only supported for mxfp8 and nvfp4."
                )
            if not isinstance(module, nn.Linear):
                raise ValueError(
                    "Activation quantization is only supported for Linear layers."
                )
            if not activation_quantized_matmul_supported(quant_mode):
                activation_fallback_count += 1
                return module.to_quantized(**kwargs)
            activation_quantized_count += 1
            return QQLinearWithBias.from_linear(module, **kwargs)

        return module.to_quantized(**kwargs)

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_maybe_quantize, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    model._activation_quantization_requested = bool(quantize_input)
    model._activation_quantization_enabled = activation_quantized_count > 0
    model._activation_quantization_fallback = activation_fallback_count > 0
