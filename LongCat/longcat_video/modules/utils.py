import torch


def _offload_kv_cache(kv_cache: tuple) -> tuple:
    """
    Offload KV cache to CPU.
    
    Args:
        kv_cache: Tuple of (k_quant, v_quant), where each can be a Tensor or dict
        
    Returns:
        Tuple of offloaded (k_quant, v_quant)
    """
    k_quant, v_quant = kv_cache
    
    def offload_item(item):
        if isinstance(item, dict):
            for name, tensor_or_list in item.items():
                if isinstance(tensor_or_list, list):
                    item[name] = [t.to("cpu") for t in tensor_or_list]
                elif isinstance(tensor_or_list, torch.Tensor):
                    item[name] = tensor_or_list.to("cpu")
                else:
                    # Seems that the item is a scalar
                    assert not hasattr(tensor_or_list, "device")
                    item[name] = tensor_or_list
            return item
        else:
            return item.to("cpu")
    
    return offload_item(k_quant), offload_item(v_quant)


def _onload_kv_cache(kv_cache: tuple, device: torch.device) -> tuple:
    """
    Onload KV cache to the specified device.
    
    Args:
        kv_cache: Tuple of (k_quant, v_quant), where each can be a Tensor or dict
        device: Target device
        
    Returns:
        Tuple of onloaded (k_quant, v_quant)
    """
    k_quant, v_quant = kv_cache
    
    def onload_item(item):
        if isinstance(item, dict):
            for name, tensor_or_list in item.items():
                if isinstance(tensor_or_list, list):
                    item[name] = [t.to(device) for t in tensor_or_list]
                elif isinstance(tensor_or_list, torch.Tensor):
                    item[name] = tensor_or_list.to(device)
                else:
                    # Seems that the item is a scalar
                    assert not hasattr(tensor_or_list, "device")
                    item[name] = tensor_or_list
            return item
        else:
            return item.to(device)
    
    return onload_item(k_quant), onload_item(v_quant)
