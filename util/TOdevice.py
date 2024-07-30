def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    # import pdb; pdb.set_trace()
    return data.to(device, non_blocking=True)