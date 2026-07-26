import torch
import numpy as np


def todevice(batch, device, callback=None, non_blocking=False):
    if callback:
        batch = callback(batch)
    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)
    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


def collate_with_cat(whatever, lists=False):
    if isinstance(whatever, dict):
        return {k: collate_with_cat(vals, lists=lists) for k, vals in whatever.items()}
    elif isinstance(whatever, (tuple, list)):
        if len(whatever) == 0:
            return whatever
        elem = whatever[0]
        T = type(whatever)
        if elem is None:
            return None
        if isinstance(elem, (bool, float, int, str)):
            return whatever
        if isinstance(elem, tuple):
            return T(collate_with_cat(x, lists=lists) for x in zip(*whatever))
        if isinstance(elem, dict):
            return {k: collate_with_cat([e[k] for e in whatever], lists=lists) for k in elem}
        if isinstance(elem, torch.Tensor):
            return listify(whatever) if lists else torch.cat(whatever)
        if isinstance(elem, np.ndarray):
            return (listify(whatever) if lists else torch.cat([torch.from_numpy(x) for x in whatever]))
        return sum(whatever, T())


def listify(elems):
    return [x for e in elems for x in e]


def loss_of_one_batch(batch, model, criterion, device, precision, ret=None, aux=None):
    for view in batch:
        for name in ("img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres".split()):
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    views = batch
    autocast_dict = dict(device_type=device.type)
    if precision == "32":
        autocast_dict["enabled"] = False
    elif precision == "16-mixed":
        autocast_dict["dtype"] = torch.float16
    elif precision in ["bf16-mixed", "bf16-mixed-no-grad-scaling"]:
        autocast_dict["dtype"] = torch.bfloat16
    elif precision == torch.bfloat16:
        autocast_dict["dtype"] = torch.bfloat16
    with torch.autocast(**autocast_dict):
        if aux is None:
            preds = model(views)
        else:
            preds = model(views, aux)
        loss = criterion(views, preds) if criterion is not None else None
    result = dict(views=views, preds=preds, loss=loss)
    return result


@torch.no_grad()
def inference(multiple_views_in_one_sample, model, device, dtype, aux=None):
    result = []
    res = loss_of_one_batch(collate_with_cat([tuple(multiple_views_in_one_sample)]), model, None, device, dtype, aux=aux)
    result.append(todevice(res, "cpu"))
    result = collate_with_cat(result, lists=False)
    return result
