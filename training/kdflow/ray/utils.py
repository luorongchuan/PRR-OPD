import os


def restore_cuda_visible_devices(visible_devices):
    """Restore the driver's CUDA view inside a co-located Ray actor."""
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")


def resolve_sglang_base_gpu_id(
    base_gpu_id,
    device_span,
    visible_device_count,
    visible_devices=None,
):
    """Map a Ray/physical GPU id into the actor's local CUDA namespace."""
    if device_span < 1:
        raise ValueError(f"device_span must be positive, got {device_span}")
    if visible_device_count < device_span:
        raise RuntimeError(
            "SGLang requires "
            f"{device_span} visible GPU(s), but the Ray actor sees "
            f"{visible_device_count}."
        )

    local_base_gpu_id = base_gpu_id
    if visible_devices:
        visible_device_ids = [item.strip() for item in visible_devices.split(",")]
        try:
            local_base_gpu_id = visible_device_ids.index(str(base_gpu_id))
        except ValueError:
            pass

    if (
        0 <= local_base_gpu_id
        and local_base_gpu_id + device_span <= visible_device_count
    ):
        return local_base_gpu_id
    if visible_device_count == device_span:
        return 0
    raise RuntimeError(
        f"SGLang base_gpu_id={base_gpu_id} resolved to local GPU "
        f"{local_base_gpu_id} with span={device_span}, outside the actor's "
        f"{visible_device_count} visible GPU(s) "
        f"(CUDA_VISIBLE_DEVICES={visible_devices or '<unset>'})."
    )


# Address https://github.com/ray-project/ray/issues/51117
# This function is used to get the bundle indices of a placement group
# and ensure that the bundles placed on the same node are grouped together.
def get_bundle_indices(placement_group, index, length):
    import ray

    pg_infos = ray.util.placement_group_table(placement_group)

    node_id_to_bundles = {}
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles.setdefault(node_id, []).append(bundle)

    sorted_bundle_indices = sum(node_id_to_bundles.values(), [])
    return sorted_bundle_indices[index * length : (index + 1) * length]


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)
