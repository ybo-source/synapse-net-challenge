import os
import pooch


def get_sample_data(name: str) -> str:
    """Get the filepath to SynapseNet sample data, stored as mrc file.

    Args:
        name: The name of the sample data. Currently, we only provide the 'tem_2d' sample data.

    Returns:
        The filepath to the downloaded sample data.
    """
    registry = {
        "tem_2d.mrc": "3c6f9ff6d7673d9bf2fd46c09750c3c7dbb8fa1aa59dcdb3363b65cc774dcf28",
    }
    urls = {
        "tem_2d.mrc": "https://owncloud.gwdg.de/index.php/s/5sAQ0U4puAspcHg/download",
    }
    key = f"{name}.mrc"

    if key not in registry:
        valid_names = [k[:-4] for k in registry.keys()]
        raise ValueError(f"Invalid sample name {name}, please choose one of {valid_names}.")

    cache_dir = os.path.expanduser(pooch.os_cache("synapse-net"))
    data_registry = pooch.create(
        path=os.path.join(cache_dir, "sample_data"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    file_path = data_registry.fetch(key)
    return file_path
