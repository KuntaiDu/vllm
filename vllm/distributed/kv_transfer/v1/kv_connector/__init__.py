from vllm.distributed.kv_transfer.v1.kv_connector.base import KVConnectorRole, KVConnectorBase
from vllm.distributed.kv_transfer.v1.kv_connector.native_connector import NativeKVConnector

_KV_CONNECTORS = {}

# TODO (ApostaC):
# - Add better error handling
# - Add connector configuration parsing
# - Make the return value optional

def init_kv_connector(role: KVConnectorRole, **kwargs) -> KVConnectorBase:
    """Initialize the KV connector.

    Args:
        role (KVConnectorRole): the role of the connector.
        **kwargs: additional arguments for the connector.
    """
    if "cache_config" not in kwargs:
        raise ValueError("cache_config is required for KVConnector initialization.")
    cache_config = kwargs["cache_config"]

    connector = NativeKVConnector(role, cache_config.block_size)
    _KV_CONNECTORS[role] = connector
    return connector


def get_kv_connector(role: KVConnectorRole) -> KVConnectorBase:
    """Get the KV connector.

    Args:
        role (KVConnectorRole): the role of the connector.

    Returns:
        KVConnectorBase: the KV connector.
    """
    return _KV_CONNECTORS[role]

__all__ = [
    "KVConnectorRole",
    "KVConnectorBase",
    "init_kv_connector",
]

