# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING, Callable, Dict, Type

from .base import KVConnectorBase
import vllm.envs as envs

# NOTE(Kuntai): We prefer not to directly the classes with "_V1" suffix.
# This makes it easier for us to deprecate code in v0 (which will happen soon).
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorRole as KVConnectorRole_V1,
    KVConnectorBase as KVConnectorBase_V1,
    RoleToKVConnector as RoleToKVConnector_V1)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class KVConnectorFactory:
    _registry: Dict[str, Callable[[], Type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str,
                           class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> Type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(cls, rank: int, local_rank: int,
                         config: "VllmConfig") -> KVConnectorBase:
        connector_name = config.kv_transfer_config.kv_connector
        if connector_name not in cls._registry:
            raise ValueError(f"Unsupported connector type: {connector_name}")

        connector_cls = cls._registry[connector_name]()

        if envs.VLLM_USE_V1:
            # NOTE(Kuntai): v1 connector is explicitly seperated into two roles.
            # Scheduler connector:
            # - Co-colate with scheduler process
            # - Should only be used inside the Scheduler class
            # Worker connector:
            # - Co-locate with worker process
            # - Should only be used inside the forward context & attention layer
            # We build these two connectors separately to enforce strict 
            # separation
            scheduler_connector = connector_cls(
                role=KVConnectorRole_V1.SCHEDULER,
                rank=rank,
                local_rank=local_rank,
                config=config,
            )
            worker_connector = connector_cls(
                role=KVConnectorRole_V1.WORKER,
                rank=rank,
                local_rank=local_rank,
                config=config,
            )
            return RoleToKVConnector_V1(
                scheduler_connector=scheduler_connector,
                worker_connector=worker_connector,
            )
        else:
            return connector_cls(rank, local_rank, config)


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.
KVConnectorFactory.register_connector(
    "PyNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "vllm.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "LMCacheConnector",
    "vllm.distributed.kv_transfer.kv_connector.lmcache_connector",
    "LMCacheConnector")

KVConnectorFactory.register_connector(
    "MooncakeStoreConnector",
    "vllm.distributed.kv_transfer.kv_connector.mooncake_store_connector",
    "MooncakeStoreConnector")

KVConnectorFactory.register_connector(
    "SharedStorageConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector",
    "SharedStorageConnector")