from typing import Any, Dict, Optional
from kedro.io import DataCatalog as KedroDataCatalog
from kedro.config import OmegaConfigLoader
from kedro.io import MemoryDataset
import os
import yaml


class DataCatalog:
    """
    A wrapper around Kedro's DataCatalog to manage RLStudio datasets.
    It automatically loads configuration from conf/catalog.yml.
    """

    def __init__(self, conf_path: str = "conf", env: str = "base"):
        self.conf_path = conf_path
        self.env = env
        self._catalog = self._load_catalog()

    def _load_catalog(self) -> KedroDataCatalog:
        # Simple loading strategy for MVP
        catalog_path = os.path.join(self.conf_path, "catalog.yml")
        if not os.path.exists(catalog_path):
            return KedroDataCatalog()

        with open(catalog_path, "r") as f:
            conf_catalog = yaml.safe_load(f) or {}

        return KedroDataCatalog.from_config(conf_catalog)

    def load(self, name: str) -> Any:
        return self._catalog.load(name)

    def save(self, name: str, data: Any) -> None:
        self._catalog.save(name, data)

    def add(self, name: str, dataset: Any) -> None:
        self._catalog.add(name, dataset)

    def exists(self, name: str) -> bool:
        return self._catalog.exists(name)
