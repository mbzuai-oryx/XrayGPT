import os
import logging
import warnings

from xraygpt.common.registry import registry
from xraygpt.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from xraygpt.datasets.datasets.openi_dataset import OpenIDataset
from xraygpt.datasets.datasets.mimic_dataset import MIMICDataset


@registry.register_builder("mimic")
class MIMICBuilder(BaseDatasetBuilder):
    train_dataset_cls = MIMICDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/mimic/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets


@registry.register_builder("openi")
class OpenIBuilder(BaseDatasetBuilder):
    train_dataset_cls = OpenIDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/openi/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets
