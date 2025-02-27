"""
This module defines dataset builders for various instructional video datasets.

Each builder class is registered with the `registry` and inherits from `Video_Instruct_Builder`.
They specify the dataset class and configuration dictionary for their respective datasets.
Attributes:
    train_dataset_cls: The class of the training dataset.
    DATASET_CONFIG_DICT: A dictionary containing the default configuration file path for the dataset.
Methods:
    _download_ann: Placeholder method for downloading annotations.
    _download_vis: Placeholder method for downloading visual data.
    build: Builds the dataset by initializing processors and creating dataset instances.
"""

from timechat.common.registry import registry
from timechat.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from timechat.datasets.datasets.video_instruct_dataset import Video_Instruct_Dataset

@registry.register_builder("video_instruct")
class Video_Instruct_Builder(BaseDatasetBuilder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/instruct/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'

        model_type = self.config.model_type if self.config.model_type else 'vicuna'
        num_frm = self.config.num_frm if self.config.num_frm else 8
        sample_type = self.config.sample_type if self.config.sample_type else 'uniform'
        max_txt_len = self.config.max_txt_len if self.config.max_txt_len else 512
        stride = self.config.stride if self.config.stride else 0

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token=num_video_query_token,
            tokenizer_name=tokenizer_name,
            data_type=self.config.data_type,
            model_type=model_type,
            num_frm=num_frm,
            sample_type=sample_type,
            max_txt_len=max_txt_len,
            stride=stride
        )

        return datasets

@registry.register_builder("youcook2_instruct")
class Youcook2Instruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/youcook2_instruct.yaml",
    }


@registry.register_builder("time_instruct")
class TimeInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/time_instruct.yaml",
    }


@registry.register_builder("valley72k_instruct")
class Valley72kInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/valley72k_instruct.yaml",
    }


@registry.register_builder("qvhighlights_instruct")
class QVhighlightsInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/qvhighlights_instruct.yaml",
    }


@registry.register_builder("charades_instruct")
class CharadesInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/charades_instruct.yaml",
    }

@registry.register_builder("didemo_instruct")
class DidemoInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/didemo_instruct.yaml",
    }

@registry.register_builder("activitynet_instruct")
class ActivityNetInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/activitynet_instruct.yaml",
    }