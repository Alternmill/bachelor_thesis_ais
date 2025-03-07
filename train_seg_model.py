# Modifications copyright (C) 2025 SOMATIC

from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes

from bcr_azure_ml_py.azure.ml.ml_client import AzureMlConnection, MLConfigs
from bcr_azure_ml_py.projects.yolo_image_segmentation.components.create_json_dataset_component import create_json_dataset_component
from bcr_azure_ml_py.projects.yolo_image_segmentation.components.yolo_data_prepare_component import prepare_and_train_yolo_data_component
from bcr_azure_ml_py.projects.yolo_image_segmentation.components.score_yolo_segmentation import score_yolo_segmentation_component
from bcr_azure_ml_py.projects.yolo_image_segmentation.configs.segmentation_config import SegmentationConfig
from bcr_azure_ml_py.projects.utils import NameGeneratorUsingLastTrainedModelVersion
from bcr_azure_ml_py.utils.config import Config
from bcr_azure_ml_py.projects.yolo_image_segmentation.configs.train_image_segmentation_global_config import TrainImageSegmentationModelGlobalConfig

CONFIG_NAME = 'train_image_segmentation.yaml'
CONFIG = Config.from_yaml_file_path(TrainImageSegmentationModelGlobalConfig, SegmentationConfig.path_to_yaml_config_by_name(CONFIG_NAME))
NAME_GENERATOR = NameGeneratorUsingLastTrainedModelVersion(CONFIG.model_name, MLConfigs.SegmentationMlConfig)

@pipeline(
    name=f"V{NAME_GENERATOR.new_last_version}: Segmentation Model from Python SDK",
    description="All categories"
)
def prepare_data_pipeline():
    ui_data_asset = SegmentationConfig.ui_data_asset_as_input()
    yaml_segmentation_file = SegmentationConfig.path_to_yaml_config_by_name(CONFIG_NAME)
    training_data_folder_path = NAME_GENERATOR.generate_training_data_folder(SegmentationConfig.azure_ml_uri_to_yolo_training_data())
    model_folder_path = NAME_GENERATOR.generate_model_folder(SegmentationConfig.azure_ml_uri_to_yolo_training_data())
    model_abs_file_path = NAME_GENERATOR.generate_model_for_download(SegmentationConfig.azure_ml_uri_to_weights_store())
    finetune_model_folder = NAME_GENERATOR.path_to_existing_models(SegmentationConfig.azure_ml_uri_to_weights_store())

    train_ds_node = create_json_dataset_component(mount_dir_path=ui_data_asset, # pylint: disable=no-value-for-parameter
                                                  yaml_file_path=yaml_segmentation_file,
                                                  work_with_train_data=True)
    train_json = train_ds_node.outputs.save_new_json_path


    test_ds_node = create_json_dataset_component(mount_dir_path=ui_data_asset, # pylint: disable=no-value-for-parameter
                                                 yaml_file_path=yaml_segmentation_file,
                                                 work_with_train_data=False)
    test_json = test_ds_node.outputs.save_new_json_path

    prepare_data_for_training_node = prepare_and_train_yolo_data_component(mount_dir_path = ui_data_asset, # pylint: disable=no-value-for-parameter
                                                                           json_train_file_folder = train_json,
                                                                           json_test_file_folder = test_json,
                                                                           finetune_models_path = finetune_model_folder,
                                                                           yaml_file_path = yaml_segmentation_file)
    prepare_data_for_training_node.outputs.training_data_folder = Output(type=AssetTypes.URI_FOLDER, path=training_data_folder_path, mode=InputOutputModes.RW_MOUNT)
    prepare_data_for_training_node.outputs.model_abs_path = Output(type=AssetTypes.URI_FOLDER, path=model_folder_path, mode=InputOutputModes.RW_MOUNT)
    prepare_data_for_training_node.outputs.model_abs_file_path = Output(type=AssetTypes.URI_FILE, path=model_abs_file_path, mode=InputOutputModes.RW_MOUNT)
    
    yolo_training_config_dir = prepare_data_for_training_node.outputs.yaml_config_path
    eval_input_model_path = prepare_data_for_training_node.outputs.model_abs_path
    training_data = prepare_data_for_training_node.outputs.training_data_folder

    score_yolo_segmentation_component(
        yolo_training_config_dir = yolo_training_config_dir,
        model_abs_path = eval_input_model_path,
        training_data = training_data,
        yaml_file_path = yaml_segmentation_file
    )

if __name__ == '__main__':
    pipeline_job = prepare_data_pipeline()
    pipeline_job.settings.default_compute = CONFIG.gpu_compute_target
    
    ml_client = AzureMlConnection.create_azure_ml_client(MLConfigs.SegmentationMlConfig)
    pipeline_job = ml_client.jobs.create_or_update(job=pipeline_job, experiment_name=CONFIG.experiment_name)
    ml_client.jobs.stream(pipeline_job.name)
