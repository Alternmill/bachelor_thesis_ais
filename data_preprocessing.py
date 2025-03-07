# Modifications copyright (C) 2025 SOMATIC

import os
import cv2
import datetime as dt
from bcr_azure_ml_py.bcr_journals.journal_processors import JournalsToFramesProcessor, JRecToVideoFrameProcessor, JRecLastRobotStateExtendedWithTimeThresh
from bcr_azure_ml_py.bcr_journals.journal_generators import JournalRecordGenerator

from bcr_azure_ml_py.projects.stall_door_segmentation.data_preprocessing_config import JournalsToSDSFramesProcessorConfig
from bcr_azure_ml_py.utils.data_processing.data_filters import L1DistThreshFrameFilter

class JournalsToSDSFramesProcessor(JournalsToFramesProcessor):
    def __init__(self, cfg: JournalsToSDSFramesProcessorConfig):
        super(JournalsToSDSFramesProcessor, self).__init__(cfg)
        self.jrec_generator = JournalRecordGenerator(cfg.jrec_generator_config)
        
        self.jrec_to_video_frame_processor_dict = {}
        for topic_name in cfg.jrec_to_video_frame_processor_config.encoded_cameras_stream_topics_name:
            self.jrec_to_video_frame_processor_dict[topic_name] = JRecToVideoFrameProcessor(topic_name)
        
        self.jrec_last_rse_with_time_thresh = JRecLastRobotStateExtendedWithTimeThresh(cfg.jrec_last_robot_state_extended_with_time_thresh)
        self.l1_dist_thresh_frame_filter = L1DistThreshFrameFilter(cfg.l1_dist_thresh_frame_filter_config)


    def __call__(self, journal_root_dir_path: str, procd_data_root_dir_path: str, clean_up: bool) -> None:
        print(f'[{self.cfg.sample_name}] Process id: [{os.getpid()}], parent id: [{os.getppid()}]')
        print(f'[{self.cfg.sample_name}] Journal_root_dir_path: [{journal_root_dir_path}]')
        print(f'[{self.cfg.sample_name}] procd_data_root_dir_path: [{procd_data_root_dir_path}]')

        self.clean_up_if_needed(procd_data_root_dir_path, clean_up)
        if self.already_succeeded(procd_data_root_dir_path):
            return

        journal_paths_rel = list(self.journal_rel_path_generator()) # TODO: ml filesystem -> azure blob storage
        print(f'[{self.cfg.sample_name}] Starting processing [{len(journal_paths_rel)}] journals...')

        for jornal_rel_path in journal_paths_rel:
            print(f'[{self.cfg.sample_name}] Processing: [{jornal_rel_path}]')
            t1 = dt.datetime.now()
            
            for jrec in self.jrec_generator(self.journal_abs_path(journal_root_dir_path, jornal_rel_path)):
                # inputs
                rse = self.jrec_last_rse_with_time_thresh(jrec)
                processor = self.jrec_to_video_frame_processor_dict.get(jrec.topic, None)
                
                if processor is None:
                    continue

                frame_and_index = processor(jrec)

                if frame_and_index is None:
                    continue
                frame, frame_index = frame_and_index
                camera_name = jrec.topic.split('/')[-1] # camera name index

                if rse is None:
                    continue

                if not self.l1_dist_thresh_frame_filter(frame):
                    continue
                
                if self.cfg.write_procd_image:
                    frame = cv2.resize(frame, [self.cfg.img_width, self.cfg.img_height])
                    self.frame_writer(frame, self.procd_image_path(procd_data_root_dir_path, jornal_rel_path, "", f"{camera_name}"))

            t2 = dt.datetime.now()
            print(f'[{self.cfg.sample_name}] Journal procesing time: [{t2-t1}]')

        self.on_succeeded(procd_data_root_dir_path)