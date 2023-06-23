# Copyright 2022 Google LLC
import os
import sys
import eval
import functools
import apache_beam as beam
import mediapy as media
import natsort
import numpy as np
import tensorflow as tf
from typing import List, Sequence
from eval import interpolator as interpolator_lib
from eval import util
from absl import app
from absl import flags
from absl import logging
from tqdm.auto import tqdm

sys.path.extend(['frame-interpolation/'])

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def _output_frames(args, frames: List[np.ndarray], frames_dir: str):
    """Writes PNG-images to a directory.
  
    If frames_dir doesn't exist, it is created. If frames_dir contains existing
    PNG-files, they are removed before saving the new ones.
  
    Args:
        frames: List of images to save.
        frames_dir: The output directory to save the images.
  
    """
    if tf.io.gfile.isdir(frames_dir):
        old_frames = tf.io.gfile.glob(f'{frames_dir}/frame_*.png')
        if old_frames:
            logging.info('Removing existing frames from %s.', frames_dir)
            for old_frame in old_frames:
                tf.io.gfile.remove(old_frame)
    else:
        tf.io.gfile.makedirs(frames_dir)
    for idx, frame in tqdm(
        enumerate(frames), total=len(frames)):
        eval.util.write_image(f'{frames_dir}/frame_{idx:09d}.png', frame)
    logging.info('Output frames saved in %s.', frames_dir)

class ProcessDirectory(beam.DoFn):
    """DoFn for running the interpolator on a single directory at the time."""
    def __init__(self, args):
        self.args = args
        self._PATTERN = args._PATTERN
        self._MODEL_PATH = args._MODEL_PATH
        self._TIMES_TO_INTERPOLATE = args._TIMES_TO_INTERPOLATE
        self._FPS = args._FPS
        self._ALIGN = args._ALIGN
        self._BLOCK_HEIGHT = args._BLOCK_HEIGHT
        self._BLOCK_WIDTH = args._BLOCK_WIDTH
        self._OUTPUT_VIDEO = args._OUTPUT_VIDEO
        self._INPUT_EXT = args._INPUT_EXT
    
    def setup(self):
        self.interpolator = interpolator_lib.Interpolator(
            self._MODEL_PATH, self._ALIGN,
            [self._BLOCK_HEIGHT, self._BLOCK_WIDTH])

        if self._OUTPUT_VIDEO:
            ffmpeg_path = eval.util.get_ffmpeg_path()
            print(ffmpeg_path)
            media.set_ffmpeg(ffmpeg_path)

    def process(self, directory: str):
        directory = self._PATTERN
        print(directory)
        print(self._INPUT_EXT)
        input_frames_list = [
            natsort.natsorted(tf.io.gfile.glob(f'{str(directory)}/*.{ext}'))
            for ext in self._INPUT_EXT
        ]
        input_frames = functools.reduce(lambda x, y: x + y, input_frames_list)
        print(input_frames)
        logging.info('Generating in-between frames for %s.', directory)
        frames = list(
            eval.util.interpolate_recursively_from_files(
                input_frames, self._TIMES_TO_INTERPOLATE, self.interpolator))
        _output_frames(self.args, frames, f'{directory}/interpolated_frames')
        if self._OUTPUT_VIDEO:
            logger.info(f'..Writing video to {directory}/interpolated.mp4')
            media.write_video(f'{directory}/interpolated.mp4', frames, fps=self._FPS)
            logging.info('Output video saved at %s/interpolated.mp4.', directory)


def film_interpolator(args):
    args = args
    directories = tf.io.gfile.glob(args._PATTERN)
    pipeline = beam.Pipeline('DirectRunner')
    (pipeline | 'Create directory names' >> beam.Create(directories)  # pylint: disable=expression-not-assigned
     | 'Process directories' >> beam.ParDo(ProcessDirectory(args)))
    
    result = pipeline.run()
    result.wait_until_finish()
