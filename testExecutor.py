from abc import abstractmethod, ABCMeta
from functools import partial
import tensorflow as tf
from speechInput import InputBatchLoader
from speechModel import SpeechModel
from preprocess import DatasetReader

class TestExecutor(metaclass=ABCMeta):

    def __init__(self):
        self.reader = DatasetReader('data')
        self.input_size = self.determine_input_size()
        self.speech_input = InputBatchLoader(self.input_size, 64, partial(self.create_sample_generator, self.get_loader_limit_count()), self.get_max_steps())

    def determine_input_size(self):
        return next(self.create_sample_generator(limit_count=1))[0].shape[1]

    def get_max_steps(self):
        return None

    @abstractmethod
    def get_loader_limit_count(self) -> int:
        raise NotImplementedError('Loader limit count needs to be implemented')

    @abstractmethod
    def create_sample_generator(self, limit_count: int):
        raise NotImplementedError('Sample generator creation needs to be implemented')

    def start_pipeline(self, sess, n_threads=1):
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        self.speech_input.start_threads(sess=sess, coord=coord, n_threads=n_threads)
        return coord

    def create_model(self, sess):
        model = SpeechModel.create_default_model('evaluate', self.input_size, self.speech_input)
        model.restore(sess, 'train/best-weights')
        return model
