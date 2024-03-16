import abc
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from speechInput import BaseInputLoader

class SpeechModel:

    def __init__(self, input_loader: BaseInputLoader, input_size: int, num_classes: int):
        self.input_loader = input_loader
        self.input_size = input_size
        self.convolution_count = 0
        self.global_step = tf.Variable(0, trainable=False)

        self.inputs, self.sequence_lengths, self.labels = input_loader.get_inputs()
        self.logits = self._create_network(num_classes)

        tf.summary.image('logits', tf.expand_dims(tf.transpose(self.logits, (1, 2, 0)), 3))
        tf.summary.histogram('logits', self.logits)

    def add_training_ops(self, learning_rate: bool = 1e-3, learning_rate_decay_factor: float = 0, max_gradient_norm: float = 5.0, momentum: float = 0.9):
        # Training operations implementation

     def add_decoding_ops(self, language_model: str = None, lm_weight: float = 0.8, word_count_weight: float = 0.0, valid_word_count_weight: float = 2.3):
        # Decoding operations implementation

      def finalize(self, log_dir: str, run_name: str, run_type: str):
        # Finalize method implementation

       def _convolution(self, value, filter_width, stride, input_channels, out_channels, apply_non_linearity=True):
        # Convolution method implementation
        pass

    def _create_network(self, num_classes):
        # Network creation implementation
        pass

    def create_default_model(self, command, input_size: int, speech_input: BaseInputLoader):
        # Default model creation implementation
        pass
