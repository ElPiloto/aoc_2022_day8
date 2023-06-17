from enum import IntEnum
from typing import List, Iterable, Optional

import ml_collections
import numpy as np
import tensorflow as tf
from hard_coded import solution as hc_solution


AOC_INPUT_FILE = './aoc_input.txt'


class SyntheticGenerator():
  """Generates synthetic values."""

  def __init__(self,
      size: int = 5,
      max_height: int = 4,
      rng_seed: Optional[int] = 112233):
    self.rng_state = np.random.RandomState(rng_seed)
    self.max_height = max_height
    self.size = size

  def get_shapes(self):
    return ((self.size, self.size), (self.size, self.size))

  def generator(self):
    def _generator():
      while True:
        # Generate a random grid of tree heights
        heights = self.rng_state.randint(
            low = 0,
            high=self.max_height,
            size=(self.size, self.size),
        )

        visibility = hc_solution.solve(heights)

        yield heights, visibility
    return _generator


# class AOCInputGenerator():

  # def __init__(self, input_file: str = AOC_INPUT_FILE):
    # lines = hc_solution.read_file(input_file)
    # self._inputs, self._targets = solve_cumulative(lines)
    # self._num_examples = len(self._inputs)

  # def generator(self):
    # def _generator():
      # for i in range(self._num_examples):
        # yield self._inputs[i], self._targets[i]
    # return _generator


class BatchDataset:

  def __init__(self, generator, shapes):
    self._generator = generator
    self._shapes = shapes

  def __call__(self, batch_size: int):
    ds = tf.data.Dataset.from_generator(
            self._generator,
            (tf.float32, tf.int32),
            output_shapes=self._shapes,
    )
    ds = ds.batch(batch_size=batch_size)
    return ds

def build_train_data(
    train_config: ml_collections.ConfigDict,
    train_seed: int,
    batch_size: int,
) -> Iterable:
  generator = SyntheticGenerator(
      size=train_config['size'],
      max_height=train_config['max_height'],
      rng_seed=train_seed,
  )
  ds = BatchDataset(generator.generator(), generator.get_shapes())
  batch_iterator = ds(batch_size=batch_size).as_numpy_iterator()
  return batch_iterator
