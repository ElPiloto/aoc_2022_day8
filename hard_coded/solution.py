from typing import Optional
from absl import app
import numpy as np

INPUT_FILE = '../aoc_input_small.txt'
INPUT_FILE = '../aoc_input.txt'

def parse_line(line: str) -> np.ndarray:
  """Returns vector from line of digits."""
  return np.array([int(x) for x in line])


def read_file(fname: Optional[str] = INPUT_FILE) -> np.ndarray:
  with open(fname) as file:
    vectors = [parse_line(line.rstrip()) for line in file]
    grid = np.vstack(vectors)
  return grid


def solve(heights: np.ndarray, should_print: bool = False):
  visibility = np.zeros_like(heights)
  # Set edges visible
  visibility[0, :] = 1
  visibility[:, 0] = 1
  visibility[-1, :] = 1
  visibility[:, -1] = 1
  # Keep track of biggest value seen by search direction at each location
  biggest_left = np.copy(heights)
  biggest_bottom = np.copy(heights)
  biggest_right = np.copy(heights)
  biggest_top = np.copy(heights)
  # Set interior to NaN to represent that we haven't searched these yet
  biggest_left[1:-2, 1:-2] = -1
  biggest_bottom[1:-2, 1:-2] = -1
  biggest_right[1:-2, 1:-2] = -1
  biggest_top[1:-2, 1:-2] = -1
  # traverse interior only
  for r in range(1, heights.shape[0]-1):
    for c in range(1, heights.shape[1]-1):
      h = heights[r, c]
      bottom = (r-1, c)
      left =(r, c-1)
      # __import__('pdb').set_trace()
      if h > biggest_left[left]:
        visibility[r, c] = 1
        biggest_left[r, c] = h
      else:
        biggest_left[r, c] = biggest_left[left]

      if h > biggest_bottom[bottom]:
        visibility[r, c] = 1
        biggest_bottom[r, c] = h
      else:
        biggest_bottom[r, c] = biggest_bottom[bottom]
  # traverse interior only
  for r in range(heights.shape[0]-2, 0, -1):
    for c in range(heights.shape[1]-2, 0, -1):
      h = heights[r, c]
      top = (r+1, c)
      right =(r, c+1)
      if h > biggest_right[right]:
        visibility[r, c] = 1
        biggest_right[r, c] = h
      else:
        biggest_right[r, c] = biggest_right[right]

      if h > biggest_top[top]:
        visibility[r, c] = 1
        biggest_top[r, c] = h
      else:
        biggest_top[r, c] = biggest_top[top]
  if should_print:
    print(heights)
    print(visibility)
    print(f'Total number of visible trees is: {visibility.sum()}')
  return visibility

def main(argv):
  del argv
  grid = read_file()
  solve(grid)

if __name__ == "__main__":
  app.run(main)

