# Day 8, Advent of Code 2021

Let's figure out how to turn [Day 8, Advent of Code 2022](https://adventofcode.com/2022/day/8) into a task for an artificial neural network.


## Part 1

For Day 8, The task is to look at a grid of tree (heights) and determine which
ones are visible from the edge of the grid.  A tree is visible if it is the
tallest tree in a straight line to the edge of the grid.  We need to count the
total number of trees that are visible.

Example:

```
Tree heights:

30373
25512
65332
33549
35390
```
* Each tree on the edge is automatically visible.
* The top-left 5 is visible from the left and top. (It isn't visible from the
  right or bottom since other trees of height 5 are in the way.)

### Converting this into a machine learning problem

**Notable omission:** We will ignore the last step of summing up the visibility
across trees. This isn't particularly interesting to learn.

#### Binary Classification

For each tree in the input grid, we predict a binary value (0 or 1) indicating whether or not it
is visible. The training loss is the standard binary classification loss
(cross-entropy or some variant thereof.)

#### Ordinal Classification
This is a harder version of the problem where we try to predict the _visibility
count_ for each tree.  The visibility count for a tree is the number of edges
from which a tree is visible.

For each tree in the input grid, we predict integer values between [0, 4]
indicating the visibility count.  The training loss is an ordinal
classification loss.  There are a few different losses we could use here. TBD
which one we'll use.
