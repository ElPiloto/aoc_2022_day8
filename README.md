# Day 8, Advent of Code 2021

Let's figure out how to turn [Day 8, Advent of Code 2022](https://adventofcode.com/2022/day/8) into a task for an artificial neural network.


## Part 1

For Day 8, The task is to look at a grid of tree (heights) and determine which
ones are visible from the edge of the grid.  A tree is visible if it is the
tallest tree in a straight line to the edge of the grid.  We need to count how
many edges a tree is visible from. We will call this the _visibility count_.
Then we need to sum up the visibility count across all the trees as the final
answer to the problem.

### Converting this into a machine learning problem

**Notable omission:** We will ignore the last step of summing up the visibility
counts. This isn't particular fun to learn.

#### Binary Classification

For each tree in the input grid, we predict a binary value (0 or 1) indicating whether or not it
is visible. The training loss is the standard binary classification loss
(cross-entropy or some variant thereof.)

**Note:** This ignores visibility count.  It's just a simplification of the
problem that is still interesting to implement.

#### Ordinal Classification
For each tree in the input grid, we predict integer values between [0, 4]
indicating the visibility count.  The training loss is an ordinal
classification loss.  There are a few different losses we could use here. TBD
which one we'll use.
