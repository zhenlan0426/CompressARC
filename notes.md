1. model return mask of shape (example, x, 2) and (example, y, 2). more expressive would be to return mask of shape (example, x, y, 2).
2. model assumes size of test output is bound by max in the training data (counter example, double grid size task). Try using a global max size of 30.
tradeoff between compute and accuracy.
3. multitensor system is conceptually elegant, but not efficient for computation. Reimplement with a single flattened tensor?
4. share down function, special treatment for x and y axes just average over the top left corner but enumerate over all possible x and y offsets in train.py.
better to not have special treatment?
5. - share up / down, pick and choose from flattened tensor to add. share down needs to be normalized by size of the tensor. another way to think of this is sparse matrix multiplication.
   - affine / residual is very natural with flattened tensor.
   - normalize more aligned with multitensor system since it is block-wise operation. why normalize along all but last axis per tensor? how about normalize along just last axis? allow tunable normalization factor like BatchNorm?
6. normalize exclude example axis?
7. cummax limits aggregation to top left corner based on masks. save as share down.