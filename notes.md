1. model return mask of shape (example, x, 2) and (example, y, 2). more expressive would be to return mask of shape (example, x, y, 2).
2. model assumes size of test output is bound by max in the training data (counter example, double grid size task). Try using a global max size of 30.
tradeoff between compute and accuracy.
4. share down function, special treatment for x and y axes just average over the top left corner but enumerate over all possible x and y offsets in train.py.
better to not have special treatment?
5. why normalize along all but last axis per tensor? how about normalize along just last axis? allow tunable normalization factor like BatchNorm?
6. normalize exclude example axis?
7. cummax limits aggregation to top left corner based on masks. save as share down.
8. mask search over both start and length. alternatively, mask search over only length, use top left corner of logits as start.
9. as part of post-processing, instead of collecting all solutions throughout training, collect solution based on last model weights and z without sample.
10. add "batch" as in multiple starting points.
   - share f or separate f for each batch - pre-train with shared f, then finetune with batched f?
   - first run batch and select top k, then run more iterations on top k, discarding the rest?
11. **optimal batch_size**:
   - (batch_size, iterations run), e.g., (1,8), (2,4), (4,2), (8,1) over all layers and all tasks.
   - walltime vs model performance, want best performance with a given walltime budget.
   - mapping from task dim -> optimal batch size (best performance given walltime budget).
   - **multi-start optimization and batch size implmentation can be separated**, e.g., I can have 8 starting points and batch size of 2 forward and backward pass (repeat 4 times in loop). and this can be done at layer level, e.g. different layers have different batch size while all have same number of starting points.
12. Test-time-training (TTT) is naively parallelizable as weights not shared across tasks.
    
does not work:
3. multitensor system is conceptually elegant, but not efficient for computation. Reimplement with a single flattened tensor?
   - share up / down, pick and choose from flattened tensor to add. share down needs to be normalized by size of the tensor. another way to think of this is sparse matrix multiplication.
   - affine / residual is very natural with flattened tensor.
   - normalize more aligned with multitensor system since it is block-wise operation. 