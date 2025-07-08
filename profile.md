Layer timing profile (sorted, most to least time-consuming):

| Layer            | Time (s)  |
|------------------|-----------|
| cummax           | 3.5682    |
| direction_share  | 2.8743    |
| share_down       | 1.2245    |
| share_up         | 0.9772    |
| softmax          | 0.8114    |
| shift            | 0.4388    |
| decode_latents   | 0.3959    |
| nonlinear        | 0.2870    |
| normalize        | 0.1464    |
| postprocess_mask | 0.0059    |
| affine_head      | 0.0016    |
| affine_x_mask    | 0.0008    |
| affine_y_mask    | 0.0007    |
