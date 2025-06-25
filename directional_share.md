# Understanding directional communication layer complexities
_Exported on 6/25/2025 at 19:14:12 EDT from Cursor (1.1.6)_

---

**User**

explain the following background in the context of function direction_share. focus more on the concept than the code. Use code as reference when things are unclear
By default, the network is equivariant to permutations of the eight directions, but we only want symmetry up to rotations and flips. So, this layer provides a way to send information between two slices in the 
 dimension, depending on the angular difference in the two directions. This layer defines a separate linear map to be used for each of the 64 possible combinations of angles, but the weights of the linear maps are minimally tied such that the directional communication layer is equivariant to reflections and rotations. This gets complicated really fast, since the 
 dimension’s indices also permute when equivariance transformations are applied. Every direction slice in a tensor accumulates it’s 8 messages, and adds the results together.9

---

**Cursor**

Think of every feature-tensor in the model as being split into eight “compass-point” slices:  
right, up-right, up, up-left, left, down-left, down, and down-right.  Those eight slices live in the extra “direction” axis that you see in many of the tensors.

1. What symmetry do we really want?  
   • If you rotate the whole problem by 90 °, or mirror it left ↔ right, the network’s answer should rotate / mirror with it.  
   • We do not need the far larger symmetry of “any arbitrary permutation of the eight direction labels”.  

   Early layers that treat the direction axis like just another channel are automatically invariant to *all* permutations – that is too much.  We therefore add a special layer that *breaks* the un­wanted permutations while *preserving* rotations and reflections.  That layer is `direction_share`.

2. Conceptual job of `direction_share`  
   For every direction slice d₁ it gathers a message from every other slice d₂, transforms those messages with a small linear layer, weights them by how far apart the two directions are, and then adds the eight messages back into slice d₁.  After the exchange each slice knows something about its neighbours’ orientations.

3. Why 64 linear maps?  
   There are 8 possible senders (d₂) × 8 possible receivers (d₁).  In principle you need 64 different linear maps.  To keep the model both compact and dihedral-equivariant we *tie* them:

   • Rotation equivariance ⇒ a map may depend only on the *angular difference* Δ = (d₂ − d₁) mod 8, not on the absolute labels.  
   • Reflection equivariance ⇒ the parameters for +Δ and −Δ must be identical.

   In code this minimal tying shows up as the fixed coefficient list

   ```501:509:layers.py
   coefficients = [1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2]
   c = coefficients[(d2 - d1) % 8]
   ```

   The coefficient array is symmetric and only a function of Δ.  If you rotate or flip the tensor, Δ stays the same, so the algebra—and therefore the network’s predictions—rotate/flip along with the data.

4. Putting the messages together  
   The body of the double loop

   ```509:520:layers.py
   x_list[d1] = x_list[d1] + c * affine(z_list[d2], weights[d1][d2], use_bias=use_bias)
   ```

   performs:
   • normalisation (`z = normalize(x)`),  
   • a small per-pair affine map `weights[d1][d2]`,  
   • scaling by the Δ-dependent coefficient `c`,  
   • accumulation into the receiver slice `d₁`.

   After all 64 interactions are accumulated, the eight updated slices are stacked back into a tensor along the original direction axis

   ```520:533:layers.py
   return torch.stack(x_list, dim=direction_dim)
   ```

   and the outer `@add_residual` decorator adds the usual residual skip-connection.

5. Effect on symmetry  
   • Because the only thing that distinguishes pairs of directions is their *relative* angle, the layer commutes with the dihedral group of the octagon (rotations by 45 ° and reflections).  
   • Because the communication is *not* invariant to arbitrary relabel­lings, the excess permutation symmetry is gone; the network can now learn orientation-aware patterns (e.g. “continuation in the same direction” or “corner junction”).

6. Effect on information flow  
   Each direction slice receives eight messages and can choose—through the learned affine weights—how much it should heed a slice that is 45 °, 90 °, 135 °, or 180 ° away.  The fixed numeric coefficients merely set a rough prior (straight-ahead and opposite directions are weighted a bit more than diagonals); the learned weight matrices decide the actual content passed along.

In short, `direction_share` is a small, explicitly dihedral-equivariant communication block that lets every orientation talk to the others while preserving only the *physically meaningful* symmetries of rotations and reflections.

```python
   coefficients = [1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2]
   c = coefficients[(d2 - d1) % 8]
```

```python
   x_list[d1] = x_list[d1] + c * affine(z_list[d2], weights[d1][d2], use_bias=use_bias)
```

```python
   return torch.stack(x_list, dim=direction_dim)
```

