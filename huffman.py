# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""
Implements the Huffman coding algorithm for lossless data compression.

This module provides functions to:
1. Build a Huffman tree from character frequencies.
2. Generate Huffman codes (a mapping from characters to binary strings).
3. Encode a text string into its Huffman-coded binary representation.
4. Decode a binary string back into the original text using the Huffman tree.

The implementation uses a min-heap to efficiently construct the Huffman tree.
"""
import heapq

class TreeNode:
    """A node in the Huffman tree."""
    def __init__(self, freq, char=None):
        """
        Initializes a TreeNode.

        Args:
            freq (float): The frequency of the character(s) in this subtree.
            char (str, optional): The character for a leaf node. Defaults to None.
        """
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        """Compare nodes by frequency, for use in the min-heap."""
        return self.freq < other.freq

def build_huffman_tree(chars, freqs):
    """
    Builds a Huffman tree from a list of characters and their frequencies.

    Args:
        chars (list[str]): A list of characters.
        freqs (list[float]): A list of corresponding character frequencies.

    Returns:
        TreeNode: The root node of the constructed Huffman tree.
    """
    # Create a leaf node for each character and its frequency.
    nodes = [TreeNode(freq, char) for char, freq in zip(chars, freqs)]
    # Use a min-heap to efficiently find the two nodes with the lowest frequencies.
    heapq.heapify(nodes)
    
    # Repeatedly merge the two lowest-frequency nodes until only one node remains.
    while len(nodes) > 1:
        # Pop the two nodes with the smallest frequencies.
        a = heapq.heappop(nodes)
        b = heapq.heappop(nodes)
        
        # Create a new internal node with the combined frequency.
        merged = TreeNode(a.freq + b.freq)
        merged.left = a
        merged.right = b
        
        # Add the new node back to the heap.
        heapq.heappush(nodes, merged)
        
    # The final node is the root of the Huffman tree.
    return nodes[0]

def build_mapping(node):
    """
    Builds a mapping from characters to their Huffman codes from a Huffman tree.

    Args:
        node (TreeNode): The root node of the Huffman tree.

    Returns:
        dict: A dictionary mapping each character to its binary Huffman code string.
    """
    mapping = {}
    
    # Helper function to traverse the tree using Depth-First Search (DFS).
    def dfs(node, cur_list):
        # Base case: if it's a leaf node, store the character and its code.
        if node.char is not None:
            mapping[node.char] = "".join(cur_list)
            return
        
        # Traverse left (append '0') and right (append '1').
        dfs(node.left, cur_list+['0'])
        dfs(node.right, cur_list+['1'])
        
    dfs(node,[])
    return mapping

def encode(chars, mapp):
    """
    Encodes a string of characters using a pre-built Huffman mapping.

    Args:
        chars (str): The string to encode.
        mapp (dict): The Huffman mapping from characters to binary codes.

    Returns:
        str: The encoded binary string.
    """
    return "".join([mapp[char] for char in chars])

def decode(chars, root):
    """
    Decodes a binary string using the Huffman tree.

    Args:
        chars (str): The binary string to decode.
        root (TreeNode): The root of the Huffman tree.

    Returns:
        str: The decoded original string.
    """
    out = []
    tree = root
    # Traverse the tree according to the bits in the encoded string.
    for char in chars:
        if char == "0":
            tree = tree.left
        else:
            tree = tree.right
        
        # If a leaf node is reached, we have found a character.
        if tree.char is not None:
            out.append(tree.char)
            # Reset to the root to find the next character.
            tree = root
    return "".join(out)


# %% [markdown]
# # Example Usage

# %%
# Define characters and their frequencies.
chars = ['a', 'b', 'c', 'd', 'e', 'f']
freqs = [0.1, 0.2, 0.3, 0.05, 0.05, 0.3]

# Build the Huffman tree.
root = build_huffman_tree(chars, freqs)

# %%
# Generate the character-to-code mapping.
mapp = build_mapping(root)
print("Huffman Mapping:", mapp)

# %%
# Encode a sample string.
encoded_text = encode("acf", mapp)
print("Encoded 'acf':", encoded_text)

# %%
# Decode the string.
decoded_text = decode(encoded_text, root)
print("Decoded text:", decoded_text)

# %%
# --- Verification ---
# Test with a randomly generated string to ensure correctness.
import numpy as np
texts = "".join(np.random.choice(list("abcdef"), 20))
encoded = encode(texts, mapp)
decoded = decode(encoded,root)

print("Original:", texts)
print("Encoded:", encoded)
print("Decoded:", decoded)
print("Verification successful:", decoded == texts)
