import torch
import torch.nn.functional as F

def simple_non_vectorized_test():
    """Test the non-vectorized function with a simple case"""
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    print(f"Input mask: {mask}")
    
    L = mask.shape[0]
    search_lengths = list(range(1, L + 1))
    max_logprob, best_slice_start, best_slice_end = None, None, None

    for length in search_lengths:
        print(f"Testing length {length}")
        logprobs = []
        for offset in range(mask.shape[0] - length + 1):
            score = -torch.sum(mask[:offset]) + torch.sum(mask[offset:offset + length]) - torch.sum(mask[offset + length:])
            logprobs.append(score)
            print(f"  offset {offset}: score = {score.item()}")
        
        logprobs = torch.stack(logprobs)
        max_score = torch.max(logprobs)
        argmax_idx = torch.argmax(logprobs).item()
        
        print(f"  Max score for length {length}: {max_score.item()} at offset {argmax_idx}")
        
        if max_logprob is None or max_score > max_logprob:
            max_logprob = max_score
            best_slice_start = argmax_idx
            best_slice_end = argmax_idx + length
            print(f"  New best: start={best_slice_start}, end={best_slice_end}")

    print(f"Final result: start={best_slice_start}, end={best_slice_end}")
    return best_slice_start, best_slice_end

if __name__ == "__main__":
    print("=== Simple Debug Test ===")
    simple_non_vectorized_test() 