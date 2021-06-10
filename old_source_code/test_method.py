import torch

x = torch.randn(10, 1).flatten()
y = torch.randn(10, 1).flatten()
print(x)
print(y)
# indices = torch.tensor([2])
# print(torch.index_select(x, 0, indices))



after, after_indices = torch.sort(x, descending=True)
print(after)
print(after_indices)

#result = torch.zeros(10, dtype=y.dtype).scatter_(0, after_indices, y)

y = y.numpy()
after_indices = after_indices.numpy()

result = y[after_indices]

result = torch.tensor(result)

def original_order(ordered, indices):
    z = torch.empty_like(ordered)
    for i in range(ordered.size(0)):
        z[indices[i]] = ordered[i]
    return z


unsorted = original_order(result, after_indices)
# print(torch.index_select(x, 1, indices))
print(result)
print(unsorted)
