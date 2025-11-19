import numpy as np
from vggt_needle.needle import backend_ndarray as nd

print("nd.cuda().enabled():", nd.cuda().enabled())

A = nd.array(np.random.randn(4, 4), device=nd.cuda())
B = nd.array(np.random.randn(4, 4), device=nd.cuda())
C = A @ B
print("C shape:", C.shape)