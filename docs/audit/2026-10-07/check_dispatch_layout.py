import runpy
from unittest.mock import patch
import torch
import torch_npu
from liger_kernel.ops import LigerEmbeddingFunction
from liger_kernel.ops.backends._ascend.ops.embedding import embedding_forward
print('BACKEND',LigerEmbeddingFunction.__module__)
cls=runpy.run_path('src/liger_kernel/transformers/experimental/embedding.py')['LigerEmbedding']
emb=cls(8,4).npu()
ids=torch.tensor([1,2],device='npu')
emb._benchmark_kernel_operation_mode='forward'
with patch('torch.nn.functional.embedding',wraps=torch.nn.functional.embedding) as spy:
    emb(ids);torch.npu.synchronize();print('FORWARD_FLAG_NATIVE_CALLS',spy.call_count)
del emb._benchmark_kernel_operation_mode
with torch.no_grad(),patch('torch.nn.functional.embedding',wraps=torch.nn.functional.embedding) as spy:
    emb(ids);torch.npu.synchronize();print('NO_GRAD_NATIVE_CALLS',spy.call_count)
ids=torch.arange(6,device='npu').reshape(2,3).t()
print('REFERENCE_SHAPE',tuple(torch.nn.functional.embedding(ids,emb.weight).shape))
try:
    embedding_forward(emb.weight,ids)
    print('NONCONTIG_OK')
except Exception as e:
    print('NONCONTIG',type(e).__name__,str(e))
