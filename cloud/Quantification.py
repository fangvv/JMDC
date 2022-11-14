import torch
import numpy as np
import sys

def calcScaleZeroPoint(min_val, max_val, num_bits=6):
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    # 防止Nan和Inf造成的错误
    if min_val==max_val and min_val==0:
        scale =(max_val - min_val)+0.001 / (qmax - qmin)
    elif min_val==max_val and min_val!=0:
        scale=max_val/(qmax-qmin)
    else:
        scale =(max_val - min_val) / (qmax - qmin)

    initial_zero_point = (qmin - min_val) / scale

    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)


    return scale, zero_point


def quantize_tensor(x, num_bits=6, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    # print("q_x",q_x)
    # print(q_x.clamp_(qmin,qmax))
    # print(q_x.round_())
    q_x.clamp_(qmin, qmax).round_()
    # print("-----------------------")
    # print(q_x.int())
    return q_x.int(),scale,zero_point


def dequantize_tensor(scale,x,zero_point):
    scale=float(scale)
    return scale * (x.float() - zero_point)

if __name__ == "__main__":
    a = torch.tensor([[0.5,0.8,-0.5,-0.3],
                     [0.7,-0.1,0.4,0.2],
                     [-0.5,0.5,0.11,0.0],
                     [-0.3,0.4,0.5,-0.6]],dtype=torch.float32)
    print(a)
    b,scale,zero=quantize_tensor(a)
    print(b)
    print("-------------------------------")
    # print(scale)
    # print(zero)
    c = dequantize_tensor(scale,b,zero)
    print(c)
    # print(type(c))
    # print("-------------------")
    # print(c)
    # a_size = sys.getsizeof(a.storage())
    # print("a_size:",a_size)
    # np.save('a.npy', a)
    # b_size = sys.getsizeof(b.storage())
    # b=(b.int().dtype)
    #
    # np.save('b.npy', b)
    # print("b_size:",b_size)
    # c_size = sys.getsizeof(c.storage())
    # print("c_size:",c_size)
    # np.save('c.npy', c)