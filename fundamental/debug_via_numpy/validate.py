import argparse
import numpy as np

nBS = 4
nSL = 64
nEmbedding = 768

def layerNormCPU(x, gamma, beta, epsilon=6e-6):
    _x = x
    _0  = np.mean(_x,2)[:,:,np.newaxis]
    _1  = _x - _0
    _2  = _1 * _1
    _3  = np.mean(_2,2)[:,:,np.newaxis]
    _4  = np.array(epsilon,dtype=np.float32)
    _5  = _4.reshape(1,1,1)
    _6  = _3 + _5
    _7  = np.sqrt(_6)
    _8  = 1 / _7                # 1/sqrt(...)
    _9  = _1 * _8
    _10 = _9 * gamma
    _11 = _10 + beta
    return _11


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-inputs", action='store_true', help="generate numpy inputs")
    parser.add_argument("-v", action='store_true', help="validate")
    args = parser.parse_args()
    
    np.random.seed(97)
    # np_out1 = np.load("../np_res.npy")
    x = np.random.rand(nBS,nSL,nEmbedding).astype(np.float32) * 2 - 1
    # gamma = np.random.rand(1, 1, nEmbedding)
    # beta = np.random.rand(1, 1, nEmbedding)
    # x = np.ones((nBS,nSL,nEmbedding)).astype(np.float32)
    gamma = np.ones((1, 1, nEmbedding)).astype(np.float32)
    beta = np.zeros((1, 1, nEmbedding)).astype(np.float32)

    if args.gen_inputs:
        np.save("inp", x)
        np.save("gamma", gamma)
        np.save("beta", beta)
    elif args.v:
        np_out = layerNormCPU(x, gamma, beta)
        # print(np_out)

        gpu_out = np.load("./out.npy")
        # print(gpu_out)

        res = np.allclose(np_out, gpu_out, rtol=1e-2, atol=1e-3)
        print("===> same? ", res)
        print("===> abs err: ", np.abs(np_out - gpu_out).max())
