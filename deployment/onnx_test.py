from torchvision.models import resnet18
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    
    resnet = resnet18(pretrained=True)
    resnet.eval()
    x = torch.rand((1, 3, 128, 192))
    y = resnet(x)
    print(len(y))
    weights_save_path = Path("./resnet18_imagenet.onnx")

    # note we don't need the onnx package to save out an onnx model. Torch has a built-in
    torch.onnx.export(resnet,
        args=x,
        f=weights_save_path,
        # export_params=False,
        opset_version=10,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}}
        )
    print(weights_save_path.exists())

    # load the model back in and check it 
    resnet_onnx_model = onnx.load(weights_save_path)
    check_out = onnx.checker.check_model(resnet_onnx_model)
    print(check_out)
    
    
    # run a timing test on some random data
    n_tests = 50
    ort_sess = ort.InferenceSession(str(weights_save_path))
    ort_times = []
    for i in range(n_tests):
        t0 = time()
        outputs = ort_sess.run(None, {"input": np.random.rand(10, 3, 128, 192).astype('float32')})
        t = time()
        ort_times.append(t - t0)
        
    torch_times = []
    for i in range(n_tests):
        t0 = time()
        outputs = resnet(torch.rand(10,3,128,192)) 
        t = time()
        torch_times.append(t-t0)
    
    ort_mean = np.mean(np.array(ort_times))
    torch_mean = np.mean(np.array(torch_times))
    
    print(f"Mean Torch inference time: {torch_mean}")
    print(f"Mean ONNX inference time: {ort_mean}")
    print(f"speedup factor: {torch_mean/ort_mean}")
    
    # plot the result
    df = pd.DataFrame({"Time (ms)": ort_times + torch_times,
                       "Framework": ["ONNX"]*50 + ["Torch"]*50})
    f, ax = plt.subplots()
    sns.boxplot(x= 'Framework', y="Time (ms)", data=df, ax=ax)
    plt.tight_layout()    
    plt.show()
    
    print('done')