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
    
    
    ### CPU tests
    # switch onnx session to CPU runtime
    print('Preparing for CPU tests...')
    ort_sess = ort.InferenceSession(str(weights_save_path), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    provs = ort_sess.get_providers()
    ort_sess.set_providers([prov  for prov in provs if "CPU" in prov])
    print(f"ORT session providers: {ort_sess.get_providers()}")
    
    
    # run a timing test on some random data
    n_tests = 50
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
        
    ### GPU tests
    # switch onnx session to GPU for test; back to original providers
    print('Preparing for GPU tests...')
    ort_sess.set_providers(provs)

    print(f"ORT session providers: {ort_sess.get_providers()}")
    ort.set_default_logger_severity(0)
    
    ort_times_c = []
    for i in range(n_tests+1):
        # do not count the time it takes to move the input tensor to GPU
        np_value = np.random.rand(10, 3, 128, 192).astype('float32')
        ort_value = ort.OrtValue.ortvalue_from_numpy(np_value, 'cuda', 0)
        Y_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type([10, 1000], np.float32, 'cuda', 0)
        
        # bind ORT output to cuda device, so that it isn't automatically copied to CPU
        io_binding = ort_sess.io_binding()
        io_binding.bind_input(name='input', 
                              device_type='cuda', 
                              device_id=0, 
                              element_type=np.float32,
                              shape=ort_value.shape(), 
                              buffer_ptr=ort_value.data_ptr())
        
        io_binding.bind_output(name='output', 
                               device_type='cuda', 
                               device_id=0,
                               element_type=np.float32,
                               shape=Y_ortvalue.shape(), 
                               buffer_ptr=Y_ortvalue.data_ptr())
        t0 = time()
        ort_sess.run_with_iobinding(io_binding)
        # outputs = ort_sess.run(None, {"input": ort_value})
        t = time()
        
        # ignore the warmup run
        if i > 0:
            ort_times_c.append(t - t0)
    
    resnet.to('cuda:0')
    torch_times_c = []
    for i in range(n_tests+1):
        t0 = time()
        outputs = resnet(torch.rand(10,3,128,192, device='cuda:0')) 
        t = time()
        
        # ignore the warmup run
        if i > 0:
            torch_times_c.append(t-t0)
    
    ort_mean = np.mean(np.array(ort_times))
    torch_mean = np.mean(np.array(torch_times))
    ort_mean_c = np.mean(np.array(ort_times_c))
    torch_mean_c = np.mean(np.array(torch_times_c))
    
    print(f"Mean Torch inference time (CPU): {torch_mean}")
    print(f"Mean ONNX inference time (CPU): {ort_mean}")
    print(f"Mean Torch inference time (GPU): {torch_mean_c}")
    print(f"Mean ONNX inference time (GPU): {ort_mean_c}")
    print(f"CPU speedup factor: {torch_mean/ort_mean}")
    print(f"GPU speedup factor: {torch_mean_c/ort_mean_c}")
    
    # plot the result
    df = pd.DataFrame({"Time (ms)": ort_times + ort_times_c + torch_times + torch_times_c,
                       "Framework": ["ONNX (CPU)"]*n_tests + 
                       ["ONNX (GPU)"]*n_tests + 
                       ["Torch (CPU)"]*n_tests + 
                       ["Torch (GPU)"]*n_tests})
    
    f, ax = plt.subplots()
    sns.boxplot(x= 'Framework', y="Time (ms)", data=df, ax=ax)
    plt.tight_layout()    
    plt.show()
    
    print('done')