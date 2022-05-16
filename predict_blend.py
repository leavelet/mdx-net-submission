import soundfile as sf
import torch
import numpy as np
from evaluator.music_demixing import MusicDemixingPredictor
from demucs.model import Demucs
from demucs.utils import apply_model
from models import get_models
import onnxruntime as ort
from time import time, sleep

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using devide {device}')

ONNX_MODE = "f16"   # select from "f32" "f16" and "cpu"

if ONNX_MODE == "f32":
    ort_providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
        }),
        'CPUExecutionProvider',
    ]
    model_device = "cpu"
elif ONNX_MODE == "f16":
    ort_providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
        }),
        'CPUExecutionProvider',
    ]
    model_device = "cpu"
elif ONNX_MODE == "cpu":
    ort_providers = [
        'CPUExecutionProvider'
    ]
    model_device = "cuda:0"
else:
    raise Exception(f"unknown ONNX_MODE: {ONNX_MODE}")
print(f'ONNX_MODE: {ONNX_MODE}')
print(f'will compute Predictor.model on {model_device}')

class Predictor(MusicDemixingPredictor):
        
    def prediction_setup(self):
        self.models = get_models(model_name, load=False, device=model_device)
        # self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=48 if '48' in demucs_name else 64)
        self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=48 if '48' in demucs_name else 64)
        self.demucs.load_state_dict(torch.load(f'model/{demucs_name}.ckpt'))
        self.demucs.eval()
        if ONNX_MODE in ['f16', 'cpu']:
            self.demucs = self.demucs.to(device)
        
    def prediction(self, mixture_file_path, bass_file_path, drums_file_path, other_file_path, vocals_file_path):
        # file_paths = [bass_file_path, drums_file_path, other_file_path, vocals_file_path]      
        file_paths = [vocals_file_path]      
        mix, rate = sf.read(mixture_file_path)
        sources = self.demix(mix.T)
        for i in range(len(sources)):
            sf.write(file_paths[i], sources[i].T, rate)
    
    def demix(self, mix):
        print(f"dexing_base...")
        base_out = self.demix_base(mix)
        print(f"dexing_demucs...")
        # demucs_out = self.demix_demucs(mix)
        demucs_out = self.demix_demucs(mix)[3:]
        sources = base_out * b + demucs_out * (1-b)
        print(f'sources: {sources.shape}')
        return sources
    
    def demix_base(self, mix):
        start_time = time()
        sources = []
        n_sample = mix.shape[1]
        for model in self.models:
            print(f'now target is {model.target_name}...')
            trim = model.n_fft//2
            gen_size = model.chunk_size-2*trim
            pad = gen_size - n_sample%gen_size
            mix_p = np.concatenate((np.zeros((2,trim)), mix, np.zeros((2,pad)), np.zeros((2,trim))), 1)

            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i:i+model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32)

            with torch.no_grad():
                torch.cuda.empty_cache()    # clean GPU memory cache
                options = ort.SessionOptions()
                options.inter_op_num_threads = 1

                if ONNX_MODE == 'f16':
                    # use f16 (half precision) model, in order to reduce GPU memory usage
                    _ort = ort.InferenceSession(f'onnx/{model.target_name}-f16.onnx', providers=ort_providers)
                    ort_result = _ort.run(None, {'input': model.stft(mix_waves).cpu().numpy().astype(np.float16, copy=False)})[0]
                elif ONNX_MODE == 'f32':
                    # use f32 model, which has the risk of running out of GPU memory
                    _ort = ort.InferenceSession(f'onnx/{model.target_name}.onnx', providers=ort_providers)
                    ort_result = _ort.run(None, {'input': model.stft(mix_waves).cpu().numpy()})[0]
                elif ONNX_MODE == 'cpu':
                    # use CPU
                    _ort = ort.InferenceSession(f'onnx/{model.target_name}.onnx', providers=ort_providers)
                    ort_result = _ort.run(None, {'input': model.stft(mix_waves).cpu().numpy()})[0]

                tar_waves = model.istft(torch.tensor(ort_result))
                tar_signal = tar_waves[:,:,trim:-trim].transpose(0,1).reshape(2, -1).cpu().numpy()[:, :-pad]
            sources.append(tar_signal)
            
        print(f"time usage (demix_base): {round(time()-start_time, 2)}s")
        return np.array(sources)
    
    def demix_demucs(self, mix):
        start_time = time()
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(self.demucs.to(device), mix.to(device), split=True, overlap=0.5)
            
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]
        torch.cuda.empty_cache()    # clean GPU memory cache
        print(f"time usage (demix_demucs): {round(time()-start_time, 2)}s")
        return sources
        

model_name = 'tdf_extra'
demucs_name = 'demucs_extra'

# b = np.array([[[0.5]], [[0.5]], [[0.7]], [[0.9]]])
b = np.array([[[0.9]]])

submission = Predictor()
submission.run()
print("Successfully completed music demixing...")
