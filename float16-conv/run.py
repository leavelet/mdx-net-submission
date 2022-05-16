import float16
import onnx

models = [
    "../onnx/bass.onnx",
    "../onnx/drums.onnx",
    "../onnx/other.onnx",
    "../onnx/vocals.onnx"
]

for model in models:
    input_path = model
    output_path = model[:-5] + "-f16.onnx"
    print(f"{input_path} -> {output_path}")
    new_onnx_model = float16.convert_float_to_float16_model_path(input_path)
    onnx.save(new_onnx_model, output_path)
