import os
import torch
import pickle

def to_cpu(obj):
    if isinstance(obj, torch.Tensor):  # 仅针对torch.Tensor执行.to("cpu")操作
        return obj.to("cpu")
    elif isinstance(obj, dict):  # 对字典类型进行递归处理
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):  # 对列表类型进行递归处理
        return [to_cpu(v) for v in obj]
    else:  # 其他类型，无需处理，直接返回
        return obj

def convert_model_for_cpu(model_path):
    print(f"正在转换的模型: {model_path}")
    
    try:
        # 判断文件类型
        if model_path.endswith(".pt") or model_path.endswith(".pth"):
            model = torch.load(model_path, map_location="cuda:0")
            model = to_cpu(model)
        if model_path.endswith(".jit"):
            model = torch.jit.load(model_path)
            model.to("cpu")
        elif model_path.endswith(".pkl"):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            model = to_cpu(model)
    except Exception as e:
        print(f"遇到问题跳过 {model_path}，原因: {e}")
        return

    # 保存转换后的模型，文件名添加'_cpu'作为区分
    base_path, ext = os.path.splitext(model_path)
    cpu_model_path = base_path + '_cpu' + ext

    if model_path.endswith(".jit"):
        torch.jit.save(model, cpu_model_path)
    elif model_path.endswith(".pt") or model_path.endswith(".pth"):
        torch.save(model, cpu_model_path)
    elif model_path.endswith(".pkl"):
        with open(cpu_model_path, 'wb') as f:
            pickle.dump(model, f)

    print(f"完成转换并保存到: {cpu_model_path}")
    
# 搜索目录下的所有文件并进行转换
def search_and_convert(directory):
    print(f"开始在 {directory} 目录下搜索模型文件...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".pt", ".pth", ".pkl", ".jit")):
                convert_model_for_cpu(os.path.join(root, file))
    print(f"已完成 {directory} 目录下所有模型文件的转换!")

# 使用示例:
search_and_convert(".")
