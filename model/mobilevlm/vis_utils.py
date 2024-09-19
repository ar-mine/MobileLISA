import json
import torch
import torch.nn as nn


# 定义递归函数来按层次结构打印模型
# def save_model_structure(module):
#     output = []
#     # 遍历子模块
#     def recursion(record: list, module, indent=0):
#         # Whether be a meta module
#         meta = True
#         for name, child in module.named_children():
#             meta = False
#             # 打印缩进和子模块名称
#             record.append(str("\n" + "  " * indent + f"{name}: {child.__class__.__name__}"))
#
#             # 递归打印子模块的子层
#             recursion(record, child, indent + 1)
#         if meta:
#             record.append(f", {module}, training={module.training}")
#     recursion(output, module)
#     output = "".join(output)
#     with open("model_structure.txt", "w") as f:
#         f.write(output)

def create_nested_dict_from_string(s, value=None):
    # 将字符串按点分隔，得到键列表
    keys = s.split('.')

    # 从最内层开始，逐层构建嵌套字典
    nested_dict = value  # 初始值为最终要存储的值，默认是 None
    for key in reversed(keys):
        nested_dict = {key: nested_dict}

    return nested_dict

def merge_nested_dicts(dict1, dict2):
    # 遍历 dict2 中的键和值，将其合并到 dict1 中
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            # 如果两者在相同键下都是字典，递归合并
            merge_nested_dicts(dict1[key], value)
        else:
            # 否则直接将 dict2 的值覆盖到 dict1
            dict1[key] = value

def save_model_structure(module):
    output = {}
    for n, p in module.named_parameters():
        new_dict = create_nested_dict_from_string(n, value={
            "shape": str(p.shape),
            "trainable": str(p.requires_grad)
        })
        merge_nested_dicts(output, new_dict)

    with open("model_structure.json", "w") as f:
        json.dump(output, f)


if __name__ == '__main__':
    # 定义一个简单的示例网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(64 * 6 * 6, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    # 初始化模型
    model = SimpleNet()

    # 输出模型的层次结构
    save_model_structure(model)
