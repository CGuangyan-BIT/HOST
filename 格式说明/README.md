# Joint/action mapping 与动作格式

本文说明 HOST real open-loop 推理使用的状态字段、动作排列、归一化及位姿还原方式。对应实现为 [`SelfGroundedPredictorEval`](../policy_training/eval/real_openloop/self_grounded_predictor_eval.py)。完整 episode 和相机数据格式见[数据预处理文档](../data_preprocessing/README.md)。

## Mapping 示例与配置

提供一个完整示例：[10042_joint_action_mapping.json](./10042_joint_action_mapping.json)。

**不同数据集的归一化区间可能不同。这个文件用于说明格式和对应数据集的参数，不是所有数据集或 checkpoint 的通用归一化配置。** 复现已有 checkpoint 时，应匹配其训练所用的字段、顺序和归一化参数；不能直接用目标数据集重新统计的区间替换。使用新数据训练时，应准备相应 mapping，并在推理时保留同一套参数。

`SelfGroundedPredictorEval` 从 checkpoint 目录的 `config.yaml` 读取 `data.train.joint_action_mapping_dir`。如果该值为 `null`，需要配置实际目录；模型权重本身不能代替 mapping。

下面是使用示例文件时的配置片段。将占位路径替换为实际的绝对路径，保留其余 checkpoint 配置：

```yaml
data:
  train:
    joint_action_mapping_dir: /absolute/path/to/HOST/格式说明
    use_relative_action: true
    use_6d_rotation: true
```

示例文件对应 `dataset_name="10042"`。读取器按 `<dataset_name>_joint_action_mapping.json` 查找文件，因此不要仅将文件改名为 `joint_action_mapping.json`。两个动作开关也必须与训练配置一致。

文件的完整结构为：

```text
{
  "<dataset path prefix>": {
    "action_keys": [...],
    "joint_keys": [...],
    "norm_min_delta": {"<field>": {"min": [...], "delta": [...]}}
  }
}
```

外层映射不可省略。训练读取器对单 entry 文件直接使用唯一 entry；对多个 entry，按 episode 路径前缀匹配。当前 real open-loop 评估的归一化工具取第一个 entry，因此不同统计应分别准备匹配的文件，不应依赖该评估工具自动选择多个 entry。示例外层保留原始数据路径，单 entry 的加载不要求该路径存在。

## 字段顺序与维度

`joint_keys` 指定输入状态的顺序，`action_keys` 指定输出动作的顺序。读取器按照列表顺序拼接，不按字段名排序。示例的输入是从臂末端位姿和夹爪状态，并非机械臂各关节角。

| 顺序 | 输入 `joint_keys` | 输出 `action_keys` | 6D 模式下维数 |
|---|---|---|---|
| 1 | `follow_left_position` | `master_left_position` | 3 |
| 2 | `follow_left_rotation` | `master_left_rotation` | 6 |
| 3 | `follow_left_gripper` | `master_left_gripper` | 1 |
| 4 | `follow_right_position` | `master_right_position` | 3 |
| 5 | `follow_right_rotation` | `master_right_rotation` | 6 |
| 6 | `follow_right_gripper` | `master_right_gripper` | 1 |

输入状态和物理动作均为 20 维。模型输出的最后两维为 progress 和 mask，因此模型输出共 22 维；这两维不作为机器人动作反归一化。旋转解码为 Euler 后，双臂物理动作为 14 维。

JSON 中额外保存的 `follow_*_joint_pos` 统计未被示例的 `joint_keys` 引用，因此不会加入这 20 维输入。

## 归一化参数

以下数值仅对应附带的示例，左右臂使用相同参数：

| 字段 | `min` | `delta` | 用途 |
|---|---|---|---|
| `*_position` | `[-0.1, -0.5, -0.5]` | `[0.6, 1.0, 1.0]` | 绝对位置，含输入状态 |
| `*_position_relative` | `[-0.2, -0.2, -0.2]` | `[0.4, 0.4, 0.4]` | 相对位置动作 |
| `*_gripper` | `[-0.5]` | `[6.5]` | 绝对夹爪值 |
| 6D rotation | 每维 `-1` | 每维 `2` | 运行时使用的旋转参数 |

JSON 中原始 Euler rotation 的参数是 `min=[-1.5,-1.5,-2.0]`、`delta=[3.0,3.0,3.5]`。开启 `use_6d_rotation` 后，读取器将旋转转换为 6D，并使用固定的 `min=-1, delta=2`，而不是这组 Euler 参数。

逐维变换为：

```python
normalized = np.clip(2.0 * (raw - norm_min) / norm_delta - 1.0, -1.0, 1.0)
raw = (normalized + 1.0) * norm_delta / 2.0 + norm_min
```

`delta` 是归一化跨度，上界为 `min + delta`，不是单步运动增量。训练时裁剪到 `[-1, 1]`，被裁剪的信息无法通过反归一化恢复。

在相对动作模式中，输入位置仍用绝对位置统计，输出位置使用对应 `*_position_relative` 统计；夹爪仍为绝对值。该训练路径使用 `ds_factor=1`，相对位置每轴的示例范围为 `[-0.2, 0.2]`。不要使用缺失统计时的透传回退来替代训练所用参数。

## 相对位姿与旋转表示

动作片段以观测时刻／片段起始时刻的从臂末端状态为固定参考。对示例中的 master/follow 字段：

```text
p_relative[t] = p_master[t] - p_follow_reference
R_relative[t] = R_master[t] @ R_follow_reference.T

p_target[t] = p_relative[t] + p_follow_reference
R_target[t] = R_relative[t] @ R_follow_reference
```

整个预测片段共享同一参考，不逐帧累加。位置差直接沿原始位姿参考系的轴计算，没有转到工具局部坐标系。

position 按米使用。原始 rotation 为弧度制 `[roll, pitch, yaw]`，代码调用 `Rotation.from_euler("ZYX", [yaw, pitch, roll])`，对应 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。6D rotation 按旋转矩阵前两列排列：

```text
[R00, R10, R20, R01, R11, R21]
```

后处理顺序是：取前 20 维物理动作 → 反归一化 → 使用观测时刻从臂位姿还原绝对位姿 → 6D rotation 转 Euler 或控制接口所需表示。

传入的 `joints` 应包含完整的原始从臂状态，尤其是左右臂 position 和 rotation，不能传入归一化后的状态来充当参考。`SelfGroundedPredictorEval` 在相对动作模式且参考状态完整时返回的 `action_raw` 已完成上述处理，不需要再次反归一化或相对转绝对。

## 末端参考点、坐标系与夹爪

模型沿用采集端提供的末端位姿，未额外施加 TCP 偏置或基座变换。代码中的 world-frame subtraction 表示在原始参考系直接作差，并不单独定义硬件上的世界原点、各臂基座或工具参考点。法兰／TCP／夹爪中心的具体选择、轴向和工具偏置需要由采集端标定配置确定；mapping 不包含这些标定信息。

gripper 每臂占 1 维，是保留原始语义的连续夹爪位置／控制量；相对动作模式不对它作差，也不将它二值化。示例的归一化区间为 `[-0.5, 6.0]`。这个区间不等于夹爪机械极限，也不表示以米为单位的开口宽度；物理单位及数值增大对应打开还是闭合，应由相应夹爪驱动定义。

## 实现参考

- [归一化参数构建](../policy_training/src/self_grounded_prediction/datasets/custom/joint_action_mapping_norms.py)
- [训练数据拼接与相对动作计算](../policy_training/src/self_grounded_prediction/datasets/custom/mydatasets.py)
- [反归一化、相对转绝对与旋转解码](../policy_training/eval/real_openloop/action_utils.py)
- [real open-loop 推理接口](../policy_training/eval/real_openloop/self_grounded_predictor_eval.py)
