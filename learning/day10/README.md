# Day 10: 拓扑排序详解

## 目标

深入理解计算图的执行顺序确定算法——DFS 后序遍历的反向拓扑排序。

## 核心概念

### 计算图节点 (RuntimeOperator)

每个算子是一个图节点，包含：
- `name`: 唯一标识符
- `type`: 层类型（如 `"nn.ReLU"`）
- `input_names`: 上游算子的输出名
- `output_names`: 本算子输出的名称
- `start_time`: 执行顺序编号（拓扑排序后赋值）

### 反向拓扑排序算法

```cpp
int ReverseTopoSort(RuntimeOperator& node, int count) {
    if (node.has_forward) return count;      // 已访问，返回
    node.has_forward = true;

    // 递归访问所有下游节点
    for (auto& [name, next_op] : node.output_operators) {
        count = ReverseTopoSort(*next_op, count);
    }

    // 后序：所有下游都访问完后，才给自己编号
    node.start_time = ++count;
    return count;
}
```

**关键**: 后序遍历保证——一个节点编号比所有下游节点小，所以按编号升序执行就是正确的拓扑序。

## 本天 Demo

1. **线性链**: `input -> conv -> relu -> pool -> output`
2. **分支图**: 一个输入分叉到 ReLU 和 Sigmoid 两条路
3. **跳跃连接**: ResNet 风格的残差块拓扑
4. **多入口图**: 两个独立输入汇合到一个输出

## 构建与运行

```bash
mkdir build && cd build
cmake .. && make && ./day10
```

## 思考题

- 为什么需要先建立节点关系再做拓扑排序？（拓扑排序需要知道节点间的依赖关系）
- 如果图中有环路会发生什么？（拓扑排序无法完成，但推理图是 DAG 不会有环）
