# Day 11: 前向执行与数据传播

## 目标

理解 Forward 的完整执行流程——算子按拓扑序依次执行，数据通过 `shared_ptr<Tensor>` 零拷贝传播。

## 核心概念

### Forward 主循环

```cpp
void RuntimeGraph::Forward() {
    // 按 start_time 升序排列
    sort(operators by start_time);

    for (auto& op : sorted_operators) {
        if (op is Input) {
            // 将外部输入注入图的输出映射表
            continue;
        }
        if (op is Output) {
            continue;
        }
        // 从上游收集输入
        collect_inputs(op);
        // 执行算子
        op->layer->Forward(inputs, outputs);
        // 将输出传播到下游
        propagate_outputs(op, outputs);
    }
}
```

### 零拷贝数据流

`shared_ptr<Tensor>` 传递的代价仅是引用计数 +1，实际数据在原地不动。
一个 Tensor 可能被多个下游算子共享，引用计数自动管理生命周期。

## 本天 Demo

在 Day 10 的基础上，新增：
1. **完整前向执行演示**：线性链从输入到输出的完整数据流，每一步打印形状变化
2. **数据传播追踪**：每个节点打印输入输出形状，追踪数据如何流动
3. **多输出分支消费**：同一个输出被两个算子消费（引用计数 +2）
4. **前向执行性能计时**：对整条图和单个算子分别计时

## 构建与运行

```bash
mkdir build && cd build
cmake .. && make && ./day11
```

## 思考题

- 如果两个算子共享同一个输入 Tensor，零拷贝会有什么影响？（`shared_ptr` 的引用计数 +1，数据不会被提前释放）
- Forward 中可以插入算子级计时吗？（可以，Debug 模式下有 `LayerTimeLogging`）
