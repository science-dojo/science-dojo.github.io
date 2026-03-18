---
title: Minimum Stack,Queue and Deque
tags:
  - interview
  - Data structure
layout: single
toc: true
toc_sticky: true
toc_label: "目录"
---
最小栈、最小队列和最小双端队列的实现。要求在O(1)时间内获得栈、队列和双端队列中的最小值，要求在数据结果的更新中动态的维护最小值。

# 最小栈

```c++
struct minstack {
 stack<pair<int, int>> st;
 int getmin() {return st.top().second;}
 bool empty() {return st.empty();}
 int size() {return st.size();}
 void push(int x) {
  int mn = x;
  if (!empty()) mn = min(mn, getmin());
  st.push({x, mn});
 }
  void pop() {st.pop();}
 int top() {return st.top().first;}
 void swap(minstack &x) {st.swap(x.st);}
};
```

- 参考题目
  - [leetcode:最小栈]()

# 最小双端队列

```c++
struct mindeque {
 minstack l, r, t;
 void rebalance() {
  bool f = false;
  if (r.empty()) {f = true; l.swap(r);}
  int sz = r.size() / 2;
  while (sz--) {t.push(r.top()); r.pop();}
  while (!r.empty()) {l.push(r.top()); r.pop();}
  while (!t.empty()) {r.push(t.top()); t.pop();}
  if (f) l.swap(r);
 }
 int getmin() {
  if (l.empty()) return r.getmin();
  if (r.empty()) return l.getmin();
  return min(l.getmin(), r.getmin());
 }
 bool empty() {return l.empty() && r.empty();}
 int size() {return l.size() + r.size();}
 void push_front(int x) {l.push(x);}
 void push_back(int x) {r.push(x);}
 void pop_front() {if (l.empty()) rebalance(); l.pop();}
 void pop_back() {if (r.empty()) rebalance(); r.pop();}
 int front() {if (l.empty()) rebalance(); return l.top();}
 int back() {if (r.empty()) rebalance(); return r.top();}
 void swap(mindeque &x) {l.swap(x.l); r.swap(x.r);}
};
```

最坏场景：交替执行pop_front和pop_back。例如所有元素都在右栈，pop_front触发全量元素倒入左栈；紧接着pop_back又触发全量元素倒回右栈，每次操作都要搬运所有元素，总时间复杂度退化到 O (N²)，也就是文中说的too slow

**优化核心思路**

- 要每次空栈时全量搬运元素，而是每次只搬运一半元素，将非空栈的元素对半拆分，一半留在原栈、一半倒入空栈。最终可以证明，该方案总时间复杂度为均摊 O (N)（每个操作均摊 O (1)）

证明:

- $k_i$：第i次栈间元素搬运（rebalance）前，双端队列的总元素数。
- 算法总时间复杂度为$O(N) + sum_{k_i}$：N是所有 push/pop 基础操作的总次数，$sum_{k_i}$是所有搬运操作的总次数（搬运耗时和元素数量成正比)
- 递推关系：$k_i = k_{i-1}/2 + q_i$。其中$q_i$是第i-1次和第i次搬运之间，新加入队列的元素数。核心逻辑是：上一次搬运后元素被对半拆分，下一次触发搬运时，最多只有$k_{i-1}/2$个元素残留，加上期间新增的$q_i$，就是本次搬运前的总元素数
- 等比数列求和：每个新增的$q_i$，对总搬运次数的贡献是$q_i + q_i/2 + q_i/4 + ... = 2q_i = O(q_i)$（等比数列和收敛）
- 总复杂度：所有$q_i$的和≤N（最多新增 N 个元素），因此$∑k_i = O(N)$，总时间复杂度为均摊 O (N)

# 最小队列

## 队列版本

```python
from collections import deque

class MinQueue:
    def __init__(self):
        self.data_queue = deque()  # 存储所有元素
        self.min_deque = deque()   # 维护单调非递减序列

    def push(self, x: int) -> None:
        """元素入队"""
        self.data_queue.append(x)
        # 维护min_deque：弹出尾部所有大于x的元素，保证单调性
        while self.min_deque and self.min_deque[-1] > x:
            self.min_deque.pop()
        self.min_deque.append(x)

    def pop(self) -> int:
        """元素出队（队首）"""
        val = self.data_queue.popleft()
        # 如果出队的是当前最小值，同步更新min_deque
        if val == self.min_deque[0]:
            self.min_deque.popleft()
        return val

    def getMin(self) -> int:
        """获取队列最小值"""
        return self.min_deque[0]
```

## 双栈版本

```python
class MinQueueUsingStacks:
    def __init__(self):
        self.in_stack = MinStack()
        self.out_stack = MinStack()

    def push(self, x: int) -> None:
        """元素入队（队尾）"""
        self.in_stack.push(x)

    def pop(self) -> int:
        """元素出队（队首）"""
        if self.out_stack.is_empty():
            # 将in_stack的元素全部倒入out_stack
            while not self.in_stack.is_empty():
                self.out_stack.push(self.in_stack.pop())
        return self.out_stack.pop()

    def getMin(self) -> int:
        """获取队列最小值"""
        # 分情况讨论两个栈的状态
        if self.in_stack.is_empty():
            return self.out_stack.getMin()
        if self.out_stack.is_empty():
            return self.in_stack.getMin()
        return min(self.in_stack.getMin(), self.out_stack.getMin())
```

# 参考文档

- (灵神题解)(<https://leetcode.cn/problems/min-stack/solutions/2974438/ben-zhi-shi-wei-hu-qian-zhui-zui-xiao-zh-x0g8/?envType=study-plan-v2&envId=top-100-liked>)
- [tutorial minimum Deque](https://codeforces.com/blog/entry/122003)
