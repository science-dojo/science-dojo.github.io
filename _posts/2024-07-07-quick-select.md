---
title: Quick-select k-th smallest elements in lists
tags:
  - Quick Sort Algorithm
categories:
  - Algorithm
layout: single
toc: true
toc_label: "开始"
---

快速选择算法基于两种partition算法(Lomuto,Hoare)实现，在两种不同的partition算法会产生完全不同的算法实现，深入了解不同的划分方式差异，对算法理解和实现有重要作用。
接下来将从pivot选择、机制、pivot最终位置、性能和稳定性上分析两者不同。


- [Lomuto partition scheme](https://en.wikipedia.org/wiki/Quickselect)
- [Hoare partition scheme](https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme)


与 Hoare partition 主要不同的是 **Lomutopartition算法返回pivot的下标k**，
partition算法将**小于等于**pivot的元素放第k个元素左边，**大于**pivot的元素放在第k个位置右边，
因此第k小的元素就是pivot自己；

## Lomuto Partition Scheme:
- pivot选择：算法导论中选择最右边，严蔚敏教材中实现选择最左边。
- 机制：双指针，指针`i`指向元素小于`pivot`的位置，指针`j`指向小于等于`pivot`的位置；`i`自增加1，交换`i`和`j`指向的元素。交换完毕后，`i`指向的元素小于`pivot`, `j`自增加1，查看一个位置和`pivot`的大小。当`j`指向最后一个元素时（满足小于等于`pivot`）必然交换`++i` 和 `j`指向的元素。
- 返回位置：返回`i`，指向pivot在排序后的真实位置。
- 性能：比Hoare慢，因为由更多的swap操作，特别当由很多重复元素时，最坏实际复杂度 $$O(n)$$。
- 稳定性：半稳定，不改变**相等元素**的相对位置。

```c++
// implementation of <Introduction of algorithm> (从小到大排序)
int partition(vector<int>& a, int l, int r) {
    int x = a[r], i = l - 1;
    for (int j = l; j < r; ++j) {
        if (a[j] <= x) {
            swap(a[++i], a[j]);
        }
    }
    swap(a[++i], a[r]);
    return i;
}
// implementation of yanweimin <data structure> (从大到小排序)
int partition2(vector<int>& nums, int l, int r) {
  int pt = nums[l], i = l, j = r;
  int t = nums[l];
  while (i != j) {
      while(i < j && nums[j] <= pt) --j;
      nums[i] = nums[j];
      while(i < j && nums[i] >= pt) ++i;
      nums[j] = nums[i];
  }
  nums[i] = t;
  return i;
}
int randomPartition(vector<int>& a, int l, int r) {
    int i = rand() % (r - l + 1) + l;
    swap(a[i], a[r]);
    return partition(a, l, r);
}

// 第k小的元素
int quickSelect(vector<int>& a, int l, int r, int index){
    while(true) {
        int q = randomPartition(a, l, r);
        if(q == index) return a[q];
        else if (q < index) l = q + 1;
        else r = q-1;
    }
}
```


## Hoare Partition Scheme

Hoare's scheme 比 Lomuto's partition scheme 更高效:

- Pivot选择：通常选取第一个元素，但也有其他更有效的选择方法，比如随机选择，三数取中
- 机制：维护两指针`i`和`j`, 分别指向小于和大于pivot的位置；`i`和`j`分别数组开头和末尾移动，直到**小于**和**大于**关系不满足，然后交换`i`和`j`指向的元素。
- 返回位置: 返回的位置不能保证是pivot在排序后的最终位置。
- 性能：平均降低三倍swap次数，即使当list中的所有值都相等的时候，Hoare partition方式能产生均衡的划分
- 稳定性：不稳定，可能改变相同元素的相对顺序。


```c++
int partition(vector<int> &nums, int l, int r){
  int partition = nums[l], i = l - 1, j = r + 1;
  while (i < j) {
      do i++; while (nums[i] < partition);
      do j--; while (nums[j] > partition);
      if (i < j)
          swap(nums[i], nums[j]);
  }
  // 循环结束时, j == i, [l, j]之间的数**小于等于**pivot, [i,r]之间的数**大于等于**pivot
  return j;
}
// 第k小的元素
int quickSelect(vector<int>& a, int l, int r, int k){
    // 因为partition算法返回的不是pivot的index，因此需要遍历完数组，当只剩下一个元素时，就是第k小的元素
    if(l==r) return a[k];
    int q = partition(a, l, r);
    if(k <= q) quickSelect(a, l, q, k);
    else quickSelect(a, q+1, r, k);
}

int quickSelect(vector<int>& a, int l, int r, int index){
    while(true) {
        if(l==r) return a[index];
        int q = partition(a, l, r);
        // if(q == index) return a[q];
        if (index <= q) r = q;
        else l = q+1;
    }
}

```

[练习题leecode problems - 215](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2&envId=top-interview-150)

