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

快速选择算法基于两种partition算法实现，在两种不同的partition算法会产生完全不同的算法实现， 深入了解不同的划分方式差异，对算法理解和实现有重要作用。

## 两类划分算法

- [Lomuto partition scheme](https://en.wikipedia.org/wiki/Quickselect)
- [Hoare partition scheme](https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme)

与 Hoare partition 不同的是 **Lomutopartition算法返回pivot的下标k**，
partition算法将**小于等于**pivot的元素放第k个元素左边，**大于**pivot的元素放在第k个位置右边，
因此第k小的元素就是pivot自己。

## Lomuto Partition Scheme:

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

- 平均降低三倍swap次数
- 即使当list中的所有值都相等的时候，Hoare partition方式能产生均衡的划分

```c++
int partition(vector<int> &nums, int l, int r){
  int partition = nums[l], i = l - 1, j = r + 1;
  while (i < j) {
      do i++; while (nums[i] < partition);
      do j--; while (nums[j] > partition);
      if (i < j)
          swap(nums[i], nums[j]);
  }
  // 循环结束时, j+1 == i, [l, j]之间的数**小于等于**pivot, [i,r]之间的数**大于等于**pivot
  return j;
}
// 第k小的元素
int quickSelect(vector<int>& a, int l, int r, int k){
    // 因为partition算法返回的不是pivot的index，因此需要遍历完数组，当只剩下一个元素时，就是第k小的元素
    if(l==r) return a[k];
    int q = partition(a, l, r);
    if(k <= j) quickSelect(a, l, j, k);
    else quickSelect(a, j+1, r, k);
}

```

[leecode problems - 215](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2&envId=top-interview-150)

$$ E = mc^2 $$

