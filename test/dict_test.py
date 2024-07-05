from collections import defaultdict

# 初始化 defaultdict，值为列表类型
grouped_data = defaultdict(list)

# 示例数据 (类别标签, 值)
data_points = [('fruit', 'apple'), ('fruit', 'banana'), ('vegetable', 'carrot'), ('fruit', 'pear'), ('vegetable', 'lettuce')]

# 将数据点添加到对应的类别列表中
for category, value in data_points:
    grouped_data[category].append(value)

# 打印分组后的数据
for category, values in grouped_data.items():
    print(f"{category}: {values}")
