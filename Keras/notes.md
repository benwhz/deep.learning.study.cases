# Keras

tensorflow图像数据格式：$(m, h, w, c)$
- $m$: batch size
- $h$: image height
- $w$: image width
- $c$: channel
- also you can set data_format in layers:
    - channels_list (default)
    - channels_first: $(m, c, h, w)$

## 1. import packages
> import tensorflow as tf

> import keras
> from keras import layers

> from tensorflow import keras
> from tensorflow.keras import layers

## 2. 序列模型（Sequential model）
1. 直接构造keras.Sequential([l1, l2, l3 ...])，传入层列表作为参数；
1. 构造空层的keras.Sequential实例，然后通过add(layer)方法添加Layer，通过pop()删除层；
1. 等效于层套层：y = layer3(layer2(layer1(x)))；
1. len(model.layers)返回层数量；
1. 不指定输入的情况下，没用有效权重；可以通过两种方式使得权重有效：
    1. 指定输入的shape，通过添加Input（输入Input不是层，不包含在层计数中）可以使得模型有有效权重。
    2. 指定输入。
1. 可以查看每一层的输出
1. 用于迁移学习：迁移学习包括冻结模型中的底层并仅训练顶层。

## 3. 函数式 API
函数式 API 是一种比 tf.keras.Sequential API 更加灵活的模型创建方式。函数式 API 可以处理具有非线性拓扑的模型、具有共享层的模型，以及具有多个输入或输出的模型。
1. 通过指定keras.Model的输入和输出来创建模型；
1. 步骤：
    - 定义模型：定义输入、输出以及中间各层的连接方式；
    - 编译模型：指定Loss函数、优化函数以及输出测量值等；
    - 训练：调用内置训练循环（fit() 方法）
    - 评估：调用内置评估循环（evaluate() 方法）
1. Conv2D | Conv2DTranspose（卷积与卷积转置）/ MaxPooling2D | UpSampling2D
1. 所有模型均可像层一样被调用

## 4. 通过子类化创建新的层和模型
1. Layer 类：状态（权重）和部分计算的组合
    - _init_ 函数中初始化权重
    - _call_ 函数中参数为输入，返回输出
1. 层可以具有不可训练权重：（weights，non_trainable_weights，trainable_weights）    
1. 将权重创建推迟到得知输入的形状之后：在许多情况下，您可能事先不知道输入的大小，并希望在得知该值时（对层进行实例化后的某个时间）再延迟创建权重。在 Keras API 中，我们建议您在层的 build(self, inputs_shape) 方法中创建层权重。
1. 层可递归组合形成Block：如果将层实例分配为另一个层的特性，则外部层将开始跟踪内部层创建的权重。我们建议在 __init__() 方法中创建此类子层，并将其留给第一个 __call__() 以触发构建它们的权重。
1. __add_loss__()
1. __add_metric__()