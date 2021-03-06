# 参数说明
- feedback_bits=512（反馈比特量）
- img_height = 126（CSI图像高）
- img_width = 128 （CSI图像高）
- img_channels= 2 （CSI图像通道数）

# 建模算力与环境
NVIDIA A100

# 运行方法
1. 在同一级目录下创建Modelsave文件夹
2. 运行Model_train.py，可以在TensorBoardX中查看train_loss、学习率、验证集NMSE的变化曲线
3. 在Modelsave文件夹中含有生成的结果

# 内含文件
- `Model_define_pytorch.py`
- `Model_train.py`
- `Model_evaluation_encoder.py`
- `Model_evaluation_decoder.py`

## 文件说明
### Model_define_pytorch.py
设计网络结构，设计Encoder与Decoder函数。
- Encoder函数，定义编码器模型。输入原始信道状态信息（x）与反馈比特量(feedback_bits)，输出为比特流编码向量。
  - 5x5卷积核
  - SiLU()
  - SiLU()
  - 全连接层
  - Sigmoid()
  - 量化层
- Decoder函数，定义解码器模型。输入比特流编码向量（x）与反馈比特量(feedback_bits)，输出为重建的CSI。
  - 解量化
  - 全连接层
  - Sigmoid()
  - 5个RefineBlock
    - 7x7卷积
    - 5x5卷积
    - 3x3卷积
  - 3x3卷积
  - Sigmoid()

### Model_train.py
模型训练 pipeline 参考代码

### Model_evaluation_encoder.py
编码模型的推理参考代码，能成功推理是成功提交的基础保证。

### Model_evaluation_decoder.py
解码模型的推理参考代码，能成功推理是成功提交的基础保证。