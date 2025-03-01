# cyh-igem pku igem比赛试题
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv1D, LSTM, Dense, Dropout, concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ====================== 1. 注意力层 ======================
class HybridAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(HybridAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel_axis = -1
        channel = input_shape[self.channel_axis]
        
        # 通道注意力
        self.channel_dense1 = Dense(channel // self.ratio, activation='relu')
        self.channel_dense2 = Dense(channel)
        
        # 空间注意力
        self.spatial_conv = Conv1D(1, kernel_size=7, padding='same', activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        # 通道注意力
        channel_avg = tf.reduce_mean(inputs, axis=1, keepdims=True)
        channel_max = tf.reduce_max(inputs, axis=1, keepdims=True)
        channel = concatenate([channel_avg, channel_max], axis=-1)
        channel = self.channel_dense1(channel)
        channel = self.channel_dense2(channel)
        channel_weights = tf.nn.sigmoid(channel)
        
        # 空间注意力
        spatial_weights = self.spatial_conv(inputs)
        
        # 合并注意力
        weighted = inputs * (channel_weights + spatial_weights)
        return weighted

# ====================== 2. 数据预处理 ======================
def preprocess_sequences_v2(sequences, max_length=100):
    base_map = {'A':0, 'T':1, 'C':2, 'G':3}
    encoded = np.zeros((len(sequences), max_length, 5))  # 新增第5位作为长度标记
    
    for i, seq in enumerate(sequences):
        effective_length = min(len(seq), max_length)
        for j in range(effective_length):
            if seq[j] in base_map:
                encoded[i, j, base_map[seq[j]]] = 1
        # 标记超长序列
        if len(seq) > max_length:
            encoded[i, -1, 4] = 1  # 第5位标记超长
    return encoded[:, :, :4]  # 仅返回前4位用于模型输入

# ====================== 3. 模型构建 ======================
def build_hmpl_model(input_shape=(100,4)):
    inputs = Input(shape=input_shape)
    
    # 多尺度卷积
    conv_large = Conv1D(32, 9, padding='same', activation='relu')(inputs)
    conv_medium = Conv1D(32, 5, padding='same', activation='relu')(inputs)
    conv_small = Conv1D(32, 3, padding='same', activation='relu')(inputs)
    merged_conv = concatenate([conv_large, conv_medium, conv_small], axis=-1)
    pooled_conv = tf.keras.layers.MaxPooling1D(2)(merged_conv)
    
    # LSTM分支
    lstm_branch = LSTM(64, return_sequences=True)(pooled_conv)
    lstm_branch = Dropout(0.4)(lstm_branch)
    
    # 注意力
    attention = HybridAttention()(lstm_branch)
    
    # 深度监督
    mid_output = GlobalAveragePooling1D()(attention)
    mid_output = Dense(1, activation='sigmoid', name='mid_output')(mid_output)
    
    # 最终输出
    final_output = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(attention)
    final_output = Dropout(0.5)(final_output)
    final_output = Dense(1, activation='sigmoid', name='final_output')(final_output)
    
    model = Model(inputs=inputs, outputs=[final_output, mid_output])
    
    model.compile(
        optimizer=Adam(0.001),
        loss={'final_output': 'binary_crossentropy', 'mid_output': 'binary_crossentropy'},
        loss_weights={'final_output': 0.7, 'mid_output': 0.3},
        metrics={'final_output': ['accuracy']}
    )
    return model

# ====================== 4. 主程序 ======================
if __name__ == "__main__":
    # 加载数据
    dataset = load_dataset('dna_core_promoter')
    df = dataset['train'].to_pandas()
    
    # 预处理
    X = preprocess_sequences_v2(df['sequence'])
    y = df['label'].values.astype(np.float32)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 构建模型
    model = build_hmpl_model()
    model.summary()
    
    # 训练
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        X_train, {'final_output': y_train, 'mid_output': y_train},
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 评估
    y_pred, _ = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_pred))
    
    # 可视化
    plt.plot(history.history['final_output_accuracy'])
    plt.plot(history.history['val_final_output_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
