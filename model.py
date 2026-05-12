import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D, LSTM, Dense, Dropout,
        BatchNormalization, Activation, Reshape,
        Bidirectional
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except ImportError:
    from keras.models import Model
    from keras.layers import (
        Input, Conv2D, MaxPooling2D, LSTM, Dense, Dropout,
        BatchNormalization, Activation, Reshape,
        Bidirectional
    )
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import TRAINING_CONFIG, NUM_CLASSES, MODELS_DIR


class EmotionRecognitionModel:
    def __init__(self, input_shape: tuple = None, num_classes: int = NUM_CLASSES):
        """
        初始化情感识别模型

        参数:
            input_shape: 输入特征形状 (时间步, 特征数)
            num_classes: 情感类别数量
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        if input_shape is not None:
            self.model = self._build_model()
            self._compile_model()

    def _build_model(self) -> Model:
        # 构建CNN-LSTM混合模型架构
        inputs = Input(shape=self.input_shape)

        # 重塑为CNN所需的4D张量 (样本, 时间, 特征, 通道)
        x = Reshape((self.input_shape[0], self.input_shape[1], 1))(inputs)

        # CNN层 - 提取局部特征
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)   #增加非线性
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)

        # 重塑为LSTM所需的3D张量
        shape_before_lstm = x.shape
        x = Reshape((shape_before_lstm[1], -1))(x)

        # LSTM层 - 提取时序特征
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.4)(x)

        x = Bidirectional(LSTM(64, return_sequences=False))(x)
        x = Dropout(0.4)(x)

        # 全连接层 - 分类
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def _compile_model(self):
        optimizer = Adam(learning_rate=TRAINING_CONFIG['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def summary(self):
        return self.model.summary()

    def train(self, x_train, y_train, x_val, y_val, epochs=None, batch_size=None):
        """
        训练模型

        参数:
            x_train: 训练数据
            y_train: 训练标签
            x_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小

        返回:
            训练历史记录
        """
        epochs = epochs or TRAINING_CONFIG['epochs']
        batch_size = batch_size or TRAINING_CONFIG['batch_size']

        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=0
            ),
            ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]

        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=2
        )

        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def save_model(self, model_name: str = 'emotion_model'):
        keras_path = os.path.join(MODELS_DIR, f'{model_name}.keras')
        self.model.save(keras_path)
        print(f"模型已保存: {keras_path}")

    @staticmethod
    def load_model(model_name: str = 'emotion_model'):
        keras_path = os.path.join(MODELS_DIR, f'{model_name}.keras')

        model = tf.keras.models.load_model(keras_path)

        print(f"模型已加载: {keras_path}")
        return model