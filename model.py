import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# CSVファイルのパス
csv_file = "hand_coordinates_and_movements.csv"

# CSVファイルからデータを読み込む関数
def load_data_from_csv(csv_file):
    X_dataset = []
    y_dataset = []

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダーをスキップ

        for row in reader:
            # データをパースしてXとyに分割
            # 1列目がジェスチャーID、2列目以降のデータを読み込む
            gesture_id, timestamp, hand_landmark, x, y, z, delta_x, delta_y, delta_z = map(float, row)

            X_dataset.append([float(x), float(y), float(z), float(delta_x), float(delta_y), float(delta_z)])
            y_dataset.append(gesture_id)

    return np.array(X_dataset), np.array(y_dataset)

# データを読み込む
X, y = load_data_from_csv(csv_file)

# データの前処理
# 例: 正規化
X = (X - X.mean(axis=0)) / X.std(axis=0)

# ジェスチャーIDを数値に変換 (文字列を数値にマッピング)
unique_gesture_ids = np.unique(y)
gesture_id_mapping = {gesture_id: i for i, gesture_id in enumerate(unique_gesture_ids)}
y = np.array([gesture_id_mapping[gesture_id] for gesture_id in y])

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの定義
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(6,)),  # 入力の次元数は6 (座標とDelta_X/Y/Zの6つの特徴)
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')  # 出力クラス数は4 (4つのジェスチャーIDの分類)
])

# モデルのコンパイル
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# モデルの訓練
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# モデルの評価
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# 予測したいデータを取得
data_to_predict = np.array([X[10000]])  # 予測したいデータをここに入れる

# モデルの予測
predict_result = model.predict(data_to_predict)


print("Prediction Result:", predict_result[0].tolist())
predicted_gesture_id = unique_gesture_ids[np.argmax(np.squeeze(predict_result))]

print("Predicted Gesture ID:", unique_gesture_ids[np.argmax(predict_result)])
# 予測結果を元のジェスチャーIDに変換


# テストデータのジェスチャーIDと予測結果を表示
print("True Gesture ID:", unique_gesture_ids[y[10000]])
print("Predicted Gesture ID:", predicted_gesture_id)

# モデルの保存
model.save("hand_gesture_classifier_model.h5")
