import mediapipe as mp
import cv2
import csv
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# CSVファイル名
csv_file = "hand_coordinates_and_movements.9csv"

# CSVファイルにヘッダーを書き込む
def write_header():
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['Timestamp', 'Hand_Landmark', 'X', 'Y', 'Z', 'Delta_X', 'Delta_Y', 'Delta_Z']
        writer.writerow(header)

# CSVファイルにデータを書き込む
def write_data(timestamp, hand_landmark, x, y, z, delta_x, delta_y, delta_z):
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        data_row = [timestamp, hand_landmark, x, y, z, delta_x, delta_y, delta_z]
        writer.writerow(data_row)

# メインの処理
def main():
    write_header()
    video_path='./sleep6.mp4'
    cap= cv2.VideoCapture(video_path)  # カメラを起動

    prev_landmarks = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.3) as hands:

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks and results.multi_hand_landmarks[0]:
                for hand_landmarks in results.multi_hand_landmarks:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        cx, cy, cz = landmark.x, landmark.y, landmark.z

                        # 前回のフレームとの差を計算
                        if prev_landmarks is not None:
                            prev_landmark = prev_landmarks[0].landmark[idx]
                            delta_x = cx - prev_landmark.x
                            delta_y = cy - prev_landmark.y
                            delta_z = cz - prev_landmark.z
                        else:
                            delta_x, delta_y, delta_z = 0, 0, 0

                        timestamp = time.time()  # フレームの時間をタイムスタンプとして使用

                        write_data(timestamp, idx, cx, cy, cz, delta_x, delta_y, delta_z)

                    prev_landmarks = results.multi_hand_landmarks

            # 手の形の描画
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
