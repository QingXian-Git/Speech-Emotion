import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from trainer import train
from predictor import ConfidencePredictor
from config import CLASS_LABELS, DATA_PATH, CONFIDENCE_WEIGHTS, API_CONFIG


def print_menu():
    print(f"\n{'=' * 60}")
    print("       面试自信度检测系统")
    print(f"{'=' * 60}")
    print("1. 训练模型")
    print("2. 分析音频自信度")
    print("3. 设置评估权重")
    print("4. 启动 API 服务")
    print("5. 查看当前配置")
    print("0. 退出")
    print(f"{'=' * 60}")


def get_weight_input(prompt: str, default: float) -> float:
    while True:
        try:
            value = input(f"{prompt} (默认: {default}): ").strip()
            if value == "":
                return default
            return float(value)
        except ValueError:
            print("请输入有效的数字")


def main():
    predictor = None

    while True:
        print_menu()
        choice = input("请选择操作: ").strip()

        if choice == '1':
            print("\n开始训练模型...")
            train()

        elif choice == '2':
            try:
                if predictor is None:
                    print("\n正在加载模型...")
                    predictor = ConfidencePredictor()
            except FileNotFoundError as e:
                print(f"错误: {e}")
                print("请先选择 '1' 训练模型")
                continue

            audio_path = input("请输入音频文件路径: ").strip().strip('"\'')

            try:
                result = predictor.predict(audio_path, show_plot=True, save_plot=True)
            except FileNotFoundError as e:
                print(f"错误: {e}")

        elif choice == '3':
            print("\n设置评估权重 (三个权重之和应为1.0)")
            emotion_w = get_weight_input("情感识别权重", 0.5)
            speech_w = get_weight_input("语速稳定性权重", 0.3)
            pause_w = get_weight_input("停顿频率权重", 0.2)

            if predictor is None:
                try:
                    predictor = ConfidencePredictor(
                        emotion_weight=emotion_w,
                        speech_rate_weight=speech_w,
                        pause_weight=pause_w
                    )
                except FileNotFoundError:
                    from confidence_evaluator import ConfidenceEvaluator
                    predictor = ConfidencePredictor.__new__(ConfidencePredictor)
                    predictor.evaluator = ConfidenceEvaluator(
                        emotion_weight=emotion_w,
                        speech_rate_weight=speech_w,
                        pause_weight=pause_w
                    )
            else:
                predictor.set_weights(emotion_w, speech_w, pause_w)

            print("权重设置完成!")

        elif choice == '4':
            print("\n启动 API 服务...")
            os.system(f"python api.py")

        elif choice == '5':
            print("\n当前配置:")
            print(f"  数据集路径: {DATA_PATH}")
            print(f"  情感类别: {CLASS_LABELS}")
            print(f"  API 配置:")
            print(f"    - 地址: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
            print(f"    - 文档: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
            print(f"  默认权重:")
            print(f"    - 情感识别: {CONFIDENCE_WEIGHTS['emotion']}")
            print(f"    - 语速稳定性: {CONFIDENCE_WEIGHTS['speech_rate']}")
            print(f"    - 停顿频率: {CONFIDENCE_WEIGHTS['pause_frequency']}")

        elif choice == '0':
            print("感谢使用，再见!")
            break
        else:
            print("无效选择，请重新输入")


if __name__ == '__main__':
    main()