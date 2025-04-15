import numpy as np
import time

# 初始化參數
matrix_size = (5, 5)  # 一級神經元的矩陣大小
threshold = 5  # 一級神經元初始閾值
threshold_limit = 20  # 閾值上限
initial_threshold = 5  # 初始閾值不應該低於此值
rest_period = 1  # 初始休息時間 1 秒
num_matrices = 30  # 總共輸入 30 個矩陣
rest_counter = np.zeros(matrix_size, dtype=int)  # 用來記錄每個一級神經元的休息期
output_history = []  # 記錄輸出歷史


# 計算輸出結果
def neuron_output(neurons_matrix, rest_counter, threshold):
    # 計算輸出 1 的神經元數量
    active_count = np.sum(neurons_matrix == 1)

    # 檢查是否超過閾值
    if active_count > threshold:
        output = 1  # 輸出 1
    else:
        output = 0  # 輸出 0

    return output


# 檢查4秒內頻繁輸出1的情況
def check_frequent_outputs(output_history, current_time, interval):
    count = sum(1 for time_stamp in output_history if current_time - time_stamp <= interval)
    return count


# 計算休息時間的變動
def update_rest_period(threshold, rest_period, initial_threshold):
    threshold_increase_percentage = (threshold - initial_threshold) / initial_threshold
    rest_period_increase_percentage = threshold_increase_percentage * 0.05  # 每 10% 閾值增加 5% 休息時間
    rest_period = 1 + rest_period_increase_percentage  # 休息時間的變動
    return rest_period


# 主運行邏輯
def multi_layer_neuron(num_matrices, threshold, rest_period):
    first_level_thresholds = []
    first_level_rest_periods = []
    second_level_input = []  # 用來記錄一級神經元的輸出，為二級神經元提供輸入

    for i in range(num_matrices):
        neurons_matrix = np.random.randint(0, 2, matrix_size)
        print(f"\nMatrix {i + 1}:\n{neurons_matrix}")

        # 計算每個一級神經元的輸出
        output = neuron_output(neurons_matrix, rest_counter, threshold)

        # 記錄一級神經元的輸出結果
        second_level_input.append(output)

        # 如果輸出1，記錄當前時間
        current_time = time.time()
        if output == 1:
            output_history.append(current_time)

        # 檢查4秒內輸出 1 的情況
        if check_frequent_outputs(output_history, current_time, 4) > 2:
            if threshold < threshold_limit:
                threshold += 1
                print(f"Threshold increased to {threshold}.")
                rest_period = update_rest_period(threshold, rest_period, initial_threshold)
                print(f"Rest period updated to {rest_period:.2f} seconds.")
            else:
                print(f"Threshold has reached its limit of {threshold_limit}.")
        elif check_frequent_outputs(output_history, current_time, 3) == 0:
            # 3 秒內沒有輸出 1 則閾值減少
            if threshold > initial_threshold:
                threshold -= 1
                print(f"Threshold decreased to {threshold}.")
                rest_period = update_rest_period(threshold, rest_period, initial_threshold)
                print(f"Rest period updated to {rest_period:.2f} seconds.")
            else:
                print(f"Threshold is at the minimum value of {initial_threshold} and cannot decrease further.")

        # 記錄一級神經元的最終閾值和休息時間
        first_level_thresholds.append(threshold)
        first_level_rest_periods.append(rest_period)

    # 將一級神經元的輸出結果轉換為二級神經元的輸入矩陣
    second_level_matrix = np.array(second_level_input).reshape(matrix_size)
    print(f"\nSecond-level neuron matrix:\n{second_level_matrix}")

    # 計算二級神經元的輸出
    second_level_output = neuron_output(second_level_matrix, rest_counter, threshold)
    print(f"\nSecond-level neuron output: {second_level_output}")

    # 輸出所有一級神經元的最終閾值和休息時間
    print("\nFirst-level neurons final thresholds:")
    print(first_level_thresholds)
    print("\nFirst-level neurons final rest periods:")
    print(first_level_rest_periods)

    # 輸出二級神經元的最終閾值和休息時間
    print(f"\nSecond-level neuron final threshold: {threshold}")
    print(f"Second-level neuron final rest period: {rest_period:.2f} seconds")


# 運行二級神經元的計算
multi_layer_neuron(num_matrices=25, threshold=threshold, rest_period=rest_period)
