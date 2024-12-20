import time
import random
import math
import matplotlib.pyplot as plt
import statistics
# Поиск: реализация алгоритмов
def linear_search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1
def binary_search(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1
def interpolation_search(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high and arr[low] <= x <= arr[high]:
        if low == high:
            if arr[low] == x:
                return low
            return -1
        pos = low + ((x - arr[low]) * (high - low) // (arr[high] - arr[low]))
        if arr[pos] == x:
            return pos
        elif arr[pos] < x:
            low = pos + 1
        else:
            high = pos - 1
    return -1
def exponential_search(arr, x):
    if arr[0] == x:
        return 0
    index = 1
    while index < len(arr) and arr[index] <= x:
        index *= 2
    return binary_search(arr[:min(index, len(arr))], x)
def jump_search(arr, x):
    length = len(arr)
    step = int(math.sqrt(length))
    prev = 0
    while arr[min(step, length) - 1] < x:
        prev = step
        step += int(math.sqrt(length))
        if prev >= length:
            return -1
    for i in range(prev, min(step, length)):
        if arr[i] == x:
            return i
    return -1

# Генерация массивов
def generate_array(size, seed=42):
    random.seed(seed)
    return random.sample(range(1, size * 10), size)
def generate_full_sorted_array(size):
    return sorted(generate_array(size))
def generate_full_sorted_array_desc(size):
    return sorted(generate_array(size), reverse=True)
#Измерение времени
def measure_time(search_function, arr, repetitions=1000):
    # Фиксируем элемент для поиска (например, средний элемент)
    x = arr[(len(arr) // 2) + 3]

    # Прогрев функции (например, 5 прогонов)
    for _ in range(5):
        search_function(arr, x)

    # Измерение времени с повторениями
    times = []
    for _ in range(repetitions):
        start_time = time.perf_counter()
        search_function(arr, x)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return sum(times) / len(times)
def measure_time_median(search_function, arr, repetitions=1000):

    # Фиксируем элемент для поиска (например, средний элемент)
    x = arr[(len(arr) // 2)+3]

    # Прогрев функции (например, 5 прогонов)
    for _ in range(5):
        search_function(arr, x)

    # Измерение времени с повторениями
    times = []
    for _ in range(repetitions):
        start_time = time.perf_counter()
        search_function(arr, x)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    # Возвращаем медиану времени
    return statistics.median(times)

#Построение графиков временной сложности
def plot_Onation_linear(sizes):
    complexities = {
        "Лучший случай O(1)":lambda n:1,
        "Средний и худший случай O(n)": lambda n: n
    }

    plt.figure(figsize=(12, 8))
    for label, func in complexities.items():
        cases = [func(size) for size in sizes]
        plt.plot(sizes, cases, label=label)

    plt.xlabel("Размер массива")
    plt.ylabel("Оценка времени выполнения")
    plt.title("Теоретические графики временной сложности для Линейного поиска")
    plt.legend()
    plt.grid()
    plt.show()
def plot_Onation_binary(sizes):
    complexities = {
        "Лучший случай O(1)":lambda n:1,
        "Средний и худший случай O(logn)": lambda n: math.log(n)
    }

    plt.figure(figsize=(12, 8))
    for label, func in complexities.items():
        cases = [func(size) for size in sizes]
        plt.plot(sizes, cases, label=label)

    plt.xlabel("Размер массива")
    plt.ylabel("Оценка времени выполнения")
    plt.title("Теоретические графики временной сложности для Бинарного поиска")
    plt.legend()
    plt.grid()
    plt.show()
def plot_Onation_interpolation(sizes):
    complexities = {
        "Лучший случай O(1)":lambda n:1,
        "Средний случай O(loglogn)": lambda n: math.log(math.log(n)),
        "Худший случай O(n)": lambda n: n
    }

    plt.figure(figsize=(12, 8))
    for label, func in complexities.items():
        cases = [func(size) for size in sizes]
        plt.plot(sizes, cases, label=label)

    plt.xlabel("Размер массива")
    plt.ylabel("Оценка времени выполнения")
    plt.title("Теоретические графики временной сложности для Интерполяционного поиска")
    plt.legend()
    plt.grid()
    plt.show()
def plot_Onation_expo(sizes):
    complexities = {
        "Лучший случай O(1)":lambda n:1,
        "Средний и худший случай O(logn)": lambda n: math.log(n)
    }

    plt.figure(figsize=(12, 8))
    for label, func in complexities.items():
        cases = [func(size) for size in sizes]
        plt.plot(sizes, cases, label=label)

    plt.xlabel("Размер массива")
    plt.ylabel("Оценка времени выполнения")
    plt.title("Теоретические графики временной сложности для Экспоненциального поиска")
    plt.legend()
    plt.grid()
    plt.show()
def plot_Onation_jump(sizes):
    complexities = {
        "Лучший случай O(1)":lambda n:1,
        "Средний и худший случай O(n^(1/2))": lambda n: math.sqrt(n)
    }

    plt.figure(figsize=(12, 8))
    for label, func in complexities.items():
        cases = [func(size) for size in sizes]
        plt.plot(sizes, cases, label=label)

    plt.xlabel("Размер массива")
    plt.ylabel("Оценка времени выполнения")
    plt.title("Теоретические графики временной сложности для Джамп-поиска")
    plt.legend()
    plt.grid()
    plt.show()

#Построение графиков практического времени
def plot_results_by_search_type1(results, sizes):

    unique_array_types = sorted(set(result[1] for result in results))
    unique_search_types = sorted(set(result[2] for result in results))

    for search_type in unique_search_types:
        plt.figure(figsize=(10, 6))
        for array_type in unique_array_types:
            # Отбираем данные для конкретного алгоритма поиска и типа массива
            times = [result[3] for result in results if result[1] == array_type and result[2] == search_type]
            plt.plot(sizes, times, label=array_type)

        plt.title(f"Время выполнения {search_type}")
        plt.xlabel("Размер массива")
        plt.ylabel("Среднее время (в секундах)")
        plt.xscale('log')  # Логарифмическая шкала для удобства просмотра
        plt.yscale('log')  # Логарифмическая шкала для времени
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()  # Упорядочить макет
        plt.show()
def plot_results_by_search_type(results, sizes):

            unique_array_types = sorted(set(result[1] for result in results))
            unique_search_types = sorted(set(result[2] for result in results))

            for search_type in unique_search_types:
                if search_type!="Линейный поиск":
                    plt.figure(figsize=(10, 6))

                         # Отбираем данные для конкретного алгоритма поиска и типа массива
                    times = [result[3] for result in results if result[1] == "отсортированный массив" and result[2] == search_type]
                    plt.plot(sizes, times, label="отсортированный массив")

                    plt.title(f"Время выполнения {search_type}")
                    plt.xlabel("Размер массива")
                    plt.ylabel("Среднее время (в секундах)")
                    plt.xscale('log')  # Логарифмическая шкала для удобства просмотра
                    plt.yscale('log')  # Логарифмическая шкала для времени
                    plt.legend()
                    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
                    plt.tight_layout()  # Упорядочить макет
                    plt.show()
def plot_combined_graph(results, sizes):
    unique_search_types = sorted(set(result[2] for result in results))  # Уникальные алгоритмы поиска

    plt.figure(figsize=(10, 6))

    for search_type in unique_search_types:
        # Отбираем данные для конкретного алгоритма поиска для случайного массива
        times = [result[3] for result in results if result[1] == "отсортированный массив" and result[2] == search_type]
        plt.plot(sizes, times, label=search_type)  # Строим график для каждого поиска

    # Настройки графика
    plt.title(f"Сравнение алгоритмов поиска (Отсортированный массив)")
    plt.xlabel("Размер массива")
    plt.ylabel("Среднее время (в секундах)")
    plt.xscale('log')  # Логарифмическая шкала для удобства просмотра
    plt.yscale('log')  # Логарифмическая шкала для времени
    plt.legend()  # Легенда для отображения типов поиска
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()  # Упорядочить макет
    plt.show()


def apply_search():
    sizes = [100000,125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000]

    search_types = [
         ("Линейный поиск", linear_search),
         ("Бинарный поиск", binary_search),
         ("Интерполяционный поиск", interpolation_search),
         ("Экспоненциальный поиск", exponential_search),
         ("Джамп-поиск", jump_search),

    ]
    array_types = {
        "случайный массив": generate_array,
        "отсортированный массив": generate_full_sorted_array,
        "отсортированный в сторону убывания массив": generate_full_sorted_array_desc,
    }
    results = []


    for size in sizes:

        for array_name, array_func in array_types.items():

            array = array_func(size)
                # Выбираем элемент для поиска в зависимости от типа массива
                # Средний элемент для массива по убыванию
            random.choice(array)

            time_taken = measure_time(linear_search, array, repetitions=1000)
            results.append((size, array_name, "Линейный поиск", time_taken))
    plot_results_by_search_type1(results, sizes)
    for size in sizes:


                # Выбираем элемент для поиска в зависимости от типа массива
                # Средний элемент для массива по убыванию
            array = generate_full_sorted_array(size)
            for search_name, search_func in search_types:
                if search_name!= "Линейный поиск":

                    time_taken = measure_time(search_func, array, repetitions=1000)
                    results.append((size,"отсортированный массив" , search_name, time_taken))
    plot_results_by_search_type(results, sizes)
    # with open("search_time.txt", "w") as f1:
    #     # Iterate over search algorithms
    #     for search_name, search_func in search_types:
    #         f1.write(f"{search_name}\n")
    #
    #         # Iterate over array types
    #         for array_name, array_func in array_types.items():
    #             f1.write(f"  {array_name}\n")
    #
    #             # Iterate over array sizes
    #             for size in sizes:
    #
    #                 # Measure the time taken for each search algorithm
    #
    #                 time_taken = measure_time_median(search_func, array,x=0, repetitions=20)
    #
    #                 f1.write(f"    {time_taken:.4e}\n")



    plot_combined_graph(results, sizes)

    # for search_name, search_func in search_types:
    #     if search_name=="Линейный поиск":
    #         plot_Onation_linear(sizes)
    #     if search_name=="Бинарный поиск":
    #         plot_Onation_binary(sizes)
    #     if search_name=="Интерполяционный поиск":
    #         plot_Onation_interpolation(sizes)
    #     if search_name == "Экспоненциальный поиск":
    #         plot_Onation_expo(sizes)
    #     if search_name == "Джамп-поиск":
    #         plot_Onation_jump(sizes)

# Запуск программы
if __name__ == "__main__":
    apply_search()