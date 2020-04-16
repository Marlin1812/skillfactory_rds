import numpy as np # Import numpy and write np.random
def game_core_v2(number):
    count = 1
    min_value = 1
    max_value = 101
    predict =  (min_value + max_value) // 2 # calculation algorithm
    while number != predict:
        count+=1
        if number > predict:
            min_value = predict
        elif number < predict:
            max_value = predict
        predict =  (min_value + max_value) // 2 # calculation algorithm
    return(count)
def score_game(game_core_v2):
    count_2 = []
    np.random.seed(1)  # fix to self-reproduce
    random_array = np.random.randint(1, 101, size=(1000))
    for number in random_array:
        count_2.append(game_core_v2(number))
    score = int(np.mean(count_2))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return(score)
score_game(game_core_v2 = game_core_v2)