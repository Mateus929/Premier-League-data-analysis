import numpy as np

from models.gradient_descent.gradient_descent_predict import predict as predict_gradient_descent


def process():
    cmd = input('Enter the number: #')
    if cmd == '0':
        return False
    if cmd == '1':
        x = np.zeros(2)
        x[0] = input("Enter Team's possession: ")
        x[1] = input("Enter Team's scored goals: ")
        prediction = predict_gradient_descent(x)
        print(f"Our prediction of xg is {prediction}")
    return True


def main():
    print("Welcome to a fun little game of statistics!")
    print("We've processed data from the last 4-5 Premier League seasons and calculated some interesting (though somewhat meaningless) statistics for the purpose of learning and analysis.")
    print("You can choose one of the statistics below by entering the corresponding number.")
    print("---------------------------------------------------------------------")
    print("#0 Exit")
    print("#1 Method: Gradient Descent | Input: Team's possession, Team's scored goals | Output: Team's xG")
    print("---------------------------------------------------------------------")
    while True:
        if not process():
            break

if __name__ == "__main__":
    main()
