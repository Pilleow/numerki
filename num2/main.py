"""
Zadanie numeryczne NUM 2
Autor: Igor Zamojski

Zadane są macierze [...]. Zdefiniujmy wektor[ [...].
Używając wybranego pakietu algebry komputerowej lub biblioteki numerycznej, rozwiąż równania macie-
rzowe A_i y = b dla i = 1, 2. Ponadto, rozwiąż analogiczne równania z zaburzonym wektorem wyrazów
wolnych, Aiy = b + ∆b. Zaburzenie ∆b wygeneruj jako losowy wektor o małej normie euklidesowej (np.
||∆b|| ≈ 10^(−6)). Przeanalizuj jak wyniki dla macierzy A_1 i A_2 zależą od ∆b i zinterpretuj zaobserwowane
różnice.
"""

import math
import random
import numpy as np
from colorama import Fore, Back, Style

# wartość tej stałej wynika bezpośrednio z równania ||∆b|| ≈ 10^(−6) .
MAX_DISTURBANCE_RAND = 2 * math.sqrt(5) / 5 * (10 ** -6)


def get_norm(v):
    sum_of_squares = 0
    for n in v:
        sum_of_squares += n[0] ** 2
    return math.sqrt(sum_of_squares)


def list_solutions(s_no_err, s_wi_err) -> None:
    errors = [abs(s_wi_err[i] - s_no_err[i]) for i in range(len(s_no_err))]
    print(Fore.BLACK + Back.LIGHTWHITE_EX + "   x1                   x2                   x3                   x4                   x5                   " + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + " - Dokładne wartości: ")
    print("   " + ", ".join(["{:<19}".format(v) for v in s_no_err]))
    print(Fore.LIGHTRED_EX + " - Po wprowadzeniu zaburzenia: ")
    print("   " + ", ".join(["{:<19}".format(v) for v in s_wi_err]))
    print(Fore.LIGHTBLUE_EX + " - Błąd poszczególnych wartości:")
    print("   " + ", ".join(["{:<19}".format(e) for e in errors]))
    print(" - Suma błędów: " + Fore.BLACK + Back.LIGHTWHITE_EX + " " + str(sum(errors)) + " " + Style.RESET_ALL)


def get_solutions(matrix_list, const_b) -> list:
    solutions = []
    for i in range(len(matrix_list)):
        try:
            solution = np.linalg.solve(matrix_list[i], const_b)
        except np.linalg.LinAlgError:
            print("LinAlgError caught while computing solution.")
            raise
        solutions.append([s[0] for s in solution.A])
    return solutions


def main():
    # wczytaj macierze A1, A2 i wektor b
    a1 = np.matrix([
        [2.554219275, 0.871733993, 0.052575899, 0.240740262, 0.316022841],
        [0.871733993, 0.553460938, -0.070921727, 0.255463951, 0.707334556],
        [0.052575899, -0.070921727, 3.409888776, 0.293510439, 0.847758171],
        [0.240740262, 0.255463951, 0.293510439, 1.108336850, -0.206925123],
        [0.316022841, 0.707334556, 0.847758171, -0.206925123, 2.374094162]
    ])
    a2 = np.matrix([
        [2.645152285, 0.544589368, 0.009976745, 0.327869824, 0.424193304],
        [0.544589368, 1.730410927, 0.082334875, -0.057997220, 0.318175706],
        [0.009976745, 0.082334875, 3.429845092, 0.252693077, 0.797083832],
        [0.327869824, -0.057997220, 0.252693077, 1.191822050, -0.103279098],
        [0.424193304, 0.318175706, 0.797083832, -0.103279098, 2.502769647]
    ])
    b = np.matrix([-0.642912346, -1.408195475, 4.595622394, -5.073473196, 2.178020609])
    a = [a1, a2]

    # rozwiąż a_i y = b dla macierzy a1, a2 oraz DOKŁADNEJ wartości b
    solutions_no_error = get_solutions(a, b.T)

    # wprowadź zaburzenie do wektora wyrazów wolnych b
    delta_b = np.matrix([random.uniform(-MAX_DISTURBANCE_RAND, MAX_DISTURBANCE_RAND) for _ in range(5)])
    b_disturbed = b + delta_b

    # rozwiąż a_i y = b dla macierzy a1, a2 oraz ZABURZONEJ wartości b
    solutions_with_error = get_solutions(a, b_disturbed.T)

    # wypisz wyniki
    print("\n\nDla A1 y = b:")
    list_solutions(solutions_no_error[0], solutions_with_error[0])
    print(Fore.BLACK + Back.LIGHTWHITE_EX + "   - WSKAŹNIK UWARUNKOWANIA macierzy A1 = " + str(int(np.linalg.cond(a1))) + " "  + Style.RESET_ALL)
    print("Wskaźnik uwarunkowania macierzy A1 jest bardzo duży, zatem rozwiązanie \nukładu liniowego macierzy A1 jest zatem numerycznie źle uwarunkowane!")

    print("\nDla A2 y = b:")
    list_solutions(solutions_no_error[1], solutions_with_error[1])
    print(Fore.BLACK + Back.LIGHTWHITE_EX + "   - WSKAŹNIK UWARUNKOWANIA macierzy A2 = " + str(int(np.linalg.cond(a2))) + " " + Style.RESET_ALL)
    print("Wskaźnik uwarunkowania macierzy A2 jest o wiele mniejszy od A1, zatem \nrozwiązanie układu liniowego macierzy A2 jest numerycznie dobrze uwarunkowane.")

    print("\nMacierz zaburzenia ∆b:")
    print("   [" + ", ".join(["{:<19}".format(v[0]) for v in delta_b.T.A]) + "]")
    print("   Norma ||∆b|| = " + np.format_float_scientific(get_norm(delta_b.T.A)) + "\n\n")


if __name__ == "__main__":
    main()
