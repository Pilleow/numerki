"""
Zadanie numeryczne NUM 1
Autor: Igor Zamojski

Napisz program wyliczający przybliżenie pochodnej ze wzorów: [...]
Przeanalizuj, jak zachowuje się błąd |Dhf(x) − f′(x)| dla funkcji f(x) = sin(x^2) oraz punktu x = 0.2 przy
zmianie parametru h dla różnych typów zmiennoprzecinkowych (float, double). Wykreśl |Dhf (x) − f ′(x)|
w funkcji h w skali logarytmicznej. Poeksperymentuj również używając innych funkcji i punktów.
"""

import math
import matplotlib.pyplot as plt


def plot(x, y, title, desc, xlabel, ylabel, yscale="linear", block=True, vlines=[], mainLabel=""):
    # niezbędne ustawienia
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, label=mainLabel)
    plt.yscale(yscale)
    plt.grid(True)

    # dodawanie tekstu
    plt.suptitle(title)
    plt.title(desc, fontsize=9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for l in vlines:
        plt.axvline(x=l[0], color=l[1], label=l[2])

    plt.legend()
    plt.show(block=block)


def df(f, x, h=None):
    """
    Oblicza przybliżenie pochodnej funkcji lambda f w punkcie x
    ze wzoru na pochodną, przyjmując względnie małe h (domyślnie 2 ^ -16)
    """
    if h is None:
        h = 2 ** -32
    return (f(x + h) - f(x)) / h


def main():
    # ustawienie funkcji, pochodnej, x
    f = lambda x: math.sin(x ** 2)
    true_df = lambda x: 2 * x * math.cos(x ** 2)
    x0 = 0.2

    # analiza błędu dla kolejnych wartości h
    err_list = []
    for i in range(2, 32):
        h = 2.0 ** -i
        d = df(f, x0, h)
        err = abs(d - true_df(x0))
        err_list.append(err)
        print("Df = {0:<22} | h = 2 ^ -{1:<4} = {2:<22} | E = {3:<22}".format(d, i, h, err))

    # wykreślenie błędu w funkcji h w skali logarytmicznej
    plot(
        [f"-{i}" for i in range(2, 32)],
        err_list,
        "Błąd algorytmu dla kolejnych wartości h.",
        "Błąd algorytmu maleje proporcjonalnie do malejącego h. Przy wartości 2^-28 wykres przestaje maleć,\nwynika to z błędu zaokrąglenia zmiennej. Dalsze iteracje h nie mają sensu, ponieważ błąd zaokrąglenia błyskawicznie rośnie.",
        "h = 2 ^ n",
        "błąd |df(x) - f'(x)|",
        "log",
        vlines=[[26, "red", "maksymalna precyzja zmiennoprzecinkowa wartości h"]],
        mainLabel="błąd algorytmu"
    )


if __name__ == "__main__":
    main()
