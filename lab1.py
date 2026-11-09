# -*- coding: utf-8 -*-
"""
Лабораторная работа № 1.
Статистическое оценивание числовых характеристик законов распределения
случайных величин.

Вариант 14 (Мартюшова Е. А., группа 4231кз)

Методика — по В.И. Сеньченкову «Статистические методы обработки
экспериментальных данных», § 5.1, 5.2.
Квантили t берутся из Приложения 4 (обратная функция Лапласа):
    t(beta) = arg{ 2*Ф(t) = beta }
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- исходные данные
X = [0.6, 1.1, 2.0, 9.4, 4.5, 6.1, 6.2, 4.7, 2.3, 1.5, 1.0, 0.5]  # табл. 1.1, вариант 14
BETA = 0.82       # доверительная вероятность (табл. 1.2)
EPS_BETA = 0.31   # максимальная вероятная погрешность (табл. 1.2)

# Квантили из Приложения 4 методических указаний (обратная функция Лапласа)
T_TABLE = {0.95: 1.96, 0.82: 1.34}


def laplace(t: float) -> float:
    """Функция Лапласа Ф(t) = P(0 <= Z < t) для стандартного нормального Z."""
    return math.erf(t / math.sqrt(2)) / 2


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    """Несмещённая оценка СКО (деление на n-1)."""
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def report_block(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ------------------------------------------------- 1. Оценка математического ожидания
report_block("1. Оценка математического ожидания (исходный ряд)")
n = len(X)
m = mean(X)
S = std(X)
print(f"n = {n}")
print(f"m* = (1/n)·Σx_i = {sum(X):.4g}/{n} = {m:.4f}")
print(f"S* = sqrt( Σ(x_i − m*)² / (n−1) ) = {S:.4f}")

# ------------------------------ 2. 95% доверительный интервал для случайной величины
report_block("2. 95-процентный доверительный интервал для случайной величины")
t95 = T_TABLE[0.95]
half = t95 * S
lo, hi = m - half, m + half
print(f"t(0,95) = {t95}  (Приложение 4)")
print(f"Интервал: m* ± t·S* = {m:.4f} ± {t95}·{S:.4f} = {m:.4f} ± {half:.4f}")
print(f"          [ {lo:.4f} ; {hi:.4f} ]")

# ------------------------------------------- 3. Отсеивание аномальных наблюдений
report_block("3. Отсеивание аномальных наблюдений")
outliers = [x for x in X if not (lo <= x <= hi)]
clean = [x for x in X if lo <= x <= hi]
if outliers:
    print(f"За пределами интервала (аномальные): {outliers}")
else:
    print("Аномальных наблюдений нет.")
print(f"Оставшийся ряд (n = {len(clean)}): {clean}")

# ------------------------------------------- 4. Уточнённая оценка математического ожидания
report_block("4. Уточнённая оценка после отсеивания")
n1 = len(clean)
m1 = mean(clean)
S1 = std(clean)
print(f"m* = {sum(clean):.4g}/{n1} = {m1:.4f}   (было {m:.4f})")
print(f"S* = {S1:.4f}   (было {S:.4f})")

# --------------------- 5а. Доверительный интервал для мат. ожидания по заданной beta
report_block(f"5а. Доверительный интервал для мат. ожидания при β = {BETA}")
t_b = T_TABLE[BETA]
eps = t_b * S1 / math.sqrt(n1)
print(f"t(β) = t({BETA}) = {t_b}  (Приложение 4)")
print(f"ε = t(β)·S*/√n = {t_b}·{S1:.4f}/√{n1} = {eps:.4f}")
print(f"Интервал для m: {m1:.4f} ± {eps:.4f}  →  [ {m1 - eps:.4f} ; {m1 + eps:.4f} ]")
print(f"С вероятностью β = {BETA} истинное m лежит в этом интервале.")

# --------------------- 5б. Доверительная вероятность по заданной погрешности eps_beta
report_block(f"5б. Доверительная вероятность при εβ = {EPS_BETA}")
t_e = EPS_BETA * math.sqrt(n1) / S1
beta_found = 2 * laplace(t_e)
print(f"t = εβ·√n/S* = {EPS_BETA}·√{n1}/{S1:.4f} = {t_e:.4f}")
print(f"Ф({t_e:.4f}) = {laplace(t_e):.4f}  (функция Лапласа, Приложение 2)")
print(f"β = 2·Ф(t) = {beta_found:.4f} ≈ {beta_found:.2f}")
print(f"С вероятностью ≈ {beta_found:.2f} ошибка оценки m не превысит {EPS_BETA}.")

# ---------------------------------------------------------------------- график
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
ax.scatter(X, [1] * n, c="steelblue", zorder=3, label="Исходные наблюдения")
if outliers:
    ax.scatter(outliers, [1] * len(outliers), facecolors="none",
               edgecolors="red", s=140, linewidths=2, zorder=4,
               label="Аномальное наблюдение")
ax.axvspan(lo, hi, color="steelblue", alpha=0.12,
           label=f"95% интервал для СВ [{lo:.2f}; {hi:.2f}]")
ax.axvline(m, color="steelblue", ls="--", label=f"m* исходное = {m:.3f}")
ax.axvline(m1, color="darkorange", ls="-", lw=2, label=f"m* уточнённое = {m1:.3f}")
ax.axvspan(m1 - eps, m1 + eps, color="darkorange", alpha=0.20,
           label=f"Интервал для m при β={BETA} [{m1 - eps:.2f}; {m1 + eps:.2f}]")
ax.set_yticks([])
ax.set_xlabel("Значение случайной величины")
ax.set_title("ЛР1, вариант 14: отсеивание аномального наблюдения и оценка m")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("lab1_plot.png")
print("\nГрафик сохранён: lab1_plot.png")
